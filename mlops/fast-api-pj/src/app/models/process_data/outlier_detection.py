# 결과는 https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from dtaidistance import dtw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from app.models.classifier.dbscan import DBSCAN as MLOPSDBSCAN
from app.utils.helpers import ensure_dir, normalize_vector

class OutlierDetection:
    def __init__(self):
        """
        이상치 감지 및 제거 클래스 초기화
        """
        # DBSCAN 파라미터 범위 설정 (DQM 문서 기준)
        self.eps_range = (0.3, 1.5)  # DQM 문서 기준
        self.min_samples_range = (3, 10)
        self.default_eps = 0.9  # DQM 문서 기준: eps=0.9
        self.default_min_samples = 3  # DQM 문서 기준: min_samples=3
        
        # 검증 임계값 설정 (DQM 문서 기준)
        self.silhouette_threshold = 0.6  # DQM 문서 기준: 실루엣 점수 ≥ 0.6
        self.unknown_ratio_max = 0.05  # DQM 문서 기준: 5%
        self.min_valid_users = 500  # DQM 문서 기준: 사용자 수 ≥ 500명
        
        # IQR 이상치 파라미터 (DQM 문서 기준)
        self.iqr_factor = 0.5  # DQM 문서 기준: Q3 + 0.5 * IQR
        
        # 재시도 관련 설정
        self.max_retries = 5
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'dbscan_params_history': {},
            'success_clusters': {},
            'needs_adjustment': False
        }
    
    def cumulative_minmax(self, vec):
        """
        누적합 정규화 함수
        """
        cum = np.cumsum(vec)
        vmin, vmax = cum.min(), cum.max()
        return (cum - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(cum)

    def normalize_each_day_minmax(self, vectors):
        """
        일일 벡터 정규화 함수
        """
        normalized = []
        for vec in vectors:
            normalized.append(normalize_vector(vec))
        return np.array(normalized)

    def process_pattern_dbscan(self, df, file_path, label='', visualize=False):
        """
        DBSCAN을 사용한 패턴 처리 및 이상치 제거 함수
        
        Args:
            df: 처리할 DataFrame
            file_path: 결과 저장 경로
            label: 패턴 라벨 (A, B, C, D 등)
            visualize: 시각화 여부 (기본값: False)
            
        Returns:
            dict: 처리 결과
        """
        # 파이프라인 상태 초기화
        self.pipeline_status = {
            'current_stage': f'processing_pattern_{label}',
            'last_error': None,
            'dbscan_params_history': {},
            'success_clusters': {},
            'needs_adjustment': False
        }
        
        try:
            df['date'] = pd.to_datetime(df['date'])

            # 하루 단위 벡터 생성
            daily_vectors = []
            person_day_mapping = []  # 각 벡터가 어떤 사람/날짜인지 매핑
            
            for (pid, date), group in df.groupby(['person_id', 'date']):
                sorted_group = group.sort_values(by=['hour', 'minute'])
                vec = sorted_group['amount'].values
                if len(vec) == 48:
                    daily_vectors.append(vec)
                    person_day_mapping.append((pid, date))
                    
            daily_vectors = np.array(daily_vectors)

            if len(daily_vectors) == 0:
                self.pipeline_status['current_stage'] = 'no_valid_vectors'
                self.pipeline_status['last_error'] = f"[{label}] 유효한 일일 벡터가 없습니다."
                return {"status": "error", "error": f"[{label}] 유효한 일일 벡터가 없습니다.", "pipeline_status": self.pipeline_status}

            # 정규화
            daily_vectors_norm = self.normalize_each_day_minmax(daily_vectors)
            
            # DBSCAN 군집화 - 자동 파라미터 조정 및 검증
            result = self._run_dbscan_with_validation(daily_vectors_norm, label)
            
            if result["status"] == "error":
                self.pipeline_status['current_stage'] = 'dbscan_failed'
                self.pipeline_status['last_error'] = result["error"]
                self.pipeline_status['needs_adjustment'] = True
                return {**result, "pipeline_status": self.pipeline_status}
                
            labels = result["labels"]
            dbscan_eps = result["eps"]
            dbscan_min_samples = result["min_samples"]
            silhouette_score = result["silhouette_score"]
            
            # DQM 문서 기준: 성공적인 DBSCAN 파라미터 저장
            self.pipeline_status['success_clusters'][label] = {
                'eps': dbscan_eps,
                'min_samples': dbscan_min_samples,
                'silhouette_score': silhouette_score
            }
            
            # PCA 시각화를 위한 변환
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(daily_vectors_norm)

            # 유효 벡터만 (노이즈 제외)
            valid_indices = labels != -1
            valid_vectors = daily_vectors_norm[valid_indices]
            valid_labels = labels[valid_indices]

            if len(valid_vectors) == 0:
                self.pipeline_status['current_stage'] = 'no_valid_clusters'
                self.pipeline_status['last_error'] = f"[{label}] 클러스터가 발견되지 않았습니다."
                self.pipeline_status['needs_adjustment'] = True
                return {"status": "error", "error": f"[{label}] 클러스터가 발견되지 않았습니다.", "pipeline_status": self.pipeline_status}
                
            # 유효 사용자 비율 계산
            valid_ratio = np.sum(valid_indices) / len(daily_vectors_norm)
            
            # DQM 기준: 사용자 비율 또는 수가 임계값 미만인 경우 실패
            if valid_ratio < (1 - self.unknown_ratio_max) and np.sum(valid_indices) < self.min_valid_users:
                self.pipeline_status['current_stage'] = 'insufficient_valid_users'
                self.pipeline_status['last_error'] = f"[{label}] 유효 사용자 비율({valid_ratio:.2%})과 수({np.sum(valid_indices)})가 모두 임계값보다 낮습니다."
                self.pipeline_status['needs_adjustment'] = True
                return {
                    "status": "error", 
                    "error": f"[{label}] 유효 사용자 비율이 낮습니다: {valid_ratio:.2%}, 최소 기준: {1 - self.unknown_ratio_max:.2%}",
                    "pipeline_status": self.pipeline_status
                }

            # 시각화 부분 (visualize=True인 경우에만 실행)
            if visualize:
                # ------------------- (1) Silhouette Score 계산 -------------------
                sil_samples = silhouette_samples(valid_vectors, valid_labels)
                sil_scores = {
                    cluster_id: np.mean(sil_samples[valid_labels == cluster_id])
                    for cluster_id in np.unique(valid_labels)
                }

                # 표로 출력
                sil_table = pd.DataFrame({
                    'Cluster': list(sil_scores.keys()),
                    'Silhouette Score': list(sil_scores.values())
                }).sort_values(by='Silhouette Score', ascending=False)

                print(f"\n[{label}] Cluster-wise Silhouette Scores:")
                print(sil_table)

            # 가장 응집도 높은 클러스터 찾기
            sil_samples = silhouette_samples(valid_vectors, valid_labels)
            sil_scores = {
                cluster_id: np.mean(sil_samples[valid_labels == cluster_id])
                for cluster_id in np.unique(valid_labels)
            }
            
            # 표로 정리
            sil_table = pd.DataFrame({
                'Cluster': list(sil_scores.keys()),
                'Silhouette Score': list(sil_scores.values())
            }).sort_values(by='Silhouette Score', ascending=False)
            
            best_cluster = sil_table.iloc[0]['Cluster']
            best_vectors = valid_vectors[valid_labels == best_cluster]

            # 가장 대표적인 실제 벡터 찾기
            dtw_matrix = np.zeros((len(best_vectors), len(best_vectors)))
            for i in range(len(best_vectors)):
                for j in range(i + 1, len(best_vectors)):
                    dist = dtw.distance(best_vectors[i], best_vectors[j])
                    dtw_matrix[i, j] = dist
                    dtw_matrix[j, i] = dist
            avg_dtw = dtw_matrix.mean(axis=1)
            center_idx = np.argmin(avg_dtw)
            center_vector = best_vectors[center_idx]

            # 전체 데이터에 대한 DTW 계산
            cum_center = self.cumulative_minmax(center_vector)
            dtw_cum_distances = [dtw.distance(cum_center, self.cumulative_minmax(vec)) for vec in daily_vectors_norm]
            dtw_plain_distances = [dtw.distance(center_vector, vec) for vec in daily_vectors_norm]

            # 시각화
            if visualize:
                self._visualize_clusters(pca_result, labels, best_cluster, label)
                self._visualize_representative_pattern(center_vector, label)
                self._visualize_dtw_histograms(dtw_cum_distances, dtw_plain_distances, label)

            # ------------------- 이상치 제거 (IQR 방식) -------------------
            q1 = np.percentile(dtw_cum_distances, 25)
            q3 = np.percentile(dtw_cum_distances, 75)
            iqr = q3 - q1
            upper_bound = q3 + self.iqr_factor * iqr

            # 이상치 마스크 및 인덱스 추출
            outlier_mask = np.array(dtw_cum_distances) > upper_bound

            # 시각화
            if visualize:
                self._visualize_outliers(pca_result, outlier_mask, label)

            # 결과 출력
            if visualize:
                print(f"[{label}] IQR 기준 이상치 개수: {outlier_mask.sum()}개 / 전체 {len(dtw_cum_distances)}개")

            # 이상치 제외한 데이터 추출
            non_outlier_indices = np.where(~outlier_mask)[0]
            filtered_vectors = daily_vectors[non_outlier_indices]  # 원본 벡터 사용 (정규화 X)
            
            # 각 벡터가 어떤 사람/날짜에 해당하는지 매칭
            person_day_info = []
            for idx in non_outlier_indices:
                pid, date = person_day_mapping[idx]
                # 해당 사람/날짜의 데이터 찾기
                day_data = df[(df['person_id'] == pid) & (df['date'] == date)]
                if len(day_data) > 0:
                    start_hours = day_data['start_hours'].iloc[0]
                    eating_hours = day_data['eating_hours'].iloc[0]
                    counts = day_data['counts'].iloc[0]
                    person_day_info.append((pid, date, start_hours, eating_hours, counts))

            # 저장용 데이터프레임 생성
            records = []
            for (pid, date, start_hours, eating_hours, counts), vec in zip(person_day_info, filtered_vectors):
                for i, amount in enumerate(vec):
                    hour = i // 2
                    minute = (i % 2) * 30
                    records.append({
                        'person_id': pid,
                        'date': date,
                        'hour': hour,
                        'minute': minute,
                        'amount': amount,
                        'start_hours': start_hours,
                        'eating_hours': eating_hours,
                        'counts': counts
                    })

            df_filtered = pd.DataFrame(records)
            
            # 저장 경로 설정
            raw_path = os.path.join(file_path, 'raw')
            os.makedirs(raw_path, exist_ok=True)
            
            # 저장
            save_path = os.path.join(raw_path, f"filtered_pattern_{label}.csv")
            df_filtered.to_csv(save_path, index=False)
            
            if visualize:
                print(f"[{label}] 이상치 제거된 데이터 저장 완료 → filtered_pattern_{label}.csv")

            # 파이프라인 상태 업데이트
            self.pipeline_status['current_stage'] = 'success'
            
            return {
                "status": "success",
                "message": f"패턴 {label} 이상치 처리 완료",
                "outlier_count": outlier_mask.sum(),
                "total_count": len(dtw_cum_distances),
                "silhouette_score": silhouette_score,
                "valid_ratio": valid_ratio,
                "dbscan_eps": dbscan_eps,
                "dbscan_min_samples": dbscan_min_samples,
                "file_path": save_path,
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'error'
            self.pipeline_status['last_error'] = str(e)
            self.pipeline_status['needs_adjustment'] = True
            
            return {
                "status": "error",
                "error": f"이상치 처리 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
            
    def _run_dbscan_with_validation(self, data, label=''):
        """
        DBSCAN 군집화를 실행하고 검증하는 함수
        
        Args:
            data: 군집화할 데이터
            label: 패턴 라벨
            
        Returns:
            dict: 검증 결과
        """
        # 초기 파라미터 설정
        eps = self.default_eps  # DQM 문서 기준: eps=0.9
        min_samples = self.default_min_samples  # DQM 문서 기준: min_samples=3
        
        for attempt in range(self.max_retries):
            # DBSCAN 실행
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            
            # 현재 시도 정보 기록
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}'] = {
                'eps': eps,
                'min_samples': min_samples
            }
            
            # 클러스터 유효성 검증
            valid_indices = labels != -1
            valid_ratio = np.sum(valid_indices) / len(data)
            valid_count = np.sum(valid_indices)
            
            # DQM 기준: 사용자 수가 너무 적은 경우
            if valid_count == 0:
                # eps 값 증가하여 재시도
                eps = min(eps * 1.2, self.eps_range[1])
                self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['result'] = 'no_clusters'
                continue
                
            # 유효 벡터만 추출
            valid_vectors = data[valid_indices]
            valid_labels = labels[valid_indices]
            
            # 실루엣 점수 계산
            sil_samples = silhouette_samples(valid_vectors, valid_labels)
            avg_silhouette = np.mean(sil_samples)
            
            # 현재 시도 결과 기록
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['silhouette_score'] = avg_silhouette
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['valid_ratio'] = valid_ratio
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['valid_count'] = valid_count
            
            # 1. DQM 기준: 실루엣 점수 검증 (≥ 0.6)
            silhouette_valid = avg_silhouette >= self.silhouette_threshold
            
            # 2. DQM 기준: 유효 사용자 수/비율 검증 (≥ 500명 또는 ≥ 95%)
            users_count_valid = valid_count >= self.min_valid_users or valid_ratio >= (1 - self.unknown_ratio_max)
            
            # 검증 결과 기록
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['silhouette_valid'] = silhouette_valid
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['users_count_valid'] = users_count_valid
            
            # DQM 기준: 모든 검증 통과
            if silhouette_valid and users_count_valid:
                return {
                    "status": "success",
                    "labels": labels,
                    "eps": eps,
                    "min_samples": min_samples,
                    "silhouette_score": avg_silhouette,
                    "valid_ratio": valid_ratio,
                    "valid_count": valid_count,
                    "attempt": attempt + 1
                }
            
            # 실루엣 점수가 낮은 경우
            if not silhouette_valid:
                # min_samples 증가 (더 응집도 높은 클러스터)
                if min_samples < self.min_samples_range[1]:
                    min_samples += 1
                else:
                    # eps 감소 (더 엄격한 군집화)
                    eps = max(eps * 0.9, self.eps_range[0])
                
                self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['result'] = 'low_silhouette'
            
            # 유효 사용자 수/비율이 낮은 경우
            if not users_count_valid:
                # eps 증가 (더 느슨한 군집화)
                eps = min(eps * 1.1, self.eps_range[1])
                # min_samples 감소 (더 많은 클러스터 허용)
                if min_samples > self.min_samples_range[0]:
                    min_samples -= 1
                    
                self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['result'] = 'low_user_count'
        
        # DQM 기준: 최대 재시도 횟수를 초과해도 검증을 통과하지 못한 경우
        # 마지막 파라미터로 실행한 결과 반환
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)
        
        valid_indices = labels != -1
        if np.sum(valid_indices) == 0:
            return {
                "status": "error",
                "error": f"[{label}] 최대 재시도 횟수({self.max_retries})를 초과해도 유효한 클러스터를 찾지 못했습니다."
            }
            
        valid_vectors = data[valid_indices]
        valid_labels = labels[valid_indices]
        avg_silhouette = np.mean(silhouette_samples(valid_vectors, valid_labels))
        valid_ratio = np.sum(valid_indices) / len(data)
        
        # 파이프라인에서 DBSCAN 파라미터 조정이 필요함을 표시
        self.pipeline_status['needs_adjustment'] = True
        
        return {
            "status": "warning", 
            "message": f"최적화된 파라미터를 찾지 못했지만, 마지막 파라미터로 진행합니다. (eps={eps}, min_samples={min_samples})",
            "labels": labels,
            "eps": eps,
            "min_samples": min_samples,
            "silhouette_score": avg_silhouette,
            "valid_ratio": valid_ratio,
            "valid_count": np.sum(valid_indices),
            "attempt": self.max_retries
        }

    def process_multi_patterns(self, path, patterns=None):
        """
        여러 패턴에 대해 이상치 처리를 수행합니다.
        
        Args:
            path: 데이터 파일 경로
            patterns: 처리할 패턴 리스트 (기본값: ['A', 'B', 'C', 'D'])
            
        Returns:
            dict: 처리 결과
        """
        if patterns is None:
            patterns = ['A', 'B', 'C', 'D']
            
        results = {}
        raw_path = os.path.join(path, 'raw')
        needs_pipeline_adjustment = False
        
        for pattern in patterns:
            file_path = os.path.join(raw_path, f"water_consumption_2022_02_01_{pattern}.csv")
            if not os.path.exists(file_path):
                results[pattern] = {
                    "status": "warning",
                    "message": f"패턴 {pattern} 파일이 존재하지 않습니다."
                }
                continue
                
            try:
                df = pd.read_csv(file_path)
                result = self.process_pattern_dbscan(df, path, pattern, visualize=False)
                results[pattern] = result
                
                # DQM 기준: 하나라도 파이프라인 조정이 필요한 경우 플래그 설정
                if result.get("status") == "error" or result.get("pipeline_status", {}).get("needs_adjustment", False):
                    needs_pipeline_adjustment = True
                    
            except Exception as e:
                results[pattern] = {
                    "status": "error",
                    "error": f"패턴 {pattern} 처리 중 오류 발생: {str(e)}"
                }
                needs_pipeline_adjustment = True
        
        # 전체 결과 요약
        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        
        return {
            "status": "warning" if needs_pipeline_adjustment else "success",
            "message": f"{success_count}/{len(patterns)} 패턴 이상치 처리 완료" + 
                      (" - 파이프라인 조정 필요" if needs_pipeline_adjustment else ""),
            "needs_pipeline_adjustment": needs_pipeline_adjustment,
            "details": results
        }
        
    def get_pipeline_status(self):
        """파이프라인 상태 반환"""
        return self.pipeline_status

    def _visualize_clusters(self, pca_result, labels, best_cluster, label):
        """클러스터 시각화"""
        plt.figure(figsize=(8, 6))
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            idx = labels == lab
            label_name = f'Cluster {lab}' if lab != -1 else 'Noise'
            color = 'red' if lab == best_cluster else None
            plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=label_name, alpha=0.5, c=color)

        plt.title(f"[{label}] PCA of DBSCAN Clusters (Best Cluster Highlighted)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _visualize_representative_pattern(self, center_vector, label):
        """대표 패턴 시각화"""
        plt.figure(figsize=(8, 3))
        plt.plot(center_vector, label='Representative Pattern', linewidth=2)
        plt.title(f"[{label}] Most Representative Pattern")
        plt.xlabel("Time Slot")
        plt.ylabel("Normalized Amount")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _visualize_dtw_histograms(self, dtw_cum_distances, dtw_plain_distances, label):
        """DTW 거리 히스토그램 시각화"""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(dtw_cum_distances, bins=40, color='orange')
        plt.title(f"[{label}] DTW (Cumulative) to Representative (ALL Data)")
        plt.xlabel("DTW Distance")
        plt.ylabel("Count")

        plt.subplot(1, 2, 2)
        plt.hist(dtw_plain_distances, bins=40, color='skyblue')
        plt.title(f"[{label}] DTW (Plain) to Representative (ALL Data)")
        plt.xlabel("DTW Distance")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    def _visualize_outliers(self, pca_result, outlier_mask, label):
        """이상치 시각화"""
        outlier_indices = np.where(outlier_mask)[0]

        plt.figure(figsize=(8, 6))

        # ① 전체 점 (연한 파랑)
        plt.scatter(pca_result[:, 0], pca_result[:, 1],
                    alpha=0.3, color='skyblue', label='All Data')

        # ② 이상치 (선명한 빨강 + X 마커)
        if len(outlier_indices) > 0:
            plt.scatter(pca_result[outlier_indices, 0], pca_result[outlier_indices, 1],
                        color='red', marker='x', s=70, label='DTW Outliers (IQR)')

        plt.title(f"[{label}] PCA with DTW Outliers Highlighted")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 이전 버전과의 호환성을 위한 함수
def process_pattern_dbscan(df, file_path, label=''):
    """
    기존 함수 형태 유지 (호환성)
    """
    outlier_detector = OutlierDetection()
    return outlier_detector.process_pattern_dbscan(df, file_path, label, visualize=True)



