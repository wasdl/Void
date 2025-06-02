"""
DBSCAN 군집화 클래스
"""
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from app.utils.helpers import ensure_dir, normalize_vector
from app.config import get_paths_dict

class DBSCAN:
    """
    DBSCAN 클러스터링 구현 클래스 (DQM 문서 기준)
    """
    def __init__(self):
        """
        DQM 기준 DBSCAN 파라미터로 초기화
        """
        # DQM 문서 기준 파라미터
        self.eps = 0.9
        self.min_samples = 3
        self.silhouette_threshold = 0.6
        self.unknown_ratio_max = 0.05
        self.min_valid_users = 500
        
        # 파라미터 조정 범위 (DQM 기준)
        self.eps_range = (0.3, 1.5) 
        self.min_samples_range = (3, 10)
        
        # 최대 재시도 횟수
        self.max_retries = 5
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'dbscan_params_history': {},
            'success_clusters': {}
        }
        
        # 경로 설정
        self.paths = get_paths_dict()
    
    def fit_transform(self, X, validate=True, visualize=False, save_dir=None):
        """
        DBSCAN 군집화 실행 및 검증
        
        Args:
            X: 군집화할 데이터 (normalized)
            validate: 검증 실행 여부
            visualize: 시각화 실행 여부
            save_dir: 결과 저장 경로
            
        Returns:
            dict: 처리 결과
        """
        self.pipeline_status['current_stage'] = 'fit_transform_started'
        
        # 데이터 검증
        if len(X) < self.min_valid_users:
            self.pipeline_status['current_stage'] = 'insufficient_data'
            self.pipeline_status['last_error'] = f"데이터 수가 부족합니다: {len(X)} < {self.min_valid_users}"
            return {
                "status": "warning",
                "message": f"데이터 수가 부족합니다: {len(X)} < {self.min_valid_users}",
                "labels": np.zeros(len(X), dtype=int),
                "pipeline_status": self.pipeline_status
            }
        
        # DBSCAN 실행 (파라미터 자동 조정)
        result = self._run_with_validation(X)
        
        if result["status"] == "error":
            self.pipeline_status['current_stage'] = 'dbscan_failed'
            self.pipeline_status['last_error'] = result["error"]
            return result
        
        labels = result["labels"]
        silhouette_score = result["silhouette_score"]
        valid_count = result["valid_count"]
        valid_ratio = result["valid_ratio"]
        
        # 시각화 (요청된 경우)
        if visualize:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            self._visualize_clusters(X_pca, labels, save_dir)
        
        # 검증 결과
        self.pipeline_status['current_stage'] = 'success'
        self.pipeline_status['success_clusters'] = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'silhouette_score': silhouette_score,
            'valid_count': valid_count,
            'valid_ratio': valid_ratio
        }
        
        return {
            "status": "success",
            "message": f"DBSCAN 군집화 완료: {valid_count}개 데이터 할당 ({valid_ratio:.2%})",
            "labels": labels,
            "silhouette_score": silhouette_score,
            "valid_ratio": valid_ratio,
            "valid_count": valid_count,
            "dbscan_params": {
                "eps": self.eps,
                "min_samples": self.min_samples
            },
            "pipeline_status": self.pipeline_status
        }
    
    def _run_with_validation(self, X):
        """
        검증이 포함된 DBSCAN 실행
        
        Args:
            X: 군집화할 데이터
            
        Returns:
            dict: 군집화 결과
        """
        # DBSCAN 파라미터 최적화
        for attempt in range(self.max_retries):
            # 현재 파라미터로 DBSCAN 실행
            dbscan = SklearnDBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(X)
            
            # 파라미터 기록
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}'] = {
                'eps': self.eps,
                'min_samples': self.min_samples
            }
            
            # 유효 클러스터 데이터
            valid_indices = labels != -1
            valid_ratio = np.sum(valid_indices) / len(X)
            valid_count = np.sum(valid_indices)
            
            # 유효 데이터가 없는 경우
            if valid_count == 0:
                self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['result'] = 'no_clusters'
                self.eps = min(self.eps * 1.2, self.eps_range[1])  # eps 증가
                continue
            
            # 실루엣 점수 계산 (유효 클러스터만)
            valid_X = X[valid_indices]
            valid_labels = labels[valid_indices]
            
            # 클러스터가 1개인 경우 실루엣 점수를 계산할 수 없음
            if len(np.unique(valid_labels)) < 2:
                avg_silhouette = 0
            else:
                avg_silhouette = silhouette_score(valid_X, valid_labels)
            
            # 결과 기록
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}'].update({
                'silhouette_score': avg_silhouette,
                'valid_ratio': valid_ratio,
                'valid_count': valid_count
            })
            
            # DQM 기준에 따른 검증
            silhouette_valid = avg_silhouette >= self.silhouette_threshold
            users_count_valid = valid_count >= self.min_valid_users or valid_ratio >= (1 - self.unknown_ratio_max)
            
            self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}'].update({
                'silhouette_valid': silhouette_valid,
                'users_count_valid': users_count_valid
            })
            
            # 검증 통과
            if silhouette_valid and users_count_valid:
                return {
                    "status": "success",
                    "labels": labels,
                    "silhouette_score": avg_silhouette,
                    "valid_ratio": valid_ratio,
                    "valid_count": valid_count
                }
            
            # 실루엣 점수가 낮은 경우 파라미터 조정
            if not silhouette_valid:
                self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['result'] = 'low_silhouette'
                if self.min_samples < self.min_samples_range[1]:
                    self.min_samples += 1  # min_samples 증가 (더 응집된 클러스터)
                else:
                    self.eps = max(self.eps * 0.9, self.eps_range[0])  # eps 감소 (더 엄격한 군집화)
            
            # 유효 사용자 비율/수가 낮은 경우 파라미터 조정
            if not users_count_valid:
                self.pipeline_status['dbscan_params_history'][f'attempt_{attempt+1}']['result'] = 'low_user_count'
                self.eps = min(self.eps * 1.1, self.eps_range[1])  # eps 증가 (더 느슨한 군집화)
                if self.min_samples > self.min_samples_range[0]:
                    self.min_samples -= 1  # min_samples 감소 (더 많은 클러스터 허용)
        
        # 최대 시도 횟수 초과 - 마지막 파라미터로 실행
        dbscan = SklearnDBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X)
        
        valid_indices = labels != -1
        valid_ratio = np.sum(valid_indices) / len(X)
        valid_count = np.sum(valid_indices)
        
        if valid_count > 0:
            valid_X = X[valid_indices]
            valid_labels = labels[valid_indices]
            
            if len(np.unique(valid_labels)) < 2:
                avg_silhouette = 0
            else:
                avg_silhouette = silhouette_score(valid_X, valid_labels)
        else:
            avg_silhouette = 0
        
        # 최대 재시도 초과 기록
        self.pipeline_status['current_stage'] = 'max_retries_exceeded'
        self.pipeline_status['last_error'] = f"최대 재시도 횟수({self.max_retries})를 초과했습니다."
        
        return {
            "status": "warning", 
            "message": f"최적 파라미터를 찾지 못했지만 마지막 파라미터로 진행합니다. (eps={self.eps}, min_samples={self.min_samples})",
            "labels": labels,
            "silhouette_score": avg_silhouette,
            "valid_ratio": valid_ratio,
            "valid_count": valid_count
        }
    
    def _visualize_clusters(self, X_pca, labels, save_dir=None):
        """클러스터 시각화"""
        plt.figure(figsize=(10, 8))
        
        # 노이즈 포인트
        noise = X_pca[labels == -1]
        if len(noise) > 0:
            plt.scatter(noise[:, 0], noise[:, 1], c='lightgray', marker='x', alpha=0.5, label='Noise')
        
        # 각 클러스터 포인트
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster = X_pca[labels == label]
            plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.8, marker='o', label=f'Cluster {label}')
        
        plt.title(f'DBSCAN Clustering (eps={self.eps}, min_samples={self.min_samples})')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            ensure_dir(save_dir)
            plt.savefig(os.path.join(save_dir, 'dbscan_clusters.png'))
            plt.close()
        else:
            plt.show()
    
    def get_pipeline_status(self):
        """파이프라인 상태 반환"""
        return self.pipeline_status
