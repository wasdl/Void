# https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing
import pandas as pd
import numpy as np
import os
import json
import random
from datetime import datetime, timedelta
from app.models.process_data.outlier_detection import OutlierDetection
from dtaidistance import dtw


class MakingData:
    def __init__(self, base_path=None):
        self.pattern_registry = {}  # 패턴 레지스트리 초기화
        self.base_patterns = ['A', 'B', 'C', 'D']  # 기본 패턴 유형
        self.outlier_detector = OutlierDetection()  # 이상치 감지 클래스 인스턴스
        self.data_path = base_path
        
    def set_data_path(self, path):
        """데이터 경로 설정"""
        self.data_path = path
        
    def _load_pattern_registry(self, path):
        """
        패턴 레지스트리 로드 또는 초기화
        """
        registry_path = os.path.join(path, 'pattern_registry.json')
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    self.pattern_registry = json.load(f)
                return {"status": "success", "message": "패턴 레지스트리 로드 완료"}
            except Exception as e:
                self.pattern_registry = {}
                return {"status": "error", "error": f"패턴 레지스트리 로드 실패: {str(e)}"}
        else:
            # 디렉토리가 없으면 생성
            os.makedirs(path, exist_ok=True)
            self.pattern_registry = {}
            return {"status": "success", "message": "새 패턴 레지스트리 초기화"}
            
    def _save_pattern_registry(self, path):
        """
        패턴 레지스트리 저장
        """
        registry_path = os.path.join(path, 'pattern_registry.json')
        try:
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.pattern_registry, f, ensure_ascii=False, indent=2)
            return {"status": "success", "message": "패턴 레지스트리 저장 완료"}
        except Exception as e:
            return {"status": "error", "error": f"패턴 레지스트리 저장 실패: {str(e)}"}
            
    def _check_pattern_exists(self, pattern_name):
        """
        패턴 이름이 이미 존재하는지 확인
        """
        # 파일 기반 체크
        pattern_files = [f for f in os.listdir(self.data_path) 
                        if f.endswith(f'_{pattern_name}_30days.csv')]
        if len(pattern_files) > 0:
            return True
        
        # 레지스트리 기반 체크
        return pattern_name in self.pattern_registry
    
    def generate_human_like_day_strong_pattern(self, pattern_type):
        """기본 패턴(A, B, C)이나 커스텀 패턴에 대한 데이터 생성"""
        usage = np.zeros(48)

        meta_info = {
            'count': [],
            'start_hours': [],
            'eating_hours': []
        }

        if pattern_type == 'A':
            start_hour2 = int(np.random.uniform(34, 38))
            eating_hour2 = int(np.random.uniform(6, 10))
            boost_hours = [list(range(start_hour2, start_hour2 + eating_hour2 + 1))]
            count = [np.random.randint(1, 5)]

            meta_info['start_hours'].append(start_hour2)
            meta_info['eating_hours'].append(eating_hour2)
            meta_info['count'] = count

        elif pattern_type == 'B':
            start_hour1 = int(np.random.uniform(20, 26))
            eating_hour1 = int(np.random.uniform(6, 10))
            start_hour2 = int(np.random.uniform(34, 38))
            eating_hour2 = int(np.random.uniform(6, 10))

            boost_hours = [
                list(range(start_hour1, start_hour1 + eating_hour1 + 1)),
                list(range(start_hour2, start_hour2 + eating_hour2 + 1))
            ]
            x = np.random.randint(1, 5)
            if x != 4:
                count = [x, np.random.randint(1, 5)]
            else:
                count = [x, np.random.randint(1, 4)]

            meta_info['start_hours'] = [start_hour1, start_hour2]
            meta_info['eating_hours'] = [eating_hour1, eating_hour2]
            meta_info['count'] = count

        elif pattern_type == 'C':
            start_hour1 = int(np.random.uniform(20, 26))
            eating_hour1 = int(np.random.uniform(6, 10))
            boost_hours = [list(range(start_hour1, start_hour1 + eating_hour1 + 1))]
            count = [np.random.randint(1, 5)]

            meta_info['start_hours'].append(start_hour1)
            meta_info['eating_hours'].append(eating_hour1)
            meta_info['count'] = count
            
        elif pattern_type == 'D':
            start_hour = int(np.random.uniform(10, 16))
            eating_hour = int(np.random.uniform(6, 10))
            boost_hours = [list(range(start_hour, start_hour + eating_hour + 1))]
            count = [np.random.randint(1, 5)]
            
            meta_info['start_hours'].append(start_hour)
            meta_info['eating_hours'].append(eating_hour)
            meta_info['count'] = count

        # 커스텀 패턴인 경우 (A, B, C, D가 아닌 경우) - 무작위 생성
        else:
            # 새로운 무작위 패턴 생성
            peak_count = random.randint(1, 3)  # 피크 시간대 1~3개
            boost_hours = []
            count = []
            
            for i in range(peak_count):
                # 무작위 시간대 선택
                start_hour = int(np.random.uniform(6, 42))  # 3시~21시
                eating_hour = int(np.random.uniform(4, 12))  # 2~6시간
                boost_hours.append(list(range(start_hour, start_hour + eating_hour + 1)))
                count.append(np.random.randint(1, 5))
                
                meta_info['start_hours'].append(start_hour)
                meta_info['eating_hours'].append(eating_hour)
                
            meta_info['count'] = count

        waterForm = [(0, 0), (450, 500), (200, 250), (150, 200), (80, 130)]

        for i in range(len(boost_hours)):
            pattern_drink_times = np.random.choice(boost_hours[i], size=count[i], replace=False)
            for hour in pattern_drink_times:
                if hour < 48:  # 48시간 범위 체크
                    usage[hour] += int(np.random.uniform(waterForm[min(count[i], 4)][0], waterForm[min(count[i], 4)][1])) // 10 * 10

        max_drinks = 8 - sum(count)
        base_hours = list(set(range(48)))
        for hours in boost_hours:
            base_hours = list(set(base_hours) - set([h for h in hours if h < 48]))

        base_drinks = np.random.randint(min(3, max_drinks), max(3, max_drinks) + 1)
        if base_hours:  # base_hours가 비어있지 않은 경우에만 실행
            drink_times = np.random.choice(base_hours, size=min(base_drinks, len(base_hours)), replace=False)
            for hour in drink_times:
                drink_usage = int(np.random.uniform(80.0, 130.0)) // 10 * 10
                usage[hour] += drink_usage

        return usage, meta_info

    def generate_usage_with_fixed_count(self, pattern_type, fixed_count):
        """
        특정 패턴과 고정된 counts 값을 기반으로 사용량을 생성 (30일치 데이터 생성용)
        """
        usage = np.zeros(48)
        start_hours = []
        eating_hours = []

        if 'A' in pattern_type:
            start_hour2 = int(np.random.uniform(34, 38))
            eating_hour2 = int(np.random.uniform(6, 10))
            boost_hours = [list(range(start_hour2, start_hour2 + eating_hour2 + 1))]
            start_hours.append(start_hour2)
            eating_hours.append(eating_hour2)

        elif 'B' in pattern_type:
            start_hour1 = int(np.random.uniform(20, 26))
            eating_hour1 = int(np.random.uniform(6, 10))
            start_hour2 = int(np.random.uniform(34, 38))
            eating_hour2 = int(np.random.uniform(6, 10))
            boost_hours = [
                list(range(start_hour1, start_hour1 + eating_hour1 + 1)),
                list(range(start_hour2, start_hour2 + eating_hour2 + 1))
            ]
            start_hours.extend([start_hour1, start_hour2])
            eating_hours.extend([eating_hour1, eating_hour2])

        elif 'C' in pattern_type:
            start_hour1 = int(np.random.uniform(20, 26))
            eating_hour1 = int(np.random.uniform(6, 10))
            boost_hours = [list(range(start_hour1, start_hour1 + eating_hour1 + 1))]
            start_hours.append(start_hour1)
            eating_hours.append(eating_hour1)
            
        elif 'D' in pattern_type:
            start_hour = int(np.random.uniform(10, 16))
            eating_hour = int(np.random.uniform(6, 10))
            boost_hours = [list(range(start_hour, start_hour + eating_hour + 1))]
            start_hours.append(start_hour)
            eating_hours.append(eating_hour)
        
        # 커스텀 패턴인 경우
        else:
            peak_count = len(fixed_count) if fixed_count else random.randint(1, 3)
            boost_hours = []
            for i in range(peak_count):
                start_hour = int(np.random.uniform(6, 42))
                eating_hour = int(np.random.uniform(4, 12))
                boost_hours.append(list(range(start_hour, start_hour + eating_hour + 1)))
                start_hours.append(start_hour)
                eating_hours.append(eating_hour)

        # 기존 boost 처리 동일
        waterForm = [(0, 0), (450, 500), (200, 250), (150, 200), (80, 130)]
        for i in range(len(boost_hours)):
            count = fixed_count[i] if i < len(fixed_count) else 0
            if count == 0:
                continue
            
            # 범위를 벗어나지 않도록 처리
            valid_hours = [h for h in boost_hours[i] if h < 48]
            if not valid_hours:
                continue
                
            times = np.random.choice(valid_hours, size=min(count, len(valid_hours)), replace=False)
            for hour in times:
                usage[hour] += int(np.random.uniform(*waterForm[min(count, 4)])) // 10 * 10

        max_drinks = 8 - sum([fixed_count[i] if i < len(fixed_count) else 0 for i in range(len(boost_hours))])
        base_hours = list(set(range(48)))
        for hours in boost_hours:
            base_hours = list(set(base_hours) - set([h for h in hours if h < 48]))
            
        if base_hours:
            base_drinks = np.random.randint(min(3, max_drinks), max(3, max_drinks) + 1)
            times = np.random.choice(base_hours, size=min(base_drinks, len(base_hours)), replace=False)
            for hour in times:
                usage[hour] += int(np.random.uniform(80.0, 130.0)) // 10 * 10

        return usage, start_hours, eating_hours

    def generate_30_days(self, pattern, user_count_dict, start_date=None, num_days=30):
        """
        30일치 데이터 생성
        
        Args:
            pattern: 패턴 이름
            user_count_dict: 사용자별 counts 딕셔너리
            start_date: 시작 날짜 (기본값: 현재 날짜)
            num_days: 생성할 날짜 수 (기본값: 30)
            
        Returns:
            DataFrame: 생성된 데이터
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=num_days)
            
        records = []
        for person_id, counts in user_count_dict.items():
            for day_offset in range(num_days):
                date = (start_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                usage, start_hours, eating_hours = self.generate_usage_with_fixed_count(pattern, counts)

                for hour in range(48):
                    records.append({
                        'person_id': person_id,
                        'date': date,
                        'hour': hour // 2,
                        'minute': (hour % 2) * 30,
                        'amount': usage[hour],
                        'pattern': pattern,
                        'start_hours': str(start_hours),
                        'eating_hours': str(eating_hours),
                        'counts': str(counts)
                    })
        return pd.DataFrame(records)

    def process_data_in_memory(self, df, pattern):
        """
        이상치 처리를 메모리에서 직접 수행
        
        Args:
            df: 처리할 DataFrame
            pattern: 패턴명
            
        Returns:
            DataFrame: 이상치가 제거된 DataFrame
        """
        try:
            # 날짜 형식 변환
            df['date'] = pd.to_datetime(df['date'])
            
            # 하루 단위 벡터 생성
            daily_vectors = []
            person_day_mapping = []
            
            for (pid, date), group in df.groupby(['person_id', 'date']):
                sorted_group = group.sort_values(by=['hour', 'minute'])
                vec = sorted_group['amount'].values
                if len(vec) == 48:
                    daily_vectors.append(vec)
                    person_day_mapping.append((pid, date))
                    
            daily_vectors = np.array(daily_vectors)
            
            if len(daily_vectors) == 0:
                return df  # 벡터가 없으면 원본 반환
                
            # 정규화
            daily_vectors_norm = self.outlier_detector.normalize_each_day_minmax(daily_vectors)
            
            # DBSCAN 군집화
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import silhouette_samples
            
            dbscan = DBSCAN(eps=0.9, min_samples=3)
            labels = dbscan.fit_predict(daily_vectors_norm)
            
            # 유효 벡터만 (노이즈 제외)
            valid_indices = labels != -1
            valid_vectors = daily_vectors_norm[valid_indices]
            valid_labels = labels[valid_indices]
            
            if len(valid_vectors) == 0:
                return df  # 유효한 클러스터가 없으면 원본 반환
                
            # 실루엣 점수로 최적 클러스터 선택
            sil_samples = silhouette_samples(valid_vectors, valid_labels)
            sil_scores = {
                cluster_id: np.mean(sil_samples[valid_labels == cluster_id])
                for cluster_id in np.unique(valid_labels)
            }
            
            sil_table = pd.DataFrame({
                'Cluster': list(sil_scores.keys()),
                'Silhouette Score': list(sil_scores.values())
            }).sort_values(by='Silhouette Score', ascending=False)
            
            best_cluster = sil_table.iloc[0]['Cluster']
            best_vectors = valid_vectors[valid_labels == best_cluster]
            
            # 중심 벡터 찾기
            
            
            dtw_matrix = np.zeros((len(best_vectors), len(best_vectors)))
            for i in range(len(best_vectors)):
                for j in range(i+1, len(best_vectors)):
                    dist = dtw.distance(best_vectors[i], best_vectors[j])
                    dtw_matrix[i, j] = dist
                    dtw_matrix[j, i] = dist
            
            avg_dtw = dtw_matrix.mean(axis=1)
            center_idx = np.argmin(avg_dtw)
            center_vector = best_vectors[center_idx]
            
            # DTW 거리 계산
            cum_center = self.outlier_detector.cumulative_minmax(center_vector)
            dtw_cum_distances = [dtw.distance(cum_center, self.outlier_detector.cumulative_minmax(vec)) 
                               for vec in daily_vectors_norm]
            
            # 이상치 감지 (IQR 방식)
            q1 = np.percentile(dtw_cum_distances, 25)
            q3 = np.percentile(dtw_cum_distances, 75)
            iqr = q3 - q1
            upper_bound = q3 + 0.5 * iqr
            
            # 이상치가 아닌 데이터 선택
            non_outlier_mask = np.array(dtw_cum_distances) <= upper_bound
            non_outlier_indices = np.where(non_outlier_mask)[0]
            
            # 이상치가 아닌 사람/날짜 조합 추출
            valid_person_days = set()
            for idx in non_outlier_indices:
                if idx < len(person_day_mapping):
                    valid_person_days.add(person_day_mapping[idx])
            
            # 유효한 데이터만 필터링
            filtered_df = df[df.apply(lambda row: (row['person_id'], pd.to_datetime(row['date'])) in valid_person_days, axis=1)]
            
            return filtered_df
            
        except Exception as e:
            print(f"이상치 처리 오류: {str(e)}")
            return df  # 오류 발생 시 원본 반환

    def create_pattern(self, pattern_names, people_cnt, start_date=None, num_days=30):
        """
        패턴 이름과 인원 수에 따라 패턴 데이터를 생성합니다.
        중복된 패턴 이름은 생성하지 않습니다.
        A, B, C, D 외의 패턴이 요청되면 무작위 패턴을 생성합니다.
        메모리 내에서 처리 후 최종 30일치 데이터만 저장합니다.
        
        Args:
            pattern_names: 생성할 패턴 이름 리스트
            people_cnt: 데이터를 생성할 인원 수
            start_date: 시작 날짜 (기본값: 현재 날짜)
            num_days: 생성할 날짜 수 (기본값: 30)
            
        Returns:
            dict: 작업 결과
        """
        try:
            if self.data_path is None:
                from app.config import get_paths_dict
                paths = get_paths_dict()
                self.data_path = paths.get('data_path', '')
            
            # 디렉토리 생성
            os.makedirs(self.data_path, exist_ok=True)
            
            # 패턴 레지스트리 로드
            registry_result = self._load_pattern_registry(self.data_path)
            if registry_result["status"] == "error":
                return registry_result
            
            # 생성이 필요한 패턴과 이미 존재하는 패턴 분류
            already_exists = []
            new_patterns = []
            
            for pattern in pattern_names:
                if self._check_pattern_exists(pattern):
                    already_exists.append(pattern)
                else:
                    new_patterns.append(pattern)
            
            # 신규 패턴 생성
            generated_patterns = []
            for pattern in new_patterns:
                # 1. 1일치 데이터 생성 (DataFrame으로 반환)
                df_1day = self.generate_1day_data(people_cnt, pattern)
                if df_1day.empty:
                    continue
                
                # 2. 이상치 처리 (메모리에서 처리)
                df_filtered = self.process_data_in_memory(df_1day, pattern)
                
                # 3. 30일치 데이터 생성 및 저장
                df_counts = df_filtered[['person_id', 'counts']].drop_duplicates()
                
                try:
                    # 문자열 → 리스트 변환
                    df_counts['counts'] = df_counts['counts'].apply(eval)
                    # 딕셔너리로 변환
                    user_counts = df_counts.set_index('person_id')['counts'].to_dict()
                    
                    # 날짜 설정
                    if start_date is None:
                        start_date = datetime.now() - timedelta(days=num_days)
                    
                    # 30일치 데이터 생성
                    df_30days = self.generate_30_days(pattern, user_counts, start_date, num_days)
                    
                    # 최종 데이터만 저장
                    df_30days.to_csv(os.path.join(self.data_path, f"pattern_{pattern}_30days.csv"), index=False)
                    
                    generated_patterns.append(pattern)
                    
                    # 패턴 레지스트리 업데이트
                    self.pattern_registry[pattern] = {
                        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'people_cnt': people_cnt,
                        'type': 'base' if pattern in self.base_patterns else 'custom'
                    }
                except Exception as e:
                    print(f"30일치 데이터 생성 오류 ({pattern}): {str(e)}")
                    continue
            
            # 패턴 레지스트리 저장
            if generated_patterns:
                save_result = self._save_pattern_registry(self.data_path)
                if save_result["status"] == "error":
                    return save_result
            
            # 결과 메시지 생성
            message = ""
            if generated_patterns:
                message += f"패턴 {', '.join(generated_patterns)} 데이터가 성공적으로 생성되었습니다. "
                message += f"{num_days}일치 데이터 생성이 완료되었습니다."
            if already_exists:
                message += f"패턴 {', '.join(already_exists)}은(는) 이미 존재합니다."
            
            return {
                "status": "success",
                "message": message.strip(),
                "generated": generated_patterns,
                "already_exists": already_exists
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    def generate_1day_data(self, num_people, pattern_type, base_date=None):
        """
        1일치 데이터를 생성하고 DataFrame으로 반환
        
        Args:
            num_people: 데이터를 생성할 인원 수
            pattern_type: 패턴 유형
            base_date: 기준 날짜 (기본값: 현재 날짜)
            
        Returns:
            DataFrame: 생성된 1일치 데이터
        """
        try:
            # 날짜 설정
            if base_date is None:
                base_date = datetime.now()
            
            records = []
            for person_id in range(1, num_people + 1):
                # 패턴별 ID 범위 구분
                if pattern_type == 'A':
                    pid = person_id
                elif pattern_type == 'B':
                    pid = person_id + num_people
                elif pattern_type == 'C':
                    pid = person_id + 2 * num_people
                elif pattern_type == 'D':
                    pid = 30000 + person_id
                else:
                    # 커스텀 패턴: 10만부터 시작
                    pid = 100000 + hash(pattern_type) % 10000 + person_id
                
                # 데이터 생성
                usage, meta = self.generate_human_like_day_strong_pattern(pattern_type)
                
                for hour in range(48):
                    records.append({
                        'person_id': pid,
                        'date': base_date.strftime('%Y-%m-%d'),
                        'hour': hour // 2,
                        'minute': (hour % 2) * 30,
                        'amount': usage[hour],
                        'pattern': pattern_type,
                        'start_hours': str(meta['start_hours']),
                        'eating_hours': str(meta['eating_hours']),
                        'counts': str(meta['count'])
                    })
            
            return pd.DataFrame(records)
        except Exception as e:
            print(f"1일치 데이터 생성 오류: {str(e)}")
            return pd.DataFrame()  # 빈 DataFrame 반환
