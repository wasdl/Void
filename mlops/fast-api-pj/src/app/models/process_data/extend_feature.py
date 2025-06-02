# https://colab.research.google.com/drive/111J_c-6lsq6-yZvZObtr68rXjEKpPVCP#scrollTo=desn5zcjbbdk
import pandas as pd
import numpy as np
import os
import cupy as cp
import cudf
from numba import cuda
import math
import gc  # 메모리 관리를 위해 추가
from datetime import datetime, timedelta
from app.config import get_paths_dict

class FeatureExtender:
    def __init__(self):
        self.base_dir = None
        self.processed_dir = None
        self.paths = get_paths_dict()
        
    def setup_directories(self):
        """디렉토리 설정"""
        self.base_dir = self.paths.get('data', '')
        
        # 하위 폴더 생성 (없는 경우)
        self.processed_dir = os.path.join(self.base_dir, "resources", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # 현재 작업 디렉토리를 변경하지 않고 절대 경로 사용
    
    def create_feature(self, files, N=0, start_date=None, end_date=None):
        """
        특성 확장 메인 함수
        
        Args:
            files: 처리할 파일 목록
            N: 처리할 사용자 수 제한 (0=전체)
            start_date: 시작 날짜 (기본값: 현재 날짜 - 30일)
            end_date: 종료 날짜 (기본값: 현재 날짜)
            
        Returns:
            dict: 패턴별 특성 파일 경로
        """
        self.setup_directories()
        
        # 특성 생성 및 패턴별 저장
        self.create_feature_all_day(files, N, start_date, end_date)
        pattern_files = self.process_by_pattern()
        
        return pattern_files  # 패턴별 파일 경로 반환

    def create_feature_all_day(self, files, N=0, start_date=None, end_date=None):
        """
        모든 날짜에 대한 특성 생성
        
        Args:
            files: 처리할 파일 목록
            N: 처리할 사용자 수 제한 (0=전체)
            start_date: 시작 날짜 (기본값: 현재 날짜 - 30일)
            end_date: 종료 날짜 (기본값: 현재 날짜)
        """
        # GPU 초기화 및 확인
        try:
            # CUDA 컨텍스트 초기화
            cuda.close()
            cuda.select_device(0)

            # 사용 가능한 GPU 확인
            n_gpu = cp.cuda.runtime.getDeviceCount()

            # GPU 메모리 상태 확인
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mem = mem_info[0] / 1024 ** 3  # GB 단위
            total_mem = mem_info[1] / 1024 ** 3  # GB 단위

            use_gpu = True
        except Exception as e:
            raise RuntimeError(f"GPU 초기화 실패: {str(e)}")

        # 원본 패턴 데이터 로드 및 통합
        pattern_files = files
        pattern_dfs = []
        
        for pattern_file in pattern_files:
            file_path = os.path.join(self.base_dir, pattern_file)  # 원본 파일은 기본 디렉토리에서 로드
            df = pd.read_csv(file_path)
            pattern_name = pattern_file.split('_')[1]  # 'A', 'B', 'C' 추출
            df['pattern'] = pattern_name  # 패턴 칼럼 추가
            pattern_dfs.append(df)

        # 모든 패턴 데이터 통합
        original_all_data = pd.concat(pattern_dfs, ignore_index=True)

        # 날짜 범위 정의
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        # 문자열 형식이면 datetime으로 변환
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

        # CUDA 커널 정의
        @cuda.jit
        def calculate_features(amount_arr, is_usage_arr, result_ratio_to_prev_day, result_prev_sum, prev_day_total):
            # 스레드 인덱스
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            bw = cuda.blockDim.x

            # 전체 스레드 인덱스
            idx = tx + bx * bw

            # 배열 길이 내에서만 처리
            if idx < len(amount_arr):
                # 공유 변수: 누적 합계
                curr_sum = 0

                # 현재 인덱스까지 누적합 계산 (amount > 0인 레코드만)
                for i in range(idx + 1):
                    if is_usage_arr[i]:
                        if i < idx:  # 자기 자신 제외
                            curr_sum += amount_arr[i]

                # 결과 저장
                result_prev_sum[idx] = curr_sum

                # 현재 레코드가 사용량이 있는 경우만 비율 계산
                if is_usage_arr[idx]:
                    curr_sum += amount_arr[idx]  # 현재 레코드 포함

                    # 전날 대비 비율 계산
                    if prev_day_total > 0:
                        result_ratio_to_prev_day[idx] = curr_sum / prev_day_total
                    else:
                        result_ratio_to_prev_day[idx] = 0

        # 특성 생성 함수 정의 (GPU 가속)
        def generate_features_for_date(current_date, prev_date_df, all_original_data, selected_ids):
            """
            지정된 날짜에 대한 데이터 및 특성을 생성합니다. (GPU 가속)
            - 원본 데이터에서 해당 날짜 데이터를 가져옵니다
            - amount > 0인 레코드만 처리합니다.
            - selected_ids: 패턴별 선택된 사용자 ID 집합 (딕셔너리)
            """
            date_str = current_date.strftime('%Y-%m-%d')

            # 원본 데이터에서 현재 날짜 데이터 필터링
            current_df = all_original_data[all_original_data['date'] == date_str].copy()

            if len(current_df) == 0:
                return pd.DataFrame()  # 빈 데이터프레임 반환

            # 각 패턴별로 선택된 사용자 ID만 필터링
            filtered_current_df = []
            for pattern, user_ids in selected_ids.items():
                pattern_df = current_df[current_df['pattern'] == pattern]
                filtered_pattern_df = pattern_df[pattern_df['person_id'].isin(user_ids)]
                filtered_current_df.append(filtered_pattern_df)

            current_df = pd.concat(filtered_current_df)

            # GPU 메모리 초기화
            cp.get_default_memory_pool().free_all_blocks()

            # GPU 데이터로 변환
            gpu_current_df = cudf.DataFrame.from_pandas(current_df)
            gpu_prev_df = cudf.DataFrame.from_pandas(prev_date_df)

            # 데이터 전처리
            # 1. 시간 정보 처리 (GPU 버전)
            for df in [gpu_current_df, gpu_prev_df]:
                # 정확한 시간 생성 (hour.minute 형식)
                df['exact_hour'] = df['hour'] + df['minute'] / 60
                # 하루 내 48개 구간으로 나눈 시간 인덱스 (0부터 47)
                df['time_idx'] = df['hour'] * 2 + (df['minute'] == 30).astype('int32')

            # 2. 각 사용자별로 물 사용 기록 찾기 (amount > 0)
            gpu_current_df['is_usage'] = gpu_current_df['amount'] > 0
            gpu_prev_df['is_usage'] = gpu_prev_df['amount'] > 0

            # 3. 사용자별로 출수 순서 계산 (실제 사용량이 있는 경우만 순서 부여)
            # CuDF에서 정렬
            gpu_current_df = gpu_current_df.sort_values(['pattern', 'person_id', 'hour', 'minute'])
            gpu_prev_df = gpu_prev_df.sort_values(['pattern', 'person_id', 'hour', 'minute'])

            # 사용자별 그룹핑 및 누적합 계산 (CuDF)
            gpu_current_df['output_seq'] = gpu_current_df.groupby('person_id')['is_usage'].cumsum() * gpu_current_df['is_usage']
            gpu_prev_df['output_seq'] = gpu_prev_df.groupby('person_id')['is_usage'].cumsum() * gpu_prev_df['is_usage']

            # 사전 계산: 사용자별 전체 사용량과 통계 (GPU 가속)
            # amount > 0인 레코드에 대해서만 계산
            usage_prev_df = gpu_prev_df[gpu_prev_df['amount'] > 0]
            grouped_amount = usage_prev_df.groupby('person_id')['amount'].sum()
            total_usage_by_user = dict(zip(grouped_amount.index.to_pandas(), grouped_amount.to_pandas()))

            # 전날 통계 (GPU 가속) - amount > 0인 레코드에 대해서만
            prev_day_stats = usage_prev_df.groupby('person_id').agg({
                'amount': ['sum', 'mean', 'std']
            }).to_pandas()

            # 멀티 인덱스 단일화
            prev_day_stats.columns = ['prev_day_total', 'prev_day_mean', 'prev_day_std']
            prev_day_stats.reset_index(inplace=True)

            # NaN 값을 0으로 변환
            prev_day_stats.fillna(0, inplace=True)

            # 전날 데이터 기반 사용자별 변화율(기울기) 미리 계산
            avg_change_rates = {}
            for user in prev_day_stats['person_id'].unique():
                user_prev_data = usage_prev_df[usage_prev_df['person_id'] == user].to_pandas()
                if len(user_prev_data) >= 2:
                    # 출수 순서별로 정렬
                    user_prev_data = user_prev_data.sort_values('output_seq')
                    x = user_prev_data['output_seq'].values
                    y = user_prev_data['amount'].values

                    # 선형 회귀 기울기 계산
                    if len(x) > 1:
                        x_mean = x.mean()
                        y_mean = y.mean()
                        numerator = np.sum((x - x_mean) * (y - y_mean))
                        denominator = np.sum((x - x_mean) ** 2)
                        slope = numerator / denominator if denominator != 0 else 0
                        avg_change_rates[user] = slope
                    else:
                        avg_change_rates[user] = 0
                else:
                    avg_change_rates[user] = 0

            # 딕셔너리 생성
            prev_day_stats_dict = {
                row['person_id']: (row['prev_day_total'],
                                row['prev_day_mean'],
                                row['prev_day_std'])
                for _, row in prev_day_stats.iterrows()
            }

            # GPU 메모리 해제 및 결과를 CPU로 가져오기
            current_df = gpu_current_df.to_pandas()
            prev_df = gpu_prev_df.to_pandas()

            # GPU 메모리 해제
            del gpu_current_df, gpu_prev_df
            cp.get_default_memory_pool().free_all_blocks()

            # 결과를 저장할 리스트
            result_rows = []

            # 각 패턴에 대해 처리
            for pattern in current_df['pattern'].unique():
                pattern_df_today = current_df[current_df['pattern'] == pattern].copy()
                pattern_df_prev = prev_df[prev_df['pattern'] == pattern].copy()

                unique_users = pattern_df_today['person_id'].unique()

                for idx, user in enumerate(unique_users):
                    # 사용자 데이터
                    user_data_today = pattern_df_today[pattern_df_today['person_id'] == user].copy()
                    user_data_prev = pattern_df_prev[pattern_df_prev['person_id'] == user].copy()

                    # 사용자에 대한 전날 통계 정보
                    prev_stats = prev_day_stats_dict.get(user, (0, 0, 0))
                    prev_day_total, prev_day_mean, prev_day_std = prev_stats

                    # 사전 계산된 기울기 가져오기
                    avg_change_rate = avg_change_rates.get(user, 0)

                    if prev_day_total == 0:
                        # 전날 사용량이 없는 경우, amount > 0인 레코드만 기본 특성 추가
                        for _, row in user_data_today[user_data_today['amount'] > 0].iterrows():
                            new_row = row.to_dict()
                            new_row.update({
                                'ratio_to_prev_day': 0,
                                'ratio_prev_to_total': 0,
                                'time_diff_prev_outputs': 0,
                                'prev_sum': 0,
                                'prev_day_mean': 0,
                                'prev_day_std': 0,
                                'prev_day_total': 0,
                                'slope_prev_day_n_n_minus_1': 0,
                                'slope_prev_day_n_minus_1_n_minus_2': 0,
                                'avg_change_rate': 0,
                                'prev_output': 0,
                                'prev_prev_output': 0
                            })
                            result_rows.append(new_row)
                        continue

                    # amount > 0인 레코드만 처리
                    user_data_today_usage = user_data_today[user_data_today['amount'] > 0].copy()

                    if len(user_data_today_usage) == 0:
                        continue  # 사용자에게 amount > 0인 레코드가 없으면 건너뛰기

                    # GPU로 사용할 데이터만 추출
                    user_data_today_sorted = user_data_today_usage.sort_values(['hour', 'minute'])
                    user_data_prev_usage = user_data_prev[user_data_prev['amount'] > 0].sort_values(['hour', 'minute'])

                    # 실제 사용량이 있는 레코드만 처리
                    n_records = len(user_data_today_sorted)

                    if n_records > 0:
                        # 임시 결과 배열
                        result_ratio_to_prev_day = np.zeros(n_records, dtype=np.float32)
                        result_prev_sum = np.zeros(n_records, dtype=np.float32)

                        # GPU 메모리로 데이터 복사
                        d_amount = cuda.to_device(user_data_today_sorted['amount'].values.astype(np.float32))
                        d_is_usage = cuda.to_device(np.ones(n_records, dtype=np.bool_))  # 모두 사용량이 있는 레코드
                        d_result_ratio = cuda.to_device(result_ratio_to_prev_day)
                        d_result_prev_sum = cuda.to_device(result_prev_sum)

                        # CUDA 커널 실행 설정
                        threads_per_block = 256
                        blocks_per_grid = math.ceil(n_records / threads_per_block)

                        # CUDA 커널 실행 (사용자별 prev_day_total 직접 전달)
                        calculate_features[blocks_per_grid, threads_per_block](
                            d_amount, d_is_usage, d_result_ratio, d_result_prev_sum, float(prev_day_total)
                        )

                        # 결과 가져오기
                        cuda.synchronize()
                        ratio_to_prev_day = d_result_ratio.copy_to_host()
                        prev_sum = d_result_prev_sum.copy_to_host()

                        # 데이터프레임에 결과 병합
                        user_data_today_sorted.loc[:, 'ratio_to_prev_day'] = ratio_to_prev_day
                        user_data_today_sorted.loc[:, 'prev_sum'] = prev_sum

                        # GPU 메모리 해제
                        del d_amount, d_is_usage, d_result_ratio, d_result_prev_sum

                        # 각 시간대에 대해 feature 계산 (amount > 0인 레코드만)
                        for i, (_, row) in enumerate(user_data_today_sorted.iterrows()):
                            current_hour = row['hour']
                            current_minute = row['minute']
                            current_exact_hour = row['exact_hour']

                            # 이미 계산된 값 사용
                            ratio_to_prev_day = row['ratio_to_prev_day']
                            prev_sum = row['prev_sum']

                            # 전날 현재시간까지 마신 양 계산
                            prev_records = user_data_prev[
                                ((user_data_prev['hour'] < current_hour) |
                                ((user_data_prev['hour'] == current_hour) & (
                                            user_data_prev['minute'] <= current_minute))) &
                                (user_data_prev['amount'] > 0)  # amount > 0인 레코드만
                                ]
                            prev_cumsum = prev_records['amount'].sum()
                            total_usage = total_usage_by_user.get(user, 0)
                            ratio_prev_to_total = prev_cumsum / total_usage if total_usage > 0 else 0

                            # 현재 출수 순서
                            current_seq = row['output_seq']

                            # 당일 이전 출수 데이터
                            today_prev_outputs = user_data_today_sorted[
                                user_data_today_sorted['output_seq'] < current_seq
                                ].sort_values('output_seq')

                            prev_output = today_prev_outputs['amount'].iloc[-1] if len(today_prev_outputs) > 0 else 0
                            prev_prev_output = today_prev_outputs['amount'].iloc[-2] if len(today_prev_outputs) >= 2 else 0

                            # [수정] 출수 간 시간 차이: 현재 출수 시간 - 이전 출수 시간 (초 단위로)
                            time_diff = 0
                            if len(today_prev_outputs) > 0:
                                last_idx = today_prev_outputs.index[-1]
                                prev_time = today_prev_outputs.loc[last_idx, 'exact_hour']
                                time_diff = (current_exact_hour - prev_time) * 60 * 60  # 초 단위로 변환

                            # 전날 데이터 기반 slope 계산
                            slope1, slope2 = 0, 0

                            # 전날 데이터를 출수 순서별로 정리
                            prev_day_seq_data = user_data_prev_usage.set_index('output_seq')['amount'] if len(
                                user_data_prev_usage) > 0 else pd.Series()

                            # slope1: 전날 n번째 출수와 n-1번째 출수의 차이
                            if current_seq in prev_day_seq_data.index and current_seq - 1 in prev_day_seq_data.index:
                                n_usage = prev_day_seq_data.loc[current_seq]
                                n_minus_1_usage = prev_day_seq_data.loc[current_seq - 1]
                                slope1 = n_usage - n_minus_1_usage

                            # slope2: 전날 n-1번째 출수와 n-2번째 출수의 차이
                            if current_seq - 1 in prev_day_seq_data.index and current_seq - 2 in prev_day_seq_data.index:
                                n_minus_1_usage = prev_day_seq_data.loc[current_seq - 1]
                                n_minus_2_usage = prev_day_seq_data.loc[current_seq - 2]
                                slope2 = n_minus_1_usage - n_minus_2_usage

                            # feature 추가
                            new_row = row.to_dict()
                            new_row.update({
                                'ratio_to_prev_day': ratio_to_prev_day,
                                'ratio_prev_to_total': ratio_prev_to_total,
                                'time_diff_prev_outputs': time_diff,
                                'prev_sum': prev_sum,
                                'prev_day_mean': prev_day_mean,
                                'prev_day_std': prev_day_std,
                                'prev_day_total': prev_day_total,
                                'slope_prev_day_n_n_minus_1': slope1,
                                'slope_prev_day_n_minus_1_n_minus_2': slope2,
                                'avg_change_rate': avg_change_rate,
                                'prev_output': prev_output,
                                'prev_prev_output': prev_prev_output
                            })
                            result_rows.append(new_row)

            # 결과 데이터프레임 생성
            result_df = pd.DataFrame(result_rows)

            # 필요 없는 컬럼 제거
            cols_to_drop = ['exact_hour', 'time_idx', 'is_usage', 'start_hours', 'eating_hours', 'counts']
            result_df = result_df.drop([col for col in cols_to_drop if col in result_df.columns], axis=1)

            # GPU 메모리 정리
            cp.get_default_memory_pool().free_all_blocks()

            return result_df

        # 첫날의 이전 날짜 데이터 가져오기
        prev_date = start_date - timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        prev_df = original_all_data[original_all_data['date'] == prev_date_str].copy()

        # 패턴별 사용자 ID 목록 생성
        pattern_user_sets = {}
        for pattern in original_all_data['pattern'].unique():
            pattern_df = original_all_data[(original_all_data['date'] == prev_date_str) &
                                        (original_all_data['pattern'] == pattern)]

            # 패턴별 고유 사용자 ID 추출 및 정렬
            pattern_users = pattern_df['person_id'].unique()
            pattern_users.sort()

            # N값이 설정된 경우, 패턴별로 처리할 사용자 수 제한
            if N > 0 and len(pattern_users) > N:
                pattern_users = pattern_users[:N]

            pattern_user_sets[pattern] = set(pattern_users)

        # 이전 날짜 데이터 필터링
        filtered_prev_dfs = []
        for pattern, user_ids in pattern_user_sets.items():
            pattern_df = prev_df[prev_df['pattern'] == pattern]
            filtered_pattern_df = pattern_df[pattern_df['person_id'].isin(user_ids)]
            filtered_prev_dfs.append(filtered_pattern_df)

        if filtered_prev_dfs:
            prev_df = pd.concat(filtered_prev_dfs)

        # 나머지 날짜 처리
        for current_date in date_range:
            date_str = current_date.strftime('%Y-%m-%d')

            # 이전 날짜 (문자열)
            prev_date = current_date - timedelta(days=1)

            # 현재 날짜에 대한 특성 생성 (이전 날짜 참조)
            current_df = generate_features_for_date(current_date, prev_df, original_all_data, pattern_user_sets)

            if len(current_df) > 0:
                # 처리 폴더에 저장
                output_path = os.path.join(self.processed_dir, f"{date_str}_with_features_nonzero.csv")
                current_df.to_csv(output_path, index=False)

                # 다음 날짜를 위해 현재 날짜를 이전 날짜로 설정
                prev_df = current_df

            # GPU 메모리 정리
            cp.get_default_memory_pool().free_all_blocks()

        # 최종 GPU 메모리 정리
        cp.get_default_memory_pool().free_all_blocks()
        cuda.close()
        
        return True

    def process_by_pattern(self):
        """특성 파일들을 패턴별로 처리"""
        # 결과 파일들 찾기
        feature_files = [f for f in os.listdir(self.processed_dir) if f.endswith('_with_features_nonzero.csv')]
        feature_files.sort()  # 날짜 순으로 정렬
        
        # 패턴별 데이터 저장
        pattern_dfs = {}
        pattern_files = {}
        
        # 모든 파일 읽기
        for file in feature_files:
            file_path = os.path.join(self.processed_dir, file)
            try:
                df = pd.read_csv(file_path)
                
                # 불필요한 컬럼 제거
                cols_to_drop = ['start_hours', 'eating_hours', 'counts', 'exact_hour', 'time_idx', 'is_usage']
                df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
                
                # 패턴별로 데이터 분리
                for pattern in df['pattern'].unique():
                    pattern_df = df[df['pattern'] == pattern]
                    
                    if pattern not in pattern_dfs:
                        pattern_dfs[pattern] = []
                    
                    pattern_dfs[pattern].append(pattern_df)
                    
            except Exception as e:
                print(f"파일 로드 오류 {file}: {str(e)}")
        
        # 패턴별로 데이터 병합 및 저장
        for pattern, dfs in pattern_dfs.items():
            if dfs:
                # 패턴별 데이터 병합
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # 중복 행 확인 및 제거
                duplicates = merged_df.duplicated().sum()
                if duplicates > 0:
                    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
                
                # person_id로 정렬
                if 'person_id' in merged_df.columns:
                    merged_df = merged_df.sort_values('person_id').reset_index(drop=True)
                
                # 패턴별 파일로 저장
                output_path = os.path.join(self.processed_dir, f"pattern_{pattern}_features.csv")
                merged_df.to_csv(output_path, index=False)
                
                # 결과 파일 경로 저장
                pattern_files[pattern] = output_path
                
                print(f"패턴 {pattern} 특성 파일 생성 완료: {len(merged_df)}개 레코드")
                
        return pattern_files

    def merge_data(self):
        """특성 파일들을 패턴별로 처리하는 기존 함수 (이전 버전 호환성 유지)"""
        return self.process_by_pattern()