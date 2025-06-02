# 이 파일은 시간 vs. 출수량 데이터 기반으로 feature를 생성하는 파일입니다.
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

# 첫 날을 마지막날이랑 매핑시키기 (Optional)
def create_feature_1_day(path, files, N = 0):

    base_dir = f"{path}/VoID_WaterPurifier"
    os.chdir(base_dir)

    # 하위 폴더 생성 (없는 경우)
    processed_dir = os.path.join(base_dir, "pre_processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    pattern_files = files

    # 일단 패턴별 사용자 ID를 추출하는 함수
    def extract_pattern_users(pattern_file, date_str, n_users=0):
        """패턴 파일에서 지정된 날짜의 사용자 ID를 추출"""
        # 필요한 열만 로드하여 메모리 절약
        df = pd.read_csv(pattern_file, usecols=['person_id', 'date'])
        df = df[df['date'] == date_str]
        unique_users = df['person_id'].unique()
        unique_users.sort()  # 정렬

        if n_users > 0 and len(unique_users) > n_users:
            unique_users = unique_users[:n_users]

        return set(unique_users)

    # 전체 패턴에서 각 패턴별 선택된 사용자 ID를 추출 (메모리 효율적 방식)
    pattern_users = {}
    for file in pattern_files:
        pattern_name = file.split('_')[1]  # 'A', 'B', 'C' 추출
        pattern_users[pattern_name] = extract_pattern_users(
            os.path.join(base_dir, file),
            date_str='2022-02-01',
            n_users=N
        )

    # 선택된 사용자 ID 목록 출력
    all_selected_users = set()
    for pattern, users in pattern_users.items():
        all_selected_users.update(users)

    # 메모리 정리
    gc.collect()

    # 1. 메모리 효율적인 데이터 로드 (필요한 사용자만)
    df_20220201_dict = {pattern: [] for pattern in pattern_users.keys()}
    df_20220302_dict = {pattern: [] for pattern in pattern_users.keys()}

    for file in pattern_files:
        pattern_name = file.split('_')[1]  # 'A', 'B', 'C' 추출
        selected_users = pattern_users[pattern_name]

        # 대용량 파일을 청크 단위로 읽어 메모리 효율적으로 처리
        chunk_size = 500000  # 적절한 청크 크기 설정
        for chunk in pd.read_csv(os.path.join(base_dir, file), chunksize=chunk_size):
            # 날짜 및 사용자 필터링
            feb01_chunk = chunk[(chunk['date'] == '2022-02-01') &
                                (chunk['person_id'].isin(selected_users))].copy()
            mar02_chunk = chunk[(chunk['date'] == '2022-03-02') &
                                (chunk['person_id'].isin(selected_users))].copy()

            # 불필요한 열 제거
            if 'start_hours' in feb01_chunk.columns:
                feb01_chunk.drop(columns=['start_hours', 'eating_hours', 'counts'], inplace=True)
                mar02_chunk.drop(columns=['start_hours', 'eating_hours', 'counts'], inplace=True)

            # 패턴 정보 추가
            feb01_chunk['pattern'] = pattern_name
            mar02_chunk['pattern'] = pattern_name

            # 리스트에 추가
            if len(feb01_chunk) > 0:
                df_20220201_dict[pattern_name].append(feb01_chunk)
            if len(mar02_chunk) > 0:
                df_20220302_dict[pattern_name].append(mar02_chunk)

            # 메모리 정리
            del feb01_chunk, mar02_chunk

        # 청크 단위 처리 후 메모리 정리
        gc.collect()

    # 패턴별 데이터 병합
    df_20220201_list = []
    df_20220302_list = []

    for pattern in pattern_users.keys():
        if df_20220201_dict[pattern]:
            df_20220201_list.append(pd.concat(df_20220201_dict[pattern], ignore_index=True))
        if df_20220302_dict[pattern]:
            df_20220302_list.append(pd.concat(df_20220302_dict[pattern], ignore_index=True))

    # 여러 파일의 데이터를 하나의 DataFrame으로 합치기
    df_20220201 = pd.concat(df_20220201_list, ignore_index=True) if df_20220201_list else pd.DataFrame()
    df_20220302 = pd.concat(df_20220302_list, ignore_index=True) if df_20220302_list else pd.DataFrame()

    # 메모리 정리
    del df_20220201_dict, df_20220302_dict, df_20220201_list, df_20220302_list
    gc.collect()

    # GPU 메모리 초기화
    cp.get_default_memory_pool().free_all_blocks()

    # GPU 데이터로 변환
    try:
        gpu_df_20220201 = cudf.DataFrame.from_pandas(df_20220201)
        gpu_df_20220302 = cudf.DataFrame.from_pandas(df_20220302)

        # 데이터 전처리
        # 1. 시간 정보 처리 (GPU 버전)
        for df in [gpu_df_20220201, gpu_df_20220302]:
            # 정확한 시간 생성 (hour.minute 형식)
            df['exact_hour'] = df['hour'] + df['minute'] / 60
            # 하루 내 48개 구간으로 나눈 시간 인덱스 (0부터 47)
            df['time_idx'] = df['hour'] * 2 + (df['minute'] == 30).astype('int32')

        # 2. 각 사용자별로 물 사용 기록 찾기 (amount > 0)
        gpu_df_20220201['is_usage'] = gpu_df_20220201['amount'] > 0
        gpu_df_20220302['is_usage'] = gpu_df_20220302['amount'] > 0

        # 3. 사용자별로 출수 순서 계산 (실제 사용량이 있는 경우만 순서 부여)
        # CuDF에서 정렬
        gpu_df_20220201 = gpu_df_20220201.sort_values(['pattern', 'person_id', 'hour', 'minute'])
        gpu_df_20220302 = gpu_df_20220302.sort_values(['pattern', 'person_id', 'hour', 'minute'])

        # 사용자별 그룹핑 및 누적합 계산 (CuDF)
        gpu_df_20220201['output_seq'] = gpu_df_20220201.groupby('person_id')['is_usage'].cumsum() * gpu_df_20220201[
            'is_usage']
        gpu_df_20220302['output_seq'] = gpu_df_20220302.groupby('person_id')['is_usage'].cumsum() * gpu_df_20220302[
            'is_usage']

        # 사전 계산: 사용자별 전체 사용량과 통계 (GPU 가속)
        # amount > 0인 레코드에 대해서만 계산
        usage_df_20220302 = gpu_df_20220302[gpu_df_20220302['amount'] > 0]
        grouped_amount = usage_df_20220302.groupby('person_id')['amount'].sum()
        total_usage_by_user = dict(zip(grouped_amount.index.to_pandas(), grouped_amount.to_pandas()))

        # 전날 통계 (GPU 가속) - amount > 0인 레코드에 대해서만
        prev_day_stats = usage_df_20220302.groupby('person_id').agg({
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
            user_prev_data = usage_df_20220302[usage_df_20220302['person_id'] == user].to_pandas()
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
        df_20220201 = gpu_df_20220201.to_pandas()
        df_20220302 = gpu_df_20220302.to_pandas()

        # GPU 메모리 정리
        del gpu_df_20220201, gpu_df_20220302, usage_df_20220302
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:

        # 기본 데이터 처리 (CPU 버전)
        # 시간 정보 처리 (CPU 버전)
        for df in [df_20220201, df_20220302]:
            df['exact_hour'] = df['hour'] + df['minute'] / 60
            df['time_idx'] = df['hour'] * 2 + (df['minute'] == 30).astype('int')
            df['is_usage'] = df['amount'] > 0

        # 사용자별로 출수 순서 계산
        for df in [df_20220201, df_20220302]:
            # 정렬
            df.sort_values(['pattern', 'person_id', 'hour', 'minute'], inplace=True)
            # 그룹핑 및 누적합 계산
            df['output_seq'] = df.groupby('person_id')['is_usage'].cumsum() * df['is_usage']

        # 전날 통계 계산
        usage_df_20220302 = df_20220302[df_20220302['amount'] > 0]

        # 사용자별 전체 사용량
        total_usage_by_user = usage_df_20220302.groupby('person_id')['amount'].sum().to_dict()

        # 전날 통계
        prev_day_stats = usage_df_20220302.groupby('person_id').agg({
            'amount': ['sum', 'mean', 'std']
        })

        prev_day_stats.columns = ['prev_day_total', 'prev_day_mean', 'prev_day_std']
        prev_day_stats.reset_index(inplace=True)

        # NaN 값을 0으로 변환
        prev_day_stats.fillna(0, inplace=True)

        # 전날 데이터 기반 사용자별 변화율(기울기) 미리 계산
        avg_change_rates = {}
        for user in prev_day_stats['person_id'].unique():
            user_prev_data = usage_df_20220302[usage_df_20220302['person_id'] == user].copy()
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

    # CUDA 커널 정의 - 단순화된 버전: 딕셔너리 전달하지 않음
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

    # CPU 버전의 계산 함수 (CUDA 실패 시 대비)
    def calculate_features_cpu(user_data, prev_day_total):
        n_records = len(user_data)
        result_ratio_to_prev_day = np.zeros(n_records, dtype=np.float32)
        result_prev_sum = np.zeros(n_records, dtype=np.float32)

        # 누적합 계산
        for idx in range(n_records):
            curr_sum = 0
            for i in range(idx):
                curr_sum += user_data.iloc[i]['amount']

            result_prev_sum[idx] = curr_sum

            # 현재 레코드 포함한 총합
            curr_sum += user_data.iloc[idx]['amount']

            # 전날 대비 비율 계산
            if prev_day_total > 0:
                result_ratio_to_prev_day[idx] = curr_sum / prev_day_total

        return result_ratio_to_prev_day, result_prev_sum

    # 결과를 저장할 리스트
    result_rows = []

    # 각 패턴에 대해 처리
    for pattern in df_20220201['pattern'].unique():

        pattern_df_today = df_20220201[df_20220201['pattern'] == pattern].copy()
        pattern_df_prev = df_20220302[df_20220302['pattern'] == pattern].copy()

        # 패턴별 고유 사용자 추출
        unique_users = pattern_df_today['person_id'].unique()
        unique_users.sort()  # person_id 기준 정렬

        user_count = len(unique_users)

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
                try:
                    # GPU 계산 시도
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
                    blocks_per_grid = max(1, math.ceil(n_records / threads_per_block))

                    # CUDA 커널 실행 (사용자별 prev_day_total 직접 전달)
                    calculate_features[blocks_per_grid, threads_per_block](
                        d_amount, d_is_usage, d_result_ratio, d_result_prev_sum, float(prev_day_total)
                    )

                    # 결과 가져오기
                    cuda.synchronize()
                    ratio_to_prev_day = d_result_ratio.copy_to_host()
                    prev_sum = d_result_prev_sum.copy_to_host()

                    # GPU 메모리 해제
                    del d_amount, d_is_usage, d_result_ratio, d_result_prev_sum

                except Exception as e:
                    # CPU 대체 계산
                    ratio_to_prev_day, prev_sum = calculate_features_cpu(user_data_today_sorted, prev_day_total)

                # 데이터프레임에 결과 병합
                user_data_today_sorted.loc[:, 'ratio_to_prev_day'] = ratio_to_prev_day
                user_data_today_sorted.loc[:, 'prev_sum'] = prev_sum

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
                         ((user_data_prev['hour'] == current_hour) & (user_data_prev['minute'] <= current_minute))) &
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

                # 정기적으로 메모리 정리
                if idx % 100 == 0:
                    gc.collect()


    # 최종 결과 생성 및 저장
    result_df = pd.DataFrame(result_rows)

    # 사용하지 않을 임시 컬럼 제거
    if 'exact_hour' in result_df.columns:
        result_df = result_df.drop(['exact_hour', 'time_idx', 'is_usage'], axis=1)

    # 처리 폴더에 저장
    output_path = os.path.join(processed_dir, "1_day_feature.csv")

    # 파일이 이미 존재하는지 확인
    if os.path.exists(output_path):
        # 기존 파일 불러오기
        existing_df = pd.read_csv(output_path)
        
        # 기존 데이터와 새 데이터 병합
        merged_df = pd.concat([existing_df, result_df], ignore_index=True)
        
        # 중복 데이터 제거 (선택적)
        merged_df = merged_df.drop_duplicates().reset_index(drop=True)
        
        # 병합된 데이터 저장
        merged_df.to_csv(output_path, index=False)
    else:
        # 파일이 없는 경우 새로 저장
        result_df.to_csv(output_path, index=False)

    # GPU 메모리 정리
    cp.get_default_memory_pool().free_all_blocks()
    try:
        cuda.close()
    except:
        pass

    return result_df



def create_feature_all_day(path, files, N = 0):
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
        use_gpu = False
    base_dir = f"{path}/VoID_WaterPurifier"
    os.chdir(base_dir)

    # 하위 폴더 생성 (없는 경우)
    processed_dir = os.path.join(base_dir, "pre_processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # 첫날 데이터 로드 (1_day_features.csv에서 로드)
    first_day_path = os.path.join(processed_dir, "1_day_feature.csv")
    first_day_df = pd.read_csv(first_day_path)

    # 원본 패턴 데이터 로드 및 통합
    pattern_files = files

    pattern_dfs = []
    for pattern_file in pattern_files:
        file_path = os.path.join(base_dir, pattern_file)  # 원본 파일은 기본 디렉토리에서 로드
        df = pd.read_csv(file_path)
        pattern_name = pattern_file.split('_')[1]  # 'A', 'B', 'C' 추출
        df['pattern'] = pattern_name  # 패턴 칼럼 추가
        pattern_dfs.append(df)

    # 모든 패턴 데이터 통합
    original_all_data = pd.concat(pattern_dfs, ignore_index=True)

    # 각 패턴별로 선택된 사용자 ID 추출 (첫날 데이터에서)
    selected_person_ids = {}
    for pattern in first_day_df['pattern'].unique():
        pattern_users = first_day_df[first_day_df['pattern'] == pattern]['person_id'].unique()
        selected_person_ids[pattern] = set(pattern_users)

    # 날짜 범위 정의 (2월 2일부터 3월 2일까지)
    start_date = datetime(2022, 2, 2)
    end_date = datetime(2022, 3, 2)  # 3월 2일까지 변경
    date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]


    # CUDA 커널 정의 - 단순화된 버전: 딕셔너리 전달하지 않음
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

    # 특성 생성 함수 정의 (GPU 가속) - 수정된 버전
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
        gpu_current_df['output_seq'] = gpu_current_df.groupby('person_id')['is_usage'].cumsum() * gpu_current_df[
            'is_usage']
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

    # 패턴별 사용자 ID 목록 생성 (첫날 데이터에서)
    pattern_user_sets = {}
    for pattern in first_day_df['pattern'].unique():
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
            output_path = os.path.join(processed_dir, f"{date_str}_with_features_nonzero.csv")
            current_df.to_csv(output_path, index=False)

            # 다음 날짜를 위해 현재 날짜를 이전 날짜로 설정
            prev_df = current_df

        # GPU 메모리 정리
        cp.get_default_memory_pool().free_all_blocks()

    # 최종 GPU 메모리 정리
    cp.get_default_memory_pool().free_all_blocks()
    cuda.close()


def merge_data(path):

    base_dir = f"{path}/VoID_WaterPurifier"
    os.chdir(base_dir)

    processed_dir = os.path.join(base_dir, "pre_processed")

    # 결과 파일들 찾기
    feature_files = [f for f in os.listdir(processed_dir) if f.endswith('_with_features_nonzero.csv')]
    feature_files.sort()  # 날짜 순으로 정렬

    # 모든 파일 병합
    all_dfs = []
    total_records = 0

    for file in feature_files:
        file_path = os.path.join(processed_dir, file)  # 처리 폴더에서 파일 로드
        try:
            df = pd.read_csv(file_path)

            # 불필요한 컬럼 제거
            cols_to_drop = ['start_hours', 'eating_hours', 'counts', 'exact_hour', 'time_idx', 'is_usage']
            df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

            all_dfs.append(df)
            total_records += len(df)
        except Exception as e:
            pass

    # 모든 데이터 통합
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        # 중복 행 확인 및 제거 (선택적)
        duplicates = merged_df.duplicated().sum()
        if duplicates > 0:
            merged_df = merged_df.drop_duplicates().reset_index(drop=True)

        # person_id로 정렬 (선택적)
        if 'person_id' in merged_df.columns:
            merged_df = merged_df.sort_values('person_id').reset_index(drop=True)

        # 처리 폴더에 파일로 저장
        output_path = os.path.join(processed_dir, "all_features_merged.csv")

        # 파일이 이미 존재하는지 확인
        if os.path.exists(output_path):
            # 기존 파일 불러오기
            existing_df = pd.read_csv(output_path)
            
            # 기존 데이터와 새 데이터 병합
            merged_df = pd.concat([existing_df, merged_df], ignore_index=True)
            
            # 중복 데이터 제거
            merged_df = merged_df.drop_duplicates().reset_index(drop=True)
            
            # person_id로 정렬 (선택적)
            if 'person_id' in merged_df.columns:
                merged_df = merged_df.sort_values('person_id').reset_index(drop=True)

        # 병합된 데이터 저장
        merged_df.to_csv(output_path, index=False)
    return merged_df

def create_feature(path, files, N):
    df_one=create_feature_1_day(path, files, N)
    create_feature_all_day(path, files, N)
    df_all=merge_data(path)
    return df_one, df_all