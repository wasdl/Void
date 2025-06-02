# 이 파일은 베이스 모델을 만들기 위한 파일입니다.
# https://colab.research.google.com/drive/1QDzQfdW5eIFdshgMy5Ma94TFhME73vMJ?usp=sharing

import cudf
import cupy as cp
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
import os

def create_base_model(path, file):
    base_dir = f"{path}/VoID_WaterPurifier"
    os.chdir(base_dir)

    # 모델 저장 디렉토리 설정
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_dir_1 = os.path.join(model_dir, "1")
    model_dir_2 = os.path.join(model_dir, "2")
    os.makedirs(model_dir_1, exist_ok=True)
    os.makedirs(model_dir_2, exist_ok=True)

    # 데이터 로드
    df = pd.read_csv(file)

    # amount가 0인 데이터 필터링
    df = df[df['amount'] > 0]  # amount가 0보다 큰 행만 유지

    # 시간 특성 통합 및 sin-cos 변환
    df['time_in_minutes'] = df['hour'] * 60 + df['minute']  # 0-1439 분 범위
    df['time_sin'] = np.sin(2 * np.pi * df['time_in_minutes'] / 1440)  # 1440 = 24시간 * 60분
    df['time_cos'] = np.cos(2 * np.pi * df['time_in_minutes'] / 1440)

    # 특성 목록 정의
    features = [
        'time_sin', 'time_cos',
        'ratio_to_prev_day',
        'ratio_prev_to_total',
        'time_diff_prev_outputs',
        'prev_sum',
        'prev_day_mean',
        'prev_day_std',
        'prev_day_total',
        'slope_prev_day_n_n_minus_1',
        'slope_prev_day_n_minus_1_n_minus_2',
        'avg_change_rate',
        'prev_output',
        'prev_prev_output',
        'output_seq',
    ]

    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'random_state': 42
    }

    # 모든 패턴 확인
    all_patterns = df['pattern'].unique()
    patterns = all_patterns
    df_2 = df[df['pattern'].isin(patterns)].copy()
    df_cudf_2 = cudf.DataFrame.from_pandas(df_2)


    # 결과 저장 딕셔너리
    results_2 = {'all': {}}
    for pattern in patterns:
        results_2[pattern] = {}

    # 전체 데이터 모델
    X_all = df_cudf_2[features]
    y_all = df_cudf_2['amount']

    # XGBoost 모델 학습
    xgb_dtrain_all = xgb.DMatrix(X_all.values, label=y_all.values, feature_names=features)
    xgb_model_all = xgb.train(params, xgb_dtrain_all, num_boost_round=100)

    # 모델 저장
    model_path_all = os.path.join(model_dir_2, 'all.model')
    xgb_model_all.save_model(model_path_all)

    # K-Fold 교차 검증 - 간소화된 평가
    X_pd_all = df_2[features]
    y_pd_all = df_2['amount']

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores_all = []
    rmse_scores_all = []
    accuracy_all = []  # 10% 이내 오차 비율

    for train_idx, test_idx in kfold.split(X_pd_all):
        X_train, X_test = X_pd_all.iloc[train_idx], X_pd_all.iloc[test_idx]
        y_train, y_test = y_pd_all.iloc[train_idx], y_pd_all.iloc[test_idx]

        # 모델 학습 및 예측
        dtrain_cv = xgb.DMatrix(cp.asarray(X_train.values), label=cp.asarray(y_train.values))
        dtest_cv = xgb.DMatrix(cp.asarray(X_test.values))
        xgb_model_cv = xgb.train(params, dtrain_cv, num_boost_round=100)
        y_pred = cp.asnumpy(xgb_model_cv.predict(dtest_cv))

        # 평가 지표 계산
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # 10% 이내 오차 비율 계산
        error_percent = np.abs((y_test.values - y_pred) / y_test.values) * 100
        acc = np.mean(error_percent < 10) * 100  # 10% 이내 오차의 비율

        r2_scores_all.append(r2)
        rmse_scores_all.append(rmse)
        accuracy_all.append(acc)

    results_2['all']['r2'] = np.mean(r2_scores_all)
    results_2['all']['rmse'] = np.mean(rmse_scores_all)
    results_2['all']['accuracy'] = np.mean(accuracy_all)


    # 패턴별 모델
    for pattern in patterns:
        # 모델 파일 경로 확인
        model_path_pattern = os.path.join(model_dir_2, f'{pattern}.model')
        
        # 이미 모델 파일이 존재하면 이 패턴 스킵
        if os.path.exists(model_path_pattern):
            print(f"모델 파일 {pattern}.model이 이미 존재합니다. 이 패턴은 스킵합니다.")
            continue

        # 패턴별 데이터 필터링
        pattern_df = df_2[df_2['pattern'] == pattern]
        pattern_df_cudf = df_cudf_2[df_cudf_2['pattern'] == pattern]

        # 모델 학습
        X_pattern = pattern_df_cudf[features]
        y_pattern = pattern_df_cudf['amount']

        xgb_dtrain_pattern = xgb.DMatrix(X_pattern.values, label=y_pattern.values, feature_names=features)
        xgb_model_pattern = xgb.train(params, xgb_dtrain_pattern, num_boost_round=100)

        # 모델 저장
        xgb_model_pattern.save_model(model_path_pattern)
        

        # K-Fold 교차 검증
        X_pd_pattern = pattern_df[features]
        y_pd_pattern = pattern_df['amount']

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores_pattern = []
        rmse_scores_pattern = []
        accuracy_pattern = []

        for train_idx, test_idx in kfold.split(X_pd_pattern):
            X_train, X_test = X_pd_pattern.iloc[train_idx], X_pd_pattern.iloc[test_idx]
            y_train, y_test = y_pd_pattern.iloc[train_idx], y_pd_pattern.iloc[test_idx]

            # 모델 학습 및 예측
            dtrain_cv = xgb.DMatrix(cp.asarray(X_train.values), label=cp.asarray(y_train.values))
            dtest_cv = xgb.DMatrix(cp.asarray(X_test.values))
            xgb_model_cv = xgb.train(params, dtrain_cv, num_boost_round=100)
            y_pred = cp.asnumpy(xgb_model_cv.predict(dtest_cv))

            # 평가 지표 계산
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # 10% 이내 오차 비율 계산
            error_percent = np.abs((y_test.values - y_pred) / y_test.values) * 100
            acc = np.mean(error_percent < 10) * 100  # 10% 이내 오차의 비율

            r2_scores_pattern.append(r2)
            rmse_scores_pattern.append(rmse)
            accuracy_pattern.append(acc)

        results_2[pattern]['r2'] = np.mean(r2_scores_pattern)
        results_2[pattern]['rmse'] = np.mean(rmse_scores_pattern)
        results_2[pattern]['accuracy'] = np.mean(accuracy_pattern)

    # GPU 메모리 정리
    cp.get_default_memory_pool().free_all_blocks()