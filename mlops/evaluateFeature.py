# 이 파일은 생성된 feature를 검증하기 위한 파일입니다.
# https://colab.research.google.com/drive/16Me_hsywPcJm32TDeKEjJLS4T_ubWGPu#scrollTo=9D_SXoXjsILL

import cudf
import cupy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import os
def evaluate_feature(path, file):
    # GPU 기반 패턴별 XGBoost 분석 코드
    # -------------------------------
    # (1) 데이터 준비 및 시간 특성 변환
    # -------------------------------

    base_dir = f"{path}/VoID_WaterPurifier"
    os.chdir(base_dir)
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

    # GPU 데이터프레임으로 변환
    df_cudf = cudf.DataFrame.from_pandas(df)

    # -------------------------------
    # (2) 패턴별 분석 준비
    # -------------------------------
    patterns = df['pattern'].unique()

    # 전체 및 패턴별 결과 저장용 딕셔너리
    results = {
        'all': {},
        'A': {},
        'B': {},
        'C': {},
        'D': {}
    }

    # -------------------------------
    # (2-1) 다중공선성 분석 (VIF) - 전체 데이터
    # -------------------------------

    # cupy로 상관행렬 및 VIF 계산
    X_all_float64 = cp.array(df_cudf[features].values, dtype=cp.float64)
    X_all_centered = X_all_float64 - X_all_float64.mean(axis=0)
    Cov_all = (X_all_centered.T @ X_all_centered) / (X_all_float64.shape[0] - 1)
    std_all = cp.sqrt(cp.diag(Cov_all))
    corr_matrix_all = Cov_all / cp.outer(std_all, std_all)

    try:
        inv_corr_all = cp.linalg.inv(corr_matrix_all)
        vifs_all = cp.diag(inv_corr_all)
        vif_df_all = pd.DataFrame({
            'Feature': features,
            'VIF': cp.asnumpy(vifs_all)
        }).sort_values('VIF', ascending=False)

        results['all']['vif'] = vif_df_all

        # VIF 시각화
        plt.figure(figsize=(12, 8))
        sns.barplot(x='VIF', y='Feature', data=vif_df_all)
        plt.title('다중공선성 (VIF) - 전체 데이터')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        pass
    # -------------------------------
    # (2-2) 특성 간 상관관계 분석 - 전체 데이터
    # -------------------------------

    # 상관관계 행렬 계산 (전체 데이터)
    corr_all = df[features].corr()
    results['all']['corr'] = corr_all

    # 상관관계 히트맵 시각화
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_all, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('특성 간 상관관계 - 전체 데이터')
    plt.tight_layout()
    plt.show()

    # 높은 상관관계(|r| > 0.7)를 가진 특성 쌍 추출
    high_corr_pairs = []
    corr_matrix_values = corr_all.values
    feature_names = corr_all.columns.tolist()

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix_values[i, j]) > 0.7:
                high_corr_pairs.append({
                    'Feature1': feature_names[i],
                    'Feature2': feature_names[j],
                    'Correlation': corr_matrix_values[i, j]
                })


    # -------------------------------
    # (3) 전체 데이터 모델 (비교 기준)
    # -------------------------------
    X_all = df_cudf[features]
    y_all = df_cudf['amount']

    # XGBoost 특성 중요도 계산 - 전체 데이터
    xgb_dtrain_all = xgb.DMatrix(X_all.values, label=y_all.values, feature_names=features)
    params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'random_state': 42
    }
    xgb_model_all = xgb.train(params, xgb_dtrain_all, num_boost_round=100)

    # 특성 중요도 - 전체 데이터
    xgb_gain_all = xgb_model_all.get_score(importance_type='gain')
    xgb_importance_all = pd.DataFrame({
        'feature': list(xgb_gain_all.keys()),
        'importance': list(xgb_gain_all.values())
    }).sort_values('importance', ascending=False)
    results['all']['importance'] = xgb_importance_all


    X_pd_all = df[features]
    y_pd_all = df['amount']

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores_all = []
    mse_scores_all = []
    all_predictions = []  # 모든 예측 결과를 저장할 리스트

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_pd_all), 1):
        X_train, X_test = X_pd_all.iloc[train_idx], X_pd_all.iloc[test_idx]
        y_train, y_test = y_pd_all.iloc[train_idx], y_pd_all.iloc[test_idx]

        # XGBoost 파라미터
        params_cv = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'random_state': 42
        }

        dtrain_cv = xgb.DMatrix(cp.asarray(X_train.values), label=cp.asarray(y_train.values))
        dtest_cv = xgb.DMatrix(cp.asarray(X_test.values))

        xgb_model_cv = xgb.train(params_cv, dtrain_cv, num_boost_round=100)
        y_pred = xgb_model_cv.predict(dtest_cv)
        y_pred = cp.asnumpy(y_pred)

        # 예측 결과 저장
        fold_predictions = pd.DataFrame({
            'actual': y_test.values,
            'pred': y_pred,
            'fold': fold_idx
        })
        all_predictions.append(fold_predictions)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2_scores_all.append(r2)
        mse_scores_all.append(mse)

    # 모든 fold의 예측 결과를 하나의 DataFrame으로 합치기
    all_predictions_df = pd.concat(all_predictions)
    results['all']['predictions'] = all_predictions_df

    results['all']['r2'] = np.mean(r2_scores_all)
    results['all']['mse'] = np.mean(mse_scores_all)

    # -------------------------------
    # (4) 패턴별 XGBoost 모델링
    # -------------------------------
    for pattern in patterns:

        # 패턴별 데이터 필터링
        pattern_df = df[df['pattern'] == pattern]
        pattern_df_cudf = df_cudf[df_cudf['pattern'] == pattern]


        # -------------------------------
        # (4-1) 패턴별 다중공선성 분석 (VIF)
        # -------------------------------

        # cupy로 상관행렬 및 VIF 계산 - 패턴별
        try:
            X_pattern_float64 = cp.array(pattern_df_cudf[features].values, dtype=cp.float64)
            X_pattern_centered = X_pattern_float64 - X_pattern_float64.mean(axis=0)
            Cov_pattern = (X_pattern_centered.T @ X_pattern_centered) / (X_pattern_float64.shape[0] - 1)
            std_pattern = cp.sqrt(cp.diag(Cov_pattern))
            corr_matrix_pattern = Cov_pattern / cp.outer(std_pattern, std_pattern)

            inv_corr_pattern = cp.linalg.inv(corr_matrix_pattern)
            vifs_pattern = cp.diag(inv_corr_pattern)
            vif_df_pattern = pd.DataFrame({
                'Feature': features,
                'VIF': cp.asnumpy(vifs_pattern)
            }).sort_values('VIF', ascending=False)

            results[pattern]['vif'] = vif_df_pattern

            # 패턴별 VIF 시각화
            plt.figure(figsize=(12, 8))
            sns.barplot(x='VIF', y='Feature', data=vif_df_pattern)
            plt.title(f'다중공선성 (VIF) - 패턴 {pattern}')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            pass
        # -------------------------------
        # (4-2) 패턴별 상관관계 분석
        # -------------------------------

        # 패턴별 상관관계 행렬 계산
        corr_pattern = pattern_df[features].corr()
        results[pattern]['corr'] = corr_pattern

        # 패턴별 상관관계 히트맵 시각화
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_pattern, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(f'특성 간 상관관계 - 패턴 {pattern}')
        plt.tight_layout()
        plt.show()

        # 패턴별 상관관계-목표변수 분석
        corr_with_target_pattern = pd.DataFrame({
            'Feature': features,
            'Correlation_with_target': [pattern_df[feature].corr(pattern_df['amount']) for feature in features]
        }).sort_values('Correlation_with_target', ascending=False)
        results[pattern]['corr_with_target'] = corr_with_target_pattern


        # 패턴별 특성 중요도 계산
        X_pattern = pattern_df_cudf[features]
        y_pattern = pattern_df_cudf['amount']

        xgb_dtrain_pattern = xgb.DMatrix(X_pattern.values, label=y_pattern.values, feature_names=features)
        xgb_model_pattern = xgb.train(params, xgb_dtrain_pattern, num_boost_round=100)

        xgb_gain_pattern = xgb_model_pattern.get_score(importance_type='gain')
        xgb_importance_pattern = pd.DataFrame({
            'feature': list(xgb_gain_pattern.keys()),
            'importance': list(xgb_gain_pattern.values())
        }).sort_values('importance', ascending=False)
        results[pattern]['importance'] = xgb_importance_pattern

        # 패턴별 K-Fold 교차 검증
        X_pd_pattern = pattern_df[features]
        y_pd_pattern = pattern_df['amount']

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores_pattern = []
        mse_scores_pattern = []
        pattern_predictions = []  # 패턴별 예측 결과를 저장할 리스트

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_pd_pattern), 1):
            X_train, X_test = X_pd_pattern.iloc[train_idx], X_pd_pattern.iloc[test_idx]
            y_train, y_test = y_pd_pattern.iloc[train_idx], y_pd_pattern.iloc[test_idx]

            # XGBoost 파라미터
            params_cv = {
                'tree_method': 'hist',
                'device': 'cuda',
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'random_state': 42
            }

            dtrain_cv = xgb.DMatrix(cp.asarray(X_train.values), label=cp.asarray(y_train.values))
            dtest_cv = xgb.DMatrix(cp.asarray(X_test.values))

            xgb_model_cv = xgb.train(params_cv, dtrain_cv, num_boost_round=100)
            y_pred = xgb_model_cv.predict(dtest_cv)
            y_pred = cp.asnumpy(y_pred)

            # 예측 결과 저장
            fold_predictions = pd.DataFrame({
                'actual': y_test.values,
                'pred': y_pred,
                'fold': fold_idx
            })
            pattern_predictions.append(fold_predictions)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2_scores_pattern.append(r2)
            mse_scores_pattern.append(mse)

        # 모든 fold의 예측 결과를 하나의 DataFrame으로 합치기
        pattern_predictions_df = pd.concat(pattern_predictions)
        results[pattern]['predictions'] = pattern_predictions_df

        results[pattern]['r2'] = np.mean(r2_scores_pattern)
        results[pattern]['mse'] = np.mean(mse_scores_pattern)

    # -------------------------------
    # (5) 결과 비교 및 시각화
    # -------------------------------
    # 패턴별 성능 비교
    performance = pd.DataFrame({
        '패턴': ['전체'] + list(patterns),
        'R²': [results['all']['r2']] + [results[p]['r2'] for p in patterns],
        'MSE': [results['all']['mse']] + [results[p]['mse'] for p in patterns]
    })

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='패턴', y='R²', data=performance)
    plt.title('패턴별 R² 비교')
    plt.ylim(0.0, 1.0)  # 차이를 더 잘 보기 위한 범위 설정

    plt.subplot(1, 2, 2)
    sns.barplot(x='패턴', y='MSE', data=performance)
    plt.title('패턴별 MSE 비교')
    plt.tight_layout()
    plt.show()

    # 패턴별 특성 중요도 비교 시각화
    plt.figure(figsize=(20, 18))  # 높이 증가

    # 전체 데이터 특성 중요도
    plt.subplot(3, 2, 1)
    sns.barplot(x='importance', y='feature', data=results['all']['importance'].head(10))
    plt.title('전체 데이터 특성 중요도 (상위 10개)')

    # 패턴별 특성 중요도 (패턴이 4개이므로 3x2 그리드에 맞게 배치)
    for i, pattern in enumerate(patterns, 2):
        if i <= 6:  # 3x2 그리드에 맞게 조정
            plt.subplot(3, 2, i)
            sns.barplot(x='importance', y='feature', data=results[pattern]['importance'].head(10))
            plt.title(f'패턴 {pattern} 특성 중요도 (상위 10개)')

    plt.tight_layout()
    plt.show()

    # 패턴별 특성-목표변수 상관관계 비교
    plt.figure(figsize=(20, 15))

    # 상위 8개 특성만 선택하여 패턴별 비교
    top_features = list(set().union(
        results['all']['importance']['feature'].head(5),
        results['A']['importance']['feature'].head(5),
        results['B']['importance']['feature'].head(5),
        results['C']['importance']['feature'].head(5),
        results['D']['importance']['feature'].head(5)
    ))[:8]

    # 각 특성별로 패턴 간 중요도 비교
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 4, i)

        # 각 패턴에서 해당 특성의 중요도 추출
        importance_values = []
        for p in ['all'] + list(patterns):
            if feature in results[p]['importance']['feature'].values:
                imp = results[p]['importance'].loc[results[p]['importance']['feature'] == feature, 'importance'].values[
                    0]
            else:
                imp = 0
            importance_values.append(imp)

        # 패턴별 특성 중요도 시각화
        pattern_labels = ['전체'] + list(patterns)  # 모든 패턴 포함
        sns.barplot(x=pattern_labels, y=importance_values)
        plt.title(f'특성: {feature}')
        plt.xticks(rotation=45)  # 레이블이 많아져서 각도 조정

    plt.tight_layout()
    plt.show()



    # -------------------------------
    # (6) 실제/예측값 샘플 확인 및 비교 분석
    # -------------------------------
    import random

    # 각 패턴별로 임의의 100개 샘플 선택 및 오차 분석
    def analyze_predictions(pred_df, pattern_name):
        # 임의의 100개 샘플 선택
        sample_size = min(100, len(pred_df))  # 데이터가 100개 미만인 경우 대비
        random_samples = pred_df.sample(sample_size, random_state=42)

        # 오차 및 백분율 오차 계산
        random_samples['error'] = random_samples['actual'] - random_samples['pred']
        random_samples['error_percent'] = (random_samples['error'] / random_samples['actual']) * 100

        # 10% 이상 차이나는 예측의 수 계산
        high_error_count = sum(abs(random_samples['error_percent']) >= 10)
        low_error_count = sum(abs(random_samples['error_percent']) < 10)

        # 전체 데이터셋에서의 오차 분석
        pred_df['error'] = pred_df['actual'] - pred_df['pred']
        pred_df['error_percent'] = (pred_df['error'] / pred_df['actual']) * 100
        total_high_error = sum(abs(pred_df['error_percent']) >= 10)
        total_low_error = sum(abs(pred_df['error_percent']) < 10)
        total_samples = len(pred_df)



        # RMSE 계산
        rmse = np.sqrt(mean_squared_error(pred_df['actual'], pred_df['pred']))

        return {
            'pattern': pattern_name,
            'random_samples': random_samples,
            'high_error_count': high_error_count,
            'low_error_count': low_error_count,
            'total_high_error': total_high_error,
            'total_low_error': total_low_error,
            'total_samples': total_samples,
            'rmse': rmse
        }

    # 각 패턴별 분석 실행
    patterns_to_analyze = ['all'] + list(patterns)
    analysis_results = {}

    for pattern in patterns_to_analyze:
        pattern_name = '전체' if pattern == 'all' else f'패턴 {pattern}'
        analysis_results[pattern] = analyze_predictions(results[pattern]['predictions'], pattern_name)

    # 오차 분포 시각화
    plt.figure(figsize=(18, 12))

    # 각 패턴별 임의 샘플의 실제값 vs 예측값 비교
    plt.subplot(2, 2, 1)
    for i, pattern in enumerate(patterns_to_analyze):
        pattern_name = '전체' if pattern == 'all' else f'패턴 {pattern}'
        samples = analysis_results[pattern]['random_samples']
        plt.scatter(samples['actual'], samples['pred'], label=pattern_name, alpha=0.7)

    min_vals = [analysis_results[p]['random_samples']['actual'].min() for p in patterns_to_analyze]
    max_vals = [analysis_results[p]['random_samples']['actual'].max() for p in patterns_to_analyze]
    min_val = min(min_vals)
    max_val = max(max_vals)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')  # 대각선 추가
    plt.xlabel('실제값')
    plt.ylabel('예측값')
    plt.title('각 패턴별 임의 샘플의 실제값 vs 예측값')
    plt.legend()

    # 10% 이상/미만 오차 비율 바 차트
    plt.subplot(2, 2, 2)
    pattern_labels = ['전체'] + [f'패턴 {p}' for p in patterns]
    high_error_pcts = [analysis_results[p]['total_high_error'] / analysis_results[p]['total_samples'] * 100
                       for p in patterns_to_analyze]
    low_error_pcts = [analysis_results[p]['total_low_error'] / analysis_results[p]['total_samples'] * 100
                      for p in patterns_to_analyze]

    x = np.arange(len(pattern_labels))
    width = 0.35
    plt.bar(x - width / 2, low_error_pcts, width, label='10% 미만 오차')
    plt.bar(x + width / 2, high_error_pcts, width, label='10% 이상 오차')
    plt.xlabel('패턴')
    plt.ylabel('비율 (%)')
    plt.title('각 패턴별 오차 비율')
    plt.xticks(x, pattern_labels)
    plt.legend()

    # 각 패턴별 오차 분포 (boxplot)
    plt.subplot(2, 2, 3)
    error_data = []
    labels = []
    for pattern in patterns_to_analyze:
        pattern_name = '전체' if pattern == 'all' else f'패턴 {pattern}'
        error_data.append(analysis_results[pattern]['random_samples']['error_percent'])
        labels.append(pattern_name)
    plt.boxplot(error_data, labels=labels)
    plt.axhline(y=10, color='r', linestyle='--')
    plt.axhline(y=-10, color='r', linestyle='--')
    plt.ylabel('백분율 오차 (%)')
    plt.title('각 패턴별 임의 샘플의 백분율 오차 분포')

    # RMSE 비교
    plt.subplot(2, 2, 4)
    rmse_values = [analysis_results[p]['rmse'] for p in patterns_to_analyze]
    sns.barplot(x=pattern_labels, y=rmse_values)
    plt.xlabel('패턴')
    plt.ylabel('RMSE')
    plt.title('각 패턴별 RMSE')

    plt.tight_layout()
    plt.show()

    # 오차 히스토그램 그래프
    plt.figure(figsize=(18, 15))
    rows = 3
    cols = 2  # 전체 + A,B,C,D = 5개 항목, 3x2 그리드로 표시
    for i, pattern in enumerate(patterns_to_analyze, 1):
        pattern_name = '전체' if pattern == 'all' else f'패턴 {pattern}'
        plt.subplot(rows, cols, i)

        # 전체 데이터의 오차 퍼센트 분포
        error_percent = results[pattern]['predictions']['error_percent']

        sns.histplot(error_percent, bins=30, kde=True)
        plt.axvline(x=10, color='r', linestyle='--')
        plt.axvline(x=-10, color='r', linestyle='--')
        plt.xlabel('백분율 오차 (%)')
        plt.ylabel('빈도')
        plt.title(f'{pattern_name} 오차 분포')

    plt.tight_layout()
    plt.show()

    # 패턴별 특성 중요도와 VIF 비교 시각화
    # (추가된 부분)
    plt.figure(figsize=(20, 16))

    # 각 패턴별 상위 5개 중요 특성과 해당 VIF 값 비교
    for i, pattern in enumerate(patterns_to_analyze):
        pattern_name = '전체' if pattern == 'all' else f'패턴 {pattern}'

        if pattern in results and 'importance' in results[pattern] and 'vif' in results[pattern]:
            plt.subplot(3, 2, i + 1)

            # 상위 5개 중요 특성 추출
            top_features = results[pattern]['importance'].head(5)['feature'].values

            # 해당 특성의 VIF 값 추출
            vif_values = []
            for feature in top_features:
                vif_row = results[pattern]['vif'][results[pattern]['vif']['Feature'] == feature]
                if not vif_row.empty:
                    vif_values.append(vif_row['VIF'].values[0])
                else:
                    vif_values.append(0)

            # 막대 그래프로 표시
            plt.bar(top_features, vif_values)
            plt.title(f'{pattern_name} 상위 5개 중요 특성의 VIF 값')
            plt.xticks(rotation=45)
            plt.ylabel('VIF')

    plt.tight_layout()
    plt.show()

    # GPU 메모리 정리
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        pass