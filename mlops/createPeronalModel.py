# =====================================================================
# "검증/홀드아웃" 모두 해당 사용자 데이터만 쓰도록 수정한 코드 예시
# + 파라미터를 고정 + Base/Personal/Ensemble 테스트 예측값 print
# =====================================================================
import matplotlib.pyplot as plt
import matplotlib as mpl

import cudf
import cupy as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


def createPersonalModel(userId):
    base_dir = "/content/drive/My Drive/Colab Notebooks/VoID_WaterPurifier"
    os.chdir(base_dir)

    juingong=userId
    # 모델 저장 디렉토리
    model_dir = os.path.join(base_dir, "models")
    model_dir_2 = os.path.join(model_dir, "2")  # 패턴 A, B, C, D 모델 디렉토리
    ensemble_dir = os.path.join(model_dir, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    print(f"앙상블 모델 저장 디렉토리: {ensemble_dir}")

    # ====== 고정된 XGB 파라미터 ======
    FIXED_PARAMS = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42
    }

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

    file = 'all_features_merged.csv'
    full_data = pd.read_csv(file)
    full_data = full_data[full_data['amount'] > 0]

    # 시간 특성
    full_data['time_in_minutes'] = full_data['hour']*60 + full_data['minute']
    full_data['time_sin'] = np.sin(2*np.pi*full_data['time_in_minutes']/1440)
    full_data['time_cos'] = np.cos(2*np.pi*full_data['time_in_minutes']/1440)

    if 'date' in full_data.columns:
        full_data['date'] = pd.to_datetime(full_data['date'])
    else:
        start_date = datetime(2022,2,1)
        full_data['date'] = start_date

    patterns = full_data['pattern'].unique()

    pattern_users = {}
    for pattern in patterns:
        users = full_data[full_data['pattern']==pattern]['person_id'].unique()
        pattern_users[pattern] = users

    selected_users = {}
    for pattern in patterns:
        # 만약 패턴에 사용자 ID=23이 존재하면, selected_users[pattern] = [23]
        if juingong in pattern_users[pattern]:
            selected_users[pattern] = [juingong]
        else:
            # pattern에 juingong=23이 없다면 빈 리스트
            selected_users[pattern] = []

    base_models = {}
    for pattern in patterns:
        model_path = os.path.join(model_dir_2, f"{pattern}.model")
        if os.path.exists(model_path):
            booster = xgb.Booster()
            booster.load_model(model_path)
            base_models[pattern] = booster

    all_model_path = os.path.join(model_dir_2, "all.model")
    if os.path.exists(all_model_path):
        booster = xgb.Booster()
        booster.load_model(all_model_path)
        base_models['all'] = booster

    def split_data(user_data):
        sorted_dates = sorted(user_data['date'].unique())
        total_days = len(sorted_dates)
        if total_days >= 29:
            selected_dates = sorted_dates[:29]
            dev_dates = selected_dates[:25]
            holdout_dates = selected_dates[25:]
            train_dates = dev_dates[:20]
            valid_dates = dev_dates[20:]

        train_data = user_data[user_data['date'].isin(train_dates)]
        valid_data = user_data[user_data['date'].isin(valid_dates)]
        holdout_data = user_data[user_data['date'].isin(holdout_dates)]
        return train_data, valid_data, holdout_data

    # (수정) 고정 파라미터로 모델 생성
    def train_user_model(X_train, y_train):
        """
        개인화 모델 학습 - 고정된 파라미터로 학습
        """
        model = xgb.XGBRegressor(**FIXED_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        return model

    def find_best_ensemble_weight(base_pred, personal_pred, y_valid):
        best_alpha = 0
        best_score = -999
        for alpha in np.arange(0,1.01,0.01):
            ensemble = alpha*base_pred + (1-alpha)*personal_pred
            r2_ = r2_score(y_valid,ensemble)
            if r2_>best_score:
                best_score = r2_
                best_alpha = alpha
        return best_alpha,best_score

    def evaluate_ensemble(base_model, personal_model, X_valid, y_valid, X_holdout, y_holdout,
                        dataset_name="", do_dynamic_weight=True):
        dtest_valid = xgb.DMatrix(X_valid)
        base_pred_valid = cp.asnumpy(base_model.predict(dtest_valid))
        personal_pred_valid = personal_model.predict(X_valid)

        # 앙상블 가중치 찾기 (생략)
        if do_dynamic_weight:
            best_alpha, best_r2 = find_best_ensemble_weight(base_pred_valid, personal_pred_valid, y_valid)
        else:
            best_alpha = 0.7

        ensemble_valid = best_alpha*base_pred_valid + (1-best_alpha)*personal_pred_valid

        # (검증) 결과 출력
        valid_metrics = {}
        for name, pred in zip(['base','personal','ensemble'],
                            [base_pred_valid, personal_pred_valid, ensemble_valid]):
            r2_ = r2_score(y_valid, pred)
            rmse_ = np.sqrt(mean_squared_error(y_valid, pred))
            err_pct = np.abs((y_valid - pred) / y_valid) * 100
            acc_10 = np.mean(err_pct < 10) * 100
            valid_metrics[name] = {'r2': r2_, 'rmse': rmse_, 'accuracy': acc_10}

        # (홀드아웃) 평가
        holdout_metrics = None
        if X_holdout is not None and len(X_holdout) > 0:
            dtest_holdout = xgb.DMatrix(X_holdout)
            base_pred_holdout = cp.asnumpy(base_model.predict(dtest_holdout))
            personal_pred_holdout = personal_model.predict(X_holdout)
            ensemble_holdout = best_alpha*base_pred_holdout + (1-best_alpha)*personal_pred_holdout

            holdout_metrics = {}
            for name, pred in zip(['base','personal','ensemble'],
                                [base_pred_holdout, personal_pred_holdout, ensemble_holdout]):
                r2_ = r2_score(y_holdout, pred)
                rmse_ = np.sqrt(mean_squared_error(y_holdout, pred))
                err_pct = np.abs((y_holdout - pred) / y_holdout) * 100
                acc_10 = np.mean(err_pct < 10) * 100
                holdout_metrics[name] = {'r2': r2_, 'rmse': rmse_, 'accuracy': acc_10}

            df_holdout = pd.DataFrame([
                {
                    "Model": model_name.title(),
                    "R2": holdout_metrics[model_name]['r2'],
                    "RMSE": holdout_metrics[model_name]['rmse'],
                    "Accuracy_±10%": holdout_metrics[model_name]['accuracy']
                }
                for model_name in ['base','personal','ensemble']
            ])

        return valid_metrics, holdout_metrics, best_alpha

    # 메인 실행
    results = {}
    for pattern in patterns:
        if pattern not in base_models:
            continue

        base_model = base_models[pattern]
        pattern_results = {}

        for user_id in selected_users[pattern]:

            user_data = full_data[
                (full_data['person_id']==user_id) &
                (full_data['pattern']==pattern)
            ].copy()

            if len(user_data)<10:
                continue

            # 날짜 기준 분할
            train_user, valid_user, holdout_user = split_data(user_data)

            # (선택) 패턴 내 다른 사용자 샘플 (가중치 낮게) - 예시
            others_data = full_data[
                (full_data['pattern']==pattern) & 
                (full_data['person_id']!=user_id)
            ]
            others_data = others_data.sample(frac=0.2, random_state=42)  # 20% 샘플
            others_data['weight']=0.2
            train_user['weight']=1.0

            # 최종 학습 세트
            train_data = pd.concat([train_user, others_data], ignore_index=True)
            valid_data = valid_user
            holdout_data = holdout_user
            X_train = train_data[features]
            y_train = train_data['amount']
            X_valid = valid_data[features]
            y_valid = valid_data['amount']
            X_holdout = holdout_data[features]
            y_holdout = holdout_data['amount']

            personal_model = train_user_model(X_train, y_train)
            # (추가) 모델 저장
            personal_model_path = os.path.join(ensemble_dir, f"personal_{pattern}_{user_id}.model")
            personal_model.save_model(personal_model_path)
            
            # 검증 성능
            valid_pred = personal_model.predict(X_valid)
            v_r2 = r2_score(y_valid, valid_pred)
            v_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

            # 앙상블 평가
            valid_metrics, holdout_metrics, best_alpha = evaluate_ensemble(
                base_model, personal_model,
                X_valid, y_valid,
                X_holdout, y_holdout,
                dataset_name=f"user_{user_id}_pattern_{pattern}",
                do_dynamic_weight=True
            )

            # 시각화(검증)
            if valid_metrics is not None:
                valid_preds_dict = {
                    'base': cp.asnumpy(base_model.predict(xgb.DMatrix(X_valid))),
                    'personal': valid_pred,
                    'ensemble': best_alpha*cp.asnumpy(base_model.predict(xgb.DMatrix(X_valid))) \
                                + (1-best_alpha)*valid_pred
                }

            # 시각화(홀드아웃)
            if holdout_metrics is not None and len(X_holdout)>0:
                holdout_base_pred = cp.asnumpy(base_model.predict(xgb.DMatrix(X_holdout)))
                holdout_personal_pred = personal_model.predict(X_holdout)
                holdout_preds_dict = {
                    'base': holdout_base_pred,
                    'personal': holdout_personal_pred,
                    'ensemble': best_alpha*holdout_base_pred + (1-best_alpha)*holdout_personal_pred
                }

            pattern_results[user_id] = {
                'best_alpha': best_alpha,
                'valid_metrics': valid_metrics,
                'holdout_metrics': holdout_metrics
            }

        results[pattern] = pattern_results

    summary_data=[]
    for pattern, p_results in results.items():
        for user_id, user_res in p_results.items():
            vm = user_res['valid_metrics']
            hm = user_res['holdout_metrics']
            alpha = user_res['best_alpha']

            row = {
                'Pattern': pattern,
                'User_ID': user_id,
                'Alpha': alpha
            }
            if vm is not None:
                row.update({
                    'Valid_Base_R2': vm['base']['r2'],
                    'Valid_Personal_R2': vm['personal']['r2'],
                    'Valid_Ensemble_R2': vm['ensemble']['r2'],
                    'Valid_Base_RMSE': vm['base']['rmse'],
                    'Valid_Personal_RMSE': vm['personal']['rmse'],
                    'Valid_Ensemble_RMSE': vm['ensemble']['rmse'],
                    'Valid_Base_Acc': vm['base']['accuracy'],
                    'Valid_Personal_Acc': vm['personal']['accuracy'],
                    'Valid_Ensemble_Acc': vm['ensemble']['accuracy'],
                })
            if hm is not None:
                row.update({
                    'Holdout_Base_R2': hm['base']['r2'],
                    'Holdout_Personal_R2': hm['personal']['r2'],
                    'Holdout_Ensemble_R2': hm['ensemble']['r2'],
                    'Holdout_Base_RMSE': hm['base']['rmse'],
                    'Holdout_Personal_RMSE': hm['personal']['rmse'],
                    'Holdout_Ensemble_RMSE': hm['ensemble']['rmse'],
                    'Holdout_Base_Acc': hm['base']['accuracy'],
                    'Holdout_Personal_Acc': hm['personal']['accuracy'],
                    'Holdout_Ensemble_Acc': hm['ensemble']['accuracy'],
                })
            summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(ensemble_dir, 'ensemble_summary_user_only.csv')
        summary_df.to_csv(summary_path,index=False)

        # 패턴별 평균 결과
        psum = summary_df.groupby('Pattern').mean(numeric_only=True).reset_index()
        for _,row in psum.iterrows():
            pat = row['Pattern']