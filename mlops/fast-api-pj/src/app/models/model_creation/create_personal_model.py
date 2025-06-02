# =====================================================================
# "검증/홀드아웃" 모두 해당 사용자 데이터만 쓰도록 수정한 코드 예시
# + 파라미터를 고정 + Base/Personal/Ensemble 테스트 예측값 print
# =====================================================================
import matplotlib.pyplot as plt
import matplotlib as mpl

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
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
warnings.filterwarnings('ignore')


class PersonalModelCreator:
    """
    개인화 모델 생성 및 관리 클래스
    """
    def __init__(self):
        """
        개인화 모델 생성 클래스 초기화
        """
        # 특성 목록 정의 (기본값, 나중에 동적으로 변경 가능)
        self._features = [
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
        
        # DQM 기준값 설정
        self.R2_THRESHOLD = 0.5       # DQM 문서 기준: R2 Score > 0.5
        self.ACCURACY_THRESHOLD = 60  # DQM 문서 기준: 정확도 ≥ 60%
        self.ENSEMBLE_IMPROVEMENT_THRESHOLD = 0.05  # DQM 문서 기준: 5% 성능 향상
        
        # 앙상블 가중치 범위 (DQM 문서 기준: α값 0.1~0.9 범위)
        self.ALPHA_RANGE = np.arange(0.1, 1.0, 0.1)  # 0.1~0.9, 0.1 간격
        
        # XGB 파라미터 설정
        self.DEFAULT_PARAMS = {
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
        
        # 모델 롤백 기준 (DQM 문서 기준: 2% 이상 성능 하락 시 롤백)
        self.ROLLBACK_THRESHOLD = 0.02  # 2% 이상 성능 하락 시 롤백
        
        # 재시도 관련 설정
        self.max_retries = 3
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'performance_history': {},
            'rollback_history': {}
        }
        
    @property
    def features(self):
        """특성 목록 속성 getter"""
        return self._features
        
    @features.setter
    def features(self, features_list):
        """특성 목록 속성 setter"""
        if not features_list or len(features_list) < 2:
            print("경고: 유효하지 않은 특성 목록입니다. 기본 특성을 사용합니다.")
            return
        
        self._features = features_list
        
    def setup_directories(self, base_dir: str) -> Dict[str, str]:
        """
        모델 저장을 위한 디렉토리 설정
        
        Args:
            base_dir: 기본 디렉토리 경로
            
        Returns:
            dict: 모델 디렉토리 정보
        """
        # 모델 저장 디렉토리
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_dir_2 = os.path.join(model_dir, "2")  # 패턴 A, B, C, D 모델 디렉토리
        os.makedirs(model_dir_2, exist_ok=True)
        
        ensemble_dir = os.path.join(model_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # 버전 관리 디렉토리
        version_dir = os.path.join(ensemble_dir, "versions")
        os.makedirs(version_dir, exist_ok=True)
        
        # 롤백 디렉토리
        rollback_dir = os.path.join(ensemble_dir, "rollback")
        os.makedirs(rollback_dir, exist_ok=True)
        
        print(f"앙상블 모델 저장 디렉토리: {ensemble_dir}")
        print(f"버전 관리 디렉토리: {version_dir}")
        
        return {
            "model_dir": model_dir,
            "model_dir_2": model_dir_2,
            "ensemble_dir": ensemble_dir,
            "version_dir": version_dir,
            "rollback_dir": rollback_dir
        }
        
    def preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        데이터 전처리 수행
        
        Args:
            file_path: CSV 파일 경로
            
        Returns:
            DataFrame: 전처리된 데이터
        """
        # 데이터 로드
        full_data = pd.read_csv(file_path)
        full_data = full_data[full_data['amount'] > 0]

        # 시간 특성
        full_data['time_in_minutes'] = full_data['hour'] * 60 + full_data['minute']
        full_data['time_sin'] = np.sin(2 * np.pi * full_data['time_in_minutes'] / 1440)
        full_data['time_cos'] = np.cos(2 * np.pi * full_data['time_in_minutes'] / 1440)

        if 'date' in full_data.columns:
            full_data['date'] = pd.to_datetime(full_data['date'])
        else:
            start_date = datetime(2022, 2, 1)
            full_data['date'] = start_date
            
        return full_data
    
    def load_base_models(self, model_dir: str) -> Dict[str, xgb.Booster]:
        """
        기본 모델 로드
        
        Args:
            model_dir: 모델 디렉토리 경로
            
        Returns:
            dict: 패턴별 베이스 모델
        """
        base_models = {}
        patterns = ['A', 'B', 'C', 'D']
        
        # 패턴별 모델 로드
        for pattern in patterns:
            model_path = os.path.join(model_dir, f"{pattern}.model")
            if os.path.exists(model_path):
                booster = xgb.Booster()
                booster.load_model(model_path)
                base_models[pattern] = booster
                print(f"패턴 {pattern} 기본 모델 로드 완료")
        
        # ALL 모델 로드
        all_model_path = os.path.join(model_dir, "all.model")
        if os.path.exists(all_model_path):
            booster = xgb.Booster()
            booster.load_model(all_model_path)
            base_models['all'] = booster
            print("ALL 기본 모델 로드 완료")
            
        if not base_models:
            print("경고: 로드된 기본 모델이 없습니다.")
            
        return base_models
    
    def split_data(self, user_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        사용자 데이터를 학습/검증/홀드아웃 세트로 분할
        
        Args:
            user_data: 사용자 데이터
            
        Returns:
            tuple: (학습 데이터, 검증 데이터, 홀드아웃 데이터)
        """
        sorted_dates = sorted(user_data['date'].unique())
        total_days = len(sorted_dates)
        
        if total_days >= 29:
            selected_dates = sorted_dates[:29]
            dev_dates = selected_dates[:25]
            holdout_dates = selected_dates[25:]
            train_dates = dev_dates[:20]
            valid_dates = dev_dates[20:]
        else:
            # 데이터가 충분하지 않은 경우의 처리
            if total_days >= 5:
                # 최소한 5일 이상 데이터가 있는 경우
                train_size = max(int(total_days * 0.7), 3)  # 최소 3일
                valid_size = max(int(total_days * 0.15), 1)  # 최소 1일
                
                train_dates = sorted_dates[:train_size]
                valid_dates = sorted_dates[train_size:train_size+valid_size]
                holdout_dates = sorted_dates[train_size+valid_size:]
            else:
                # 매우 적은 데이터의 경우
                print(f"경고: 데이터 일수가 매우 적습니다 ({total_days}일). 임의 분할을 수행합니다.")
                return train_test_split(
                    user_data, 
                    test_size=0.3, 
                    random_state=42,
                    stratify=user_data['date'] if len(user_data['date'].unique()) > 1 else None
                )

        train_data = user_data[user_data['date'].isin(train_dates)]
        valid_data = user_data[user_data['date'].isin(valid_dates)]
        holdout_data = user_data[user_data['date'].isin(holdout_dates)]
        
        return train_data, valid_data, holdout_data
    
    def train_user_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """
        개인화 모델 학습
        
        Args:
            X_train: 학습 특성 데이터
            y_train: 학습 라벨 데이터
            params: XGBoost 파라미터 (None일 경우 기본값 사용)
            
        Returns:
            XGBRegressor: 학습된 모델
        """
        if params is None:
            params = self.DEFAULT_PARAMS.copy()
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        return model
    
    def find_best_ensemble_weight(self, base_pred: np.ndarray, personal_pred: np.ndarray, 
                                y_valid: Union[pd.Series, np.ndarray]) -> Tuple[float, float]:
        """
        최적의 앙상블 가중치 탐색
        
        Args:
            base_pred: 기본 모델 예측값
            personal_pred: 개인 모델 예측값
            y_valid: 실제 값
            
        Returns:
            tuple: (최적 가중치, 최고 R2 점수)
        """
        best_alpha = 0
        best_score = -float('inf')
        
        for alpha in self.ALPHA_RANGE:
            ensemble = alpha * base_pred + (1 - alpha) * personal_pred
            r2_ = r2_score(y_valid, ensemble)
            if r2_ > best_score:
                best_score = r2_
                best_alpha = alpha
                
        return best_alpha, best_score
    
    def evaluate_ensemble(self, base_model: xgb.Booster, personal_model: xgb.XGBRegressor,
                        X_valid: pd.DataFrame, y_valid: pd.Series,
                        X_holdout: Optional[pd.DataFrame] = None, 
                        y_holdout: Optional[pd.Series] = None,
                        dataset_name: str = "", 
                        do_dynamic_weight: bool = True) -> Tuple[Dict, Optional[Dict], float]:
        """
        앙상블 모델 평가
        
        Args:
            base_model: 기본 모델
            personal_model: 개인화 모델
            X_valid: 검증 특성 데이터
            y_valid: 검증 라벨 데이터
            X_holdout: 홀드아웃 특성 데이터 (옵션)
            y_holdout: 홀드아웃 라벨 데이터 (옵션)
            dataset_name: 데이터셋 이름 (로깅용)
            do_dynamic_weight: 동적 가중치 탐색 여부
            
        Returns:
            tuple: (검증 지표, 홀드아웃 지표, 최적 가중치)
        """
        # 검증 세트 예측
        dtest_valid = xgb.DMatrix(X_valid)
        base_pred_valid = cp.asnumpy(base_model.predict(dtest_valid))
        personal_pred_valid = personal_model.predict(X_valid)
        
        # 앙상블 가중치 찾기
        if do_dynamic_weight:
            best_alpha, best_r2 = self.find_best_ensemble_weight(
                base_pred_valid, personal_pred_valid, y_valid
            )
        else:
            best_alpha = 0.7  # 기본 가중치
            
        # 앙상블 예측
        ensemble_valid = best_alpha * base_pred_valid + (1 - best_alpha) * personal_pred_valid
        
        # 검증 성능 계산
        valid_metrics = {}
        for name, pred in zip(['base', 'personal', 'ensemble'],
                            [base_pred_valid, personal_pred_valid, ensemble_valid]):
            r2_ = r2_score(y_valid, pred)
            rmse_ = np.sqrt(mean_squared_error(y_valid, pred))
            err_pct = np.abs((y_valid - pred) / y_valid) * 100
            acc_10 = np.mean(err_pct < 10) * 100
            valid_metrics[name] = {'r2': r2_, 'rmse': rmse_, 'accuracy': acc_10}
            
        # 홀드아웃 평가 (선택적)
        holdout_metrics = None
        if X_holdout is not None and len(X_holdout) > 0 and y_holdout is not None and len(y_holdout) > 0:
            dtest_holdout = xgb.DMatrix(X_holdout)
            base_pred_holdout = cp.asnumpy(base_model.predict(dtest_holdout))
            personal_pred_holdout = personal_model.predict(X_holdout)
            ensemble_holdout = best_alpha * base_pred_holdout + (1 - best_alpha) * personal_pred_holdout

            holdout_metrics = {}
            for name, pred in zip(['base', 'personal', 'ensemble'],
                                [base_pred_holdout, personal_pred_holdout, ensemble_holdout]):
                r2_ = r2_score(y_holdout, pred)
                rmse_ = np.sqrt(mean_squared_error(y_holdout, pred))
                err_pct = np.abs((y_holdout - pred) / y_holdout) * 100
                acc_10 = np.mean(err_pct < 10) * 100
                holdout_metrics[name] = {'r2': r2_, 'rmse': rmse_, 'accuracy': acc_10}
                
        return valid_metrics, holdout_metrics, best_alpha
    
    def adapt_hyperparameters(self, r2: float, acc: float, 
                            current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        성능에 따라 하이퍼파라미터 자동 조정
        
        Args:
            r2: R2 점수
            acc: 정확도 (%)
            current_params: 현재 파라미터
            
        Returns:
            dict: 조정된 파라미터
        """
        params = current_params.copy()
        
        # R2 점수가 기준보다 낮은 경우
        if r2 < self.R2_THRESHOLD:
            # 모델 복잡도 증가
            if params.get('max_depth', 5) < 8:
                params['max_depth'] = params.get('max_depth', 5) + 1
                
            # 트리 수 증가
            params['n_estimators'] = int(params.get('n_estimators', 1000) * 1.2)
            
        # 정확도가 기준보다 낮은 경우
        if acc < self.ACCURACY_THRESHOLD:
            # 학습률 조정
            if params.get('learning_rate', 0.1) > 0.02:
                params['learning_rate'] = params.get('learning_rate', 0.1) / 1.5
            
            # 정규화 파라미터 조정
            params['reg_lambda'] = params.get('reg_lambda', 1) * 1.2
            
        return params
    
    def check_performance_improvement(self, personal_metrics: Dict, ensemble_metrics: Dict) -> bool:
        """
        앙상블이 개인 모델 대비 성능 향상이 있는지 확인
        
        Args:
            personal_metrics: 개인 모델 성능 지표
            ensemble_metrics: 앙상블 모델 성능 지표
            
        Returns:
            bool: 성능 향상 여부
        """
        # DQM 문서 기준: 개인 모델 대비 5% 이상 성능 향상
        r2_improvement = ensemble_metrics['r2'] - personal_metrics['r2']
        
        # 정확도 기준 성능 비교
        acc_improvement = ensemble_metrics['accuracy'] - personal_metrics['accuracy']
        
        # 최소 하나의 지표에서 유의미한 개선이 있는지 확인
        if r2_improvement >= self.ENSEMBLE_IMPROVEMENT_THRESHOLD or \
           acc_improvement >= (self.ENSEMBLE_IMPROVEMENT_THRESHOLD * 100):
            return True
            
        return False
    
    def save_ensemble_info(self, ensemble_dir: str, pattern: str, user_id: int,
                         personal_model: xgb.XGBRegressor, alpha: float, 
                         metrics: Dict, version: int = 1) -> str:
        """
        앙상블 모델 정보 저장
        
        Args:
            ensemble_dir: 앙상블 디렉토리
            pattern: 패턴 ID
            user_id: 사용자 ID
            personal_model: 개인화 모델
            alpha: 앙상블 가중치
            metrics: 성능 지표
            version: 모델 버전
            
        Returns:
            str: 저장된 모델 경로
        """
        # 모델 저장 (DQM 문서 기준: [사용자ID]_[군집ID]_[생성날짜]_v[버전번호] 형식)
        current_date = datetime.now().strftime('%Y%m%d')
        model_path = os.path.join(
            ensemble_dir, 
            f"{user_id}_{pattern}_{current_date}_v{version}.model"
        )
        personal_model.save_model(model_path)
        
        # 메타데이터 저장
        metadata = {
            'user_id': user_id,
            'pattern': pattern,
            'alpha': alpha,
            'performance': metrics,
            'version': version,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        meta_path = os.path.join(ensemble_dir, "versions", f"{user_id}_{pattern}_{current_date}_v{version}.meta")
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"모델 및 메타데이터 저장 완료: {model_path}, {meta_path}")
        return model_path
    
    def compare_with_previous_model(self, ensemble_dir: str, pattern: str, user_id: int,
                                   current_metrics: Dict, version: int) -> bool:
        """
        이전 버전 모델과 성능 비교
        
        Args:
            ensemble_dir: 앙상블 디렉토리
            pattern: 패턴 ID
            user_id: 사용자 ID
            current_metrics: 현재 모델 성능
            version: 현재 모델 버전
            
        Returns:
            bool: 롤백 필요 여부
        """
        if version <= 1:
            return False  # 첫 번째 버전은 비교 대상 없음
            
        # 파이프라인 상태 업데이트
        self.pipeline_status['current_stage'] = 'comparing_with_previous_model'
        
        # 이전 버전 찾기 (날짜가 포함된 버전 형식 고려)
        versions_dir = os.path.join(ensemble_dir, "versions")
        prev_meta_files = [
            f for f in os.listdir(versions_dir) 
            if f.startswith(f"{user_id}_{pattern}_") and f.endswith(f"_v{version-1}.meta")
        ]
        
        if not prev_meta_files:
            return False
            
        # 가장 최근 이전 버전 선택
        prev_meta_path = os.path.join(versions_dir, sorted(prev_meta_files)[-1])
        
        try:
            with open(prev_meta_path, 'rb') as f:
                prev_metadata = pickle.load(f)
                
            # 이전 모델 성능
            prev_metrics = prev_metadata.get('performance', {}).get('holdout', {}).get('ensemble', {})
            
            if not prev_metrics:
                return False
                
            # 현재 모델 성능
            curr_metrics = current_metrics.get('holdout', {}).get('ensemble', {})
            
            if not curr_metrics:
                return False
                
            # DQM 기준: 2% 이상 성능 하락 시 롤백
            r2_decline = prev_metrics.get('r2', 0) - curr_metrics.get('r2', 0)
            accuracy_decline = prev_metrics.get('accuracy', 0) - curr_metrics.get('accuracy', 0)
            
            # 롤백 결정 및 로깅
            needs_rollback = False
            reason = ""
            
            if r2_decline > self.ROLLBACK_THRESHOLD:
                needs_rollback = True
                reason = f"R2 점수 하락: {r2_decline:.4f} (이전: {prev_metrics.get('r2', 0):.4f}, 현재: {curr_metrics.get('r2', 0):.4f})"
                print(f"성능 저하 감지! {reason}")
                
            if accuracy_decline > self.ROLLBACK_THRESHOLD * 100:
                needs_rollback = True
                reason += f" 정확도 하락: {accuracy_decline:.2f}% (이전: {prev_metrics.get('accuracy', 0):.2f}%, 현재: {curr_metrics.get('accuracy', 0):.2f}%)"
                print(f"성능 저하 감지! 정확도 하락: {accuracy_decline:.2f}%")
            
            # 파이프라인 상태 업데이트
            if needs_rollback:
                self.pipeline_status['current_stage'] = 'rollback_needed'
                self.pipeline_status['rollback_history'][f"{user_id}_{pattern}_v{version}"] = {
                    'reason': reason,
                    'prev_metrics': prev_metrics,
                    'curr_metrics': curr_metrics
                }
                
            return needs_rollback
                
        except Exception as e:
            print(f"이전 모델 비교 중 오류 발생: {str(e)}")
            
        return False
    
    def rollback_model(self, ensemble_dir: str, pattern: str, user_id: int, version: int) -> bool:
        """
        이전 버전 모델로 롤백
        
        Args:
            ensemble_dir: 앙상블 디렉토리
            pattern: 패턴 ID
            user_id: 사용자 ID
            version: 현재 모델 버전
            
        Returns:
            bool: 롤백 성공 여부
        """
        if version <= 1:
            return False
        
        self.pipeline_status['current_stage'] = 'rolling_back_model'
        
        # 현재 및 이전 버전 파일 찾기
        versions_dir = os.path.join(ensemble_dir, "versions")
        curr_date_pattern = datetime.now().strftime('%Y%m%d')
        
        # 현재 버전 파일
        curr_model_files = [
            f for f in os.listdir(ensemble_dir)
            if f.startswith(f"{user_id}_{pattern}_") and f.endswith(f"_v{version}.model")
        ]
        
        # 이전 버전 파일
        prev_model_files = [
            f for f in os.listdir(ensemble_dir)
            if f.startswith(f"{user_id}_{pattern}_") and f.endswith(f"_v{version-1}.model")
        ]
        
        if not prev_model_files or not curr_model_files:
            self.pipeline_status['last_error'] = "롤백할 모델 파일을 찾을 수 없습니다."
            return False
        
        # 가장 최근 파일 선택
        curr_model_path = os.path.join(ensemble_dir, sorted(curr_model_files)[-1])
        prev_model_path = os.path.join(ensemble_dir, sorted(prev_model_files)[-1])
        
        if not os.path.exists(prev_model_path) or not os.path.exists(curr_model_path):
            self.pipeline_status['last_error'] = "롤백할 모델 파일이 존재하지 않습니다."
            return False
            
        try:
            # 롤백 디렉토리에 현재 모델 백업
            rollback_dir = os.path.join(ensemble_dir, "rollback")
            os.makedirs(rollback_dir, exist_ok=True)
            
            rollback_path = os.path.join(
                rollback_dir, 
                f"{user_id}_{pattern}_{curr_date_pattern}_v{version}.model.rolledback"
            )
            
            import shutil
            # 현재 모델 백업
            shutil.copy2(curr_model_path, rollback_path)
            
            # 이전 모델을 현재 위치로 복사
            shutil.copy2(prev_model_path, curr_model_path)
            
            print(f"모델 롤백 완료: v{version} → v{version-1}")
            
            # 메타데이터도 롤백
            curr_meta_files = [
                f for f in os.listdir(versions_dir)
                if f.startswith(f"{user_id}_{pattern}_") and f.endswith(f"_v{version}.meta")
            ]
            
            prev_meta_files = [
                f for f in os.listdir(versions_dir)
                if f.startswith(f"{user_id}_{pattern}_") and f.endswith(f"_v{version-1}.meta")
            ]
            
            if curr_meta_files and prev_meta_files:
                curr_meta_path = os.path.join(versions_dir, sorted(curr_meta_files)[-1])
                prev_meta_path = os.path.join(versions_dir, sorted(prev_meta_files)[-1])
                
                if os.path.exists(curr_meta_path) and os.path.exists(prev_meta_path):
                    rollback_meta_path = os.path.join(
                        rollback_dir, 
                        f"{user_id}_{pattern}_{curr_date_pattern}_v{version}.meta.rolledback"
                    )
                    
                    shutil.copy2(curr_meta_path, rollback_meta_path)
                    shutil.copy2(prev_meta_path, curr_meta_path)
                    
            # 파이프라인 상태 업데이트
            self.pipeline_status['current_stage'] = 'rollback_complete'
            self.pipeline_status['rollback_history'][f"{user_id}_{pattern}_v{version}"]["success"] = True
                
            return True
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'rollback_failed'
            self.pipeline_status['last_error'] = str(e)
            self.pipeline_status['rollback_history'][f"{user_id}_{pattern}_v{version}"]["success"] = False
            print(f"모델 롤백 중 오류 발생: {str(e)}")
            return False
    
    def get_next_version(self, ensemble_dir: str, pattern: str, user_id: int) -> int:
        """
        다음 모델 버전 번호 결정
        
        Args:
            ensemble_dir: 앙상블 디렉토리
            pattern: 패턴 ID
            user_id: 사용자 ID
            
        Returns:
            int: 다음 버전 번호
        """
        # 버전 파일 경로
        version_file = os.path.join(
            ensemble_dir, "versions", f"{user_id}_{pattern}.version"
        )
        
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    current_version = int(f.read().strip())
                next_version = current_version + 1
            except Exception:
                next_version = 1
        else:
            next_version = 1
            
        # 새 버전 저장
        with open(version_file, 'w') as f:
            f.write(str(next_version))
            
        return next_version
    
    def create_personal_model(self, user_id: int, base_dir: str = None, file_path: str = None) -> Dict[str, Any]:
        """
        개인화 모델 생성 메인 함수
        
        Args:
            user_id: 사용자 ID
            base_dir: 기본 디렉토리 경로 (None이면 기본값 사용)
            file_path: 데이터 파일 경로 (None이면 기본값 사용)
            
        Returns:
            dict: 모델 생성 결과
        """
        # 파이프라인 상태 초기화
        self.pipeline_status = {
            'current_stage': 'create_personal_model_started',
            'last_error': None,
            'performance_history': {},
            'rollback_history': {}
        }
        
        # 기본값 설정
        if base_dir is None:
            base_dir = "/content/drive/My Drive/Colab Notebooks/VoID_WaterPurifier"
            
        if file_path is None:
            file_path = 'all_features_merged.csv'
            
        # 작업 디렉토리 변경
        original_dir = os.getcwd()
        os.chdir(base_dir)
        
        try:
            # 디렉토리 설정
            dirs = self.setup_directories(base_dir)
            model_dir_2 = dirs['model_dir_2']
            ensemble_dir = dirs['ensemble_dir']
            
            # 데이터 로드 및 전처리
            full_data = self.preprocess_data(file_path)
            
            # 패턴 정보 추출
            patterns = full_data['pattern'].unique()
            
            # 베이스 모델 로드
            base_models = self.load_base_models(model_dir_2)
            
            # 사용자별 패턴 확인
            pattern_users = {}
            for pattern in patterns:
                users = full_data[full_data['pattern'] == pattern]['person_id'].unique()
                pattern_users[pattern] = users
                
            # 대상 사용자가 속한 패턴 확인
            selected_users = {}
            for pattern in patterns:
                if user_id in pattern_users[pattern]:
                    selected_users[pattern] = [user_id]
                else:
                    selected_users[pattern] = []
                    
            # 패턴별 결과 저장
            results = {}
            summary_data = []
            
            # 패턴별 처리
            for pattern in patterns:
                self.pipeline_status['current_stage'] = f'processing_pattern_{pattern}'
                
                if pattern not in base_models:
                    print(f"패턴 {pattern}에 대한 베이스 모델이 없습니다. 건너뜁니다.")
                    continue
                    
                base_model = base_models[pattern]
                pattern_results = {}
                
                # 사용자별 모델 생성
                for user_id in selected_users[pattern]:
                    # 사용자 데이터 추출
                    user_data = full_data[
                        (full_data['person_id'] == user_id) &
                        (full_data['pattern'] == pattern)
                    ].copy()
                    
                    # 충분한 데이터가 있는지 확인
                    if len(user_data) < 10:
                        print(f"사용자 {user_id}, 패턴 {pattern}에 대한 데이터가 부족합니다 (n={len(user_data)}). 건너뜁니다.")
                        continue
                        
                    # 날짜 기준 데이터 분할
                    train_user, valid_user, holdout_user = self.split_data(user_data)
                    
                    # 학습, 검증, 홀드아웃 세트 준비
                    X_train = train_user[self.features]
                    y_train = train_user['amount']
                    X_valid = valid_user[self.features]
                    y_valid = valid_user['amount']
                    X_holdout = holdout_user[self.features]
                    y_holdout = holdout_user['amount']
                    
                    # 개인화 모델 학습
                    for attempt in range(self.max_retries):
                        self.pipeline_status['current_stage'] = f'training_personal_model_{pattern}_{user_id}_attempt_{attempt+1}'
                        
                        # 현재 파라미터로 모델 학습
                        params = self.DEFAULT_PARAMS.copy()
                        personal_model = self.train_user_model(X_train, y_train, params)
                        
                        # 검증 세트에서 모델 평가
                        valid_pred = personal_model.predict(X_valid)
                        v_r2 = r2_score(y_valid, valid_pred)
                        v_rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
                        err_pct = np.abs((y_valid - valid_pred) / y_valid) * 100
                        v_acc = np.mean(err_pct < 10) * 100
                        
                        # DQM 기준: 성능이 기준을 충족하는지 확인
                        if v_r2 >= self.R2_THRESHOLD and v_acc >= self.ACCURACY_THRESHOLD:
                            print(f"개인화 모델 성능 충족: R2={v_r2:.4f}, 정확도={v_acc:.2f}%")
                            break
                        elif attempt < self.max_retries - 1:
                            # 파라미터 자동 조정
                            params = self.adapt_hyperparameters(v_r2, v_acc, params)
                            print(f"개인화 모델 성능 미달: R2={v_r2:.4f}, 정확도={v_acc:.2f}%. 파라미터 조정 (시도 {attempt+1}/{self.max_retries})")
                        else:
                            print(f"최대 재시도 횟수 초과. 최종 성능: R2={v_r2:.4f}, 정확도={v_acc:.2f}%")
                            
                    # 다음 버전 번호 가져오기
                    next_version = self.get_next_version(ensemble_dir, pattern, user_id)
                    
                    # 앙상블 모델 평가
                    valid_metrics, holdout_metrics, best_alpha = self.evaluate_ensemble(
                        base_model, personal_model,
                        X_valid, y_valid,
                        X_holdout, y_holdout,
                        dataset_name=f"user_{user_id}_pattern_{pattern}",
                        do_dynamic_weight=True
                    )
                    
                    # 성능 기록
                    self.pipeline_status['performance_history'][f"{user_id}_{pattern}_v{next_version}"] = {
                        "valid": valid_metrics,
                        "holdout": holdout_metrics,
                        "best_alpha": best_alpha
                    }
                    
                    # DQM 문서 기준: 앙상블 개선 여부 확인
                    if valid_metrics:
                        use_ensemble = self.check_performance_improvement(
                            valid_metrics['personal'], valid_metrics['ensemble'])
                        
                        if use_ensemble:
                            print(f"앙상블 모델이 개인 모델보다 성능이 좋습니다 (알파: {best_alpha:.2f})")
                        else:
                            best_alpha = 0  # 개인 모델만 사용 (alpha=0)
                            print("개인 모델만 사용하는 것이 더 좋습니다. 앙상블을 사용하지 않습니다.")
                    else:
                        use_ensemble = False
                    
                    # 모델 저장
                    model_path = self.save_ensemble_info(
                        ensemble_dir, pattern, user_id, 
                        personal_model, best_alpha, 
                        {"valid": valid_metrics, "holdout": holdout_metrics},
                        next_version
                    )
                    
                    # DQM 문서 기준: 성능 비교 및 롤백 체크
                    if next_version > 1:
                        need_rollback = self.compare_with_previous_model(
                            ensemble_dir, pattern, user_id,
                            {"valid": valid_metrics, "holdout": holdout_metrics},
                            next_version
                        )
                        
                        if need_rollback:
                            rollback_success = self.rollback_model(
                                ensemble_dir, pattern, user_id, next_version
                            )
                            if rollback_success:
                                print(f"모델 롤백 완료: 패턴 {pattern}, 사용자 {user_id}")
                                
                                # 버전 정보 롤백
                                with open(os.path.join(ensemble_dir, "versions", f"{user_id}_{pattern}.version"), 'w') as f:
                                    f.write(str(next_version - 1))
                    
                    # 결과 저장
                    pattern_results[user_id] = {
                        'best_alpha': best_alpha,
                        'valid_metrics': valid_metrics,
                        'holdout_metrics': holdout_metrics,
                        'use_ensemble': use_ensemble,
                        'model_path': model_path,
                        'version': next_version
                    }
                    
                    # 요약 데이터 추가
                    row = {
                        'Pattern': pattern,
                        'User_ID': user_id,
                        'Alpha': best_alpha,
                        'Version': next_version,
                        'Use_Ensemble': use_ensemble
                    }
                    
                    if valid_metrics:
                        row.update({
                            'Valid_Base_R2': valid_metrics['base']['r2'],
                            'Valid_Personal_R2': valid_metrics['personal']['r2'],
                            'Valid_Ensemble_R2': valid_metrics['ensemble']['r2'],
                            'Valid_Base_RMSE': valid_metrics['base']['rmse'],
                            'Valid_Personal_RMSE': valid_metrics['personal']['rmse'],
                            'Valid_Ensemble_RMSE': valid_metrics['ensemble']['rmse'],
                            'Valid_Base_Acc': valid_metrics['base']['accuracy'],
                            'Valid_Personal_Acc': valid_metrics['personal']['accuracy'],
                            'Valid_Ensemble_Acc': valid_metrics['ensemble']['accuracy'],
                        })
                        
                    if holdout_metrics:
                        row.update({
                            'Holdout_Base_R2': holdout_metrics['base']['r2'],
                            'Holdout_Personal_R2': holdout_metrics['personal']['r2'],
                            'Holdout_Ensemble_R2': holdout_metrics['ensemble']['r2'],
                            'Holdout_Base_RMSE': holdout_metrics['base']['rmse'],
                            'Holdout_Personal_RMSE': holdout_metrics['personal']['rmse'],
                            'Holdout_Ensemble_RMSE': holdout_metrics['ensemble']['rmse'],
                            'Holdout_Base_Acc': holdout_metrics['base']['accuracy'],
                            'Holdout_Personal_Acc': holdout_metrics['personal']['accuracy'],
                            'Holdout_Ensemble_Acc': holdout_metrics['ensemble']['accuracy'],
                        })
                        
                    summary_data.append(row)
                
                # 패턴별 결과 저장
                results[pattern] = pattern_results
                
            # 요약 데이터 저장
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(ensemble_dir, f'ensemble_summary_user_{user_id}.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"요약 정보 저장 완료: {summary_path}")
                
                # 패턴별 평균 결과
                if len(summary_df) > 0:
                    psum = summary_df.groupby('Pattern').mean(numeric_only=True).reset_index()
                    for _, row in psum.iterrows():
                        pat = row['Pattern']
                        print(f"\n패턴 {pat} 평균 성능:")
                        if 'Holdout_Ensemble_R2' in row:
                            print(f"  홀드아웃 R2: {row['Holdout_Ensemble_R2']:.4f}, 정확도: {row['Holdout_Ensemble_Acc']:.2f}%")
                        if 'Valid_Ensemble_R2' in row:
                            print(f"  검증 R2: {row['Valid_Ensemble_R2']:.4f}, 정확도: {row['Valid_Ensemble_Acc']:.2f}%")
            
            # 파이프라인 상태 업데이트
            self.pipeline_status['current_stage'] = 'create_personal_model_completed'
                
            return {
                "status": "success",
                "message": f"사용자 {user_id}에 대한 개인화 모델 생성 완료",
                "results": results,
                "summary_path": summary_path if summary_data else None,
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'create_personal_model_error'
            self.pipeline_status['last_error'] = str(e)
            
            return {
                "status": "error",
                "message": f"개인화 모델 생성 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
            
        finally:
            # 원래 작업 디렉토리로 복원
            os.chdir(original_dir)
            
    def get_pipeline_status(self):
        """파이프라인 상태 반환"""
        return self.pipeline_status


# 이전 버전과의 호환성 유지를 위한 함수
def createPersonalModel(userId):
    """
    개인화 모델 생성 함수 (이전 버전 호환용)
    """
    creator = PersonalModelCreator()
    return creator.create_personal_model(userId)