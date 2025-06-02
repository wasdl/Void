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
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
import json
from app.config import get_paths_dict
from app.utils.helpers import ensure_dir, save_model_results, load_model_results, calculate_model_accuracy
import logging

# 로깅 설정
logger = logging.getLogger('create_base_model')

class BaseModelCreator:
    """
    베이스 모델 생성 및 관리 클래스
    """
    def __init__(self):
        """
        베이스 모델 생성 클래스 초기화
        """
        # 기본 파라미터 설정
        self.default_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'random_state': 42
        }
        
        # 성능 검증 임계값 (DQM 문서 기준으로 설정)
        self.r2_threshold = 0.6  # DQM 기준: R2 Score > 0.6
        self.accuracy_threshold = 60.0  # DQM 기준: 정확도 ≥ 60%
        self.rmse_threshold = 30.0  # DQM 기준: RMSE < 30
        
        # 모델 파라미터 조정 범위
        self.param_ranges = {
            'max_depth': [4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150, 200]
        }
        
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
        
        # 재시도 관련 설정
        self.max_retries = 3
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'needs_autoencoder_adjustment': False,
            'needs_dbscan_adjustment': False,
            'performance_history': {}
        }
        
        # 경로 정보
        self.paths = get_paths_dict()
    
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
        # 모델 저장 디렉토리 설정
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_dir_1 = os.path.join(model_dir, "1")
        model_dir_2 = os.path.join(model_dir, "2")
        os.makedirs(model_dir_1, exist_ok=True)
        os.makedirs(model_dir_2, exist_ok=True)
        
        # 버전 정보 디렉토리
        version_dir = os.path.join(model_dir, "versions")
        os.makedirs(version_dir, exist_ok=True)
        
        return {
            "model_dir": model_dir,
            "model_dir_1": model_dir_1,
            "model_dir_2": model_dir_2,
            "version_dir": version_dir
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
        df = pd.read_csv(file_path)

        # amount가 0인 데이터 필터링
        df = df[df['amount'] > 0]  # amount가 0보다 큰 행만 유지

        # 시간 특성 통합 및 sin-cos 변환
        df['time_in_minutes'] = df['hour'] * 60 + df['minute']  # 0-1439 분 범위
        df['time_sin'] = np.sin(2 * np.pi * df['time_in_minutes'] / 1440)  # 1440 = 24시간 * 60분
        df['time_cos'] = np.cos(2 * np.pi * df['time_in_minutes'] / 1440)
        
        return df
    
    def train_all_model(self, df: pd.DataFrame, model_dir: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        전체 패턴에 대한 모델 학습
        
        Args:
            df: 전처리된 데이터프레임
            model_dir: 모델 저장 디렉토리
            params: XGBoost 파라미터 (기본값 사용 시 None)
            
        Returns:
            dict: 학습 결과
        """
        self.pipeline_status['current_stage'] = 'training_all_model'
        logger.info("ALL 모델 학습 시작")
        
        if params is None:
            params = self.default_params.copy()
            
        # 패턴 필터링 (없으면 모든 패턴 사용)
        patterns = df['pattern'].unique()
        df_2 = df.copy()
        df_cudf_2 = cudf.DataFrame.from_pandas(df_2)
        
        # 결과 저장 딕셔너리
        results = {'all': {}}
        for pattern in patterns:
            results[pattern] = {}
            
        # 전체 데이터 모델
        X_all = df_cudf_2[self.features]
        y_all = df_cudf_2['amount']

        # XGBoost 모델 학습
        logger.info("XGBoost ALL 모델 훈련 중...")
        xgb_dtrain_all = xgb.DMatrix(X_all.values, label=y_all.values, feature_names=self.features)
        xgb_model_all = xgb.train(params, xgb_dtrain_all, num_boost_round=100)

        # 모델 저장
        model_path_all = os.path.join(model_dir, 'all.model')
        xgb_model_all.save_model(model_path_all)
        logger.info(f"ALL 모델 저장됨: {model_path_all}")
        
        # 성능 평가
        eval_result = self._evaluate_model(df_2, model_path_all, 'all')
        results['all'] = eval_result
        
        # DQM 기준: ALL 모델의 R2 Score가 임계값 미만인 경우 오토인코더 재조정 필요
        if eval_result['r2'] < self.r2_threshold:
            self.pipeline_status['current_stage'] = 'all_model_r2_low'
            self.pipeline_status['last_error'] = f"ALL 모델 R2 점수({eval_result['r2']:.4f})가 임계값({self.r2_threshold})보다 낮습니다."
            self.pipeline_status['needs_autoencoder_adjustment'] = True
            
            logger.warning(f"ALL 모델 R2 점수({eval_result['r2']:.4f})가 임계값({self.r2_threshold})보다 낮습니다.")
            logger.warning("DQM 기준에 따라 오토인코더 임계값 재조정이 필요합니다.")
            
            # 성능 개선 시도
            improved_params = self._optimize_model_params(df_2, 'all')
            if improved_params:
                logger.info("최적화된 파라미터로 모델 재학습을 시도합니다.")
                return self.train_all_model(df, model_dir, improved_params)
            else:
                return {
                    "status": "error",
                    "error": "ALL 모델 성능이 낮고 파라미터 최적화도 실패했습니다. 오토인코더 재조정이 필요합니다.",
                    "results": results,
                    "pipeline_status": self.pipeline_status
                }
        
        # 성능이 충족되면 성공 반환
        self.pipeline_status['current_stage'] = 'all_model_success'
        self.pipeline_status['performance_history']['all'] = eval_result
        logger.info(f"ALL 모델 학습 성공: R2={eval_result['r2']:.4f}, 정확도={eval_result['accuracy']:.2f}%")
        
        return {
            "status": "success",
            "message": "ALL 모델 학습 완료",
            "results": results,
            "pipeline_status": self.pipeline_status
        }

    def train_pattern_models(self, df: pd.DataFrame, model_dir: str, params: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        패턴별 모델 학습
        
        Args:
            df: 전처리된 데이터프레임
            model_dir: 모델 저장 디렉토리
            params: XGBoost 파라미터 (기본값 사용 시 None)
            
        Returns:
            dict: 패턴별 학습 결과
        """
        self.pipeline_status['current_stage'] = 'training_pattern_models'
        logger.info("패턴별 모델 학습 시작")
        
        if params is None:
            params = self.default_params.copy()
            
        # 패턴 필터링 (없으면 모든 패턴 사용)
        patterns = df['pattern'].unique()
        df_2 = df.copy()
        df_cudf_2 = cudf.DataFrame.from_pandas(df_2)
        
        # 결과 저장 딕셔너리
        results = {}
        failed_patterns = []
        
        # 패턴별 모델 학습
        for pattern in patterns:
            logger.info(f"패턴 {pattern} 모델 학습 시작")
            # 모델 파일 경로 확인
            model_path_pattern = os.path.join(model_dir, f'{pattern}.model')
            
            # 이미 모델 파일이 존재하는지 확인
            if os.path.exists(model_path_pattern):
                logger.info(f"모델 파일 {pattern}.model이 이미 존재합니다. 성능 검증을 진행합니다.")
                
                # 기존 모델의 성능 평가
                eval_result = self._evaluate_model(df_2[df_2['pattern'] == pattern], model_path_pattern, pattern)
                
                # DQM 기준에 따라 성능이 기준에 미달하는 경우
                if eval_result['r2'] < self.r2_threshold or eval_result['accuracy'] < self.accuracy_threshold:
                    logger.warning(f"패턴 {pattern} 모델의 성능이 기준에 미달합니다. 재학습을 진행합니다. (R2: {eval_result['r2']:.4f}, 정확도: {eval_result['accuracy']:.2f}%)")
                    # 백업 생성
                    self._backup_model(model_path_pattern)
                else:
                    results[pattern] = eval_result
                    self.pipeline_status['performance_history'][pattern] = eval_result
                    logger.info(f"패턴 {pattern} 기존 모델 성능 양호: R2={eval_result['r2']:.4f}, 정확도={eval_result['accuracy']:.2f}%")
                    continue
            
            # 패턴별 데이터 필터링
            pattern_df = df_2[df_2['pattern'] == pattern]
            pattern_df_cudf = df_cudf_2[df_cudf_2['pattern'] == pattern]
            
            # 데이터가 충분한지 확인
            if len(pattern_df) < 100:
                logger.warning(f"패턴 {pattern}의 데이터가 충분하지 않습니다 (데이터 수: {len(pattern_df)}).")
                results[pattern] = {"status": "error", "message": f"데이터 부족 (n={len(pattern_df)})"}
                continue
                
            # 모델 학습
            for attempt in range(self.max_retries):
                # 현재 파라미터로 모델 학습
                X_pattern = pattern_df_cudf[self.features]
                y_pattern = pattern_df_cudf['amount']

                try:
                    logger.info(f"패턴 {pattern} 모델 훈련 중... (시도 {attempt+1}/{self.max_retries})")
                    xgb_dtrain_pattern = xgb.DMatrix(X_pattern.values, label=y_pattern.values, feature_names=self.features)
                    xgb_model_pattern = xgb.train(params, xgb_dtrain_pattern, num_boost_round=100)

                    # 모델 저장
                    xgb_model_pattern.save_model(model_path_pattern)
                    logger.info(f"패턴 {pattern} 모델 저장됨: {model_path_pattern}")
                    
                    # 성능 평가
                    eval_result = self._evaluate_model(pattern_df, model_path_pattern, pattern)
                    
                    # DQM 기준 검증
                    if eval_result['r2'] >= self.r2_threshold and eval_result['accuracy'] >= self.accuracy_threshold:
                        # 성능이 기준을 충족하면 완료
                        results[pattern] = eval_result
                        self.pipeline_status['performance_history'][pattern] = eval_result
                        logger.info(f"패턴 {pattern} 모델 학습 완료: R2={eval_result['r2']:.4f}, 정확도={eval_result['accuracy']:.2f}%")
                        break
                    elif attempt < self.max_retries - 1:
                        # 성능이 기준에 미달하고 아직 재시도 기회가 있으면 파라미터 조정
                        params = self._adjust_params(params, eval_result)
                        logger.warning(f"패턴 {pattern} 모델 성능 미달: R2={eval_result['r2']:.4f}, 정확도={eval_result['accuracy']:.2f}%. 파라미터 조정 후 재시도 {attempt+1}/{self.max_retries}")
                    else:
                        # 최대 재시도 횟수 초과, 오토인코더 재조정 필요 표시
                        failed_patterns.append(pattern)
                        results[pattern] = eval_result
                        self.pipeline_status['performance_history'][pattern] = eval_result
                        logger.warning(f"패턴 {pattern} 모델 최대 재시도 횟수 초과. 최종 성능: R2={eval_result['r2']:.4f}, 정확도={eval_result['accuracy']:.2f}%")
                        logger.warning("DQM 기준에 따라 오토인코더 임계값 재조정이 필요합니다.")
                
                except Exception as e:
                    logger.error(f"패턴 {pattern} 모델 학습 중 오류 발생: {str(e)}")
                    if attempt == self.max_retries - 1:
                        failed_patterns.append(pattern)
                        results[pattern] = {"status": "error", "message": f"모델 학습 오류: {str(e)}"}
            
            # 버전 관리 정보 저장
            version_info = {
                'pattern': pattern,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'params': params,
                'performance': eval_result,
                'version': self._get_next_version(pattern)
            }
            self._save_version_info(version_info, model_dir)
        
        # DQM 기준에 따라 오토인코더 재조정 필요 여부 결정
        if failed_patterns:
            self.pipeline_status['current_stage'] = 'pattern_models_partial_failure'
            self.pipeline_status['last_error'] = f"패턴 {', '.join(failed_patterns)} 모델이 성능 기준을 충족하지 못했습니다."
            self.pipeline_status['needs_autoencoder_adjustment'] = True
            
            logger.warning(f"일부 패턴 모델({', '.join(failed_patterns)}) 학습 실패. 오토인코더 임계값 재조정이 필요합니다.")
            
            return {
                "status": "warning",
                "message": f"{len(patterns) - len(failed_patterns)}/{len(patterns)} 패턴 모델 학습 완료. {len(failed_patterns)}개 패턴은 성능 미달.",
                "results": results,
                "failed_patterns": failed_patterns,
                "pipeline_status": self.pipeline_status
            }
        
        # 모든 패턴 모델이 성공적으로 학습된 경우
        self.pipeline_status['current_stage'] = 'pattern_models_success'
        logger.info("모든 패턴 모델 학습 성공")
        
        return {
            "status": "success",
            "message": f"모든 패턴 모델 학습 완료",
            "results": results,
            "pipeline_status": self.pipeline_status
        }
        
    def create_base_model(self, path: str, file: str) -> Dict[str, Any]:
        """
        베이스 모델 생성 및 평가 (메인 메서드)
        
        Args:
            path: 기본 경로
            file: 데이터 파일 경로
            
        Returns:
            dict: 학습 결과 및 성능 평가
        """
        # 파이프라인 상태 초기화
        self.pipeline_status = {
            'current_stage': 'create_base_model_started',
            'last_error': None,
            'needs_autoencoder_adjustment': False,
            'needs_dbscan_adjustment': False,
            'performance_history': {}
        }
        
        logger.info("베이스 모델 생성 시작")
        
        # 경로 설정에 config 사용
        paths = get_paths_dict()
        
        # 디렉토리 설정
        dirs = self.setup_directories(paths.get('data', ''))
        model_dir = dirs['model_dir_2']
        
        # 작업 디렉토리 변경을 피하고 절대 경로 사용
        try:
            # 데이터 전처리
            df = self.preprocess_data(os.path.join(paths.get('data', ''), file))
            
            # 모든 패턴에 대한 ALL 모델 학습
            all_results = self.train_all_model(df, model_dir)
            
            # ALL 모델 학습 실패 시 파이프라인 중단
            if all_results.get("status") == "error":
                self.pipeline_status['current_stage'] = 'create_base_model_failed'
                self.pipeline_status['last_error'] = all_results.get("error", "ALL 모델 학습 실패")
                logger.error(f"ALL 모델 학습 실패: {all_results.get('error', '')}")
                return all_results
            
            # 패턴별 모델 학습
            pattern_results = self.train_pattern_models(df, model_dir)
            
            # 결과 통합
            results = all_results.get("results", {}).copy()
            results.update(pattern_results.get("results", {}))
            
            # 파이프라인 상태 업데이트
            self.pipeline_status['current_stage'] = 'create_base_model_completed'
            if pattern_results.get("status") == "warning":
                self.pipeline_status['last_error'] = pattern_results.get("message", "일부 패턴 모델 학습 실패")
            
            # 결과 저장
            results_path = os.path.join(model_dir, "model_evaluation_results.pkl")
            save_model_results(results, results_path)
                
            logger.info(f"모델 평가 결과 저장: {results_path}")
            
            # GPU 메모리 정리
            cp.get_default_memory_pool().free_all_blocks()
            
            # DQM 기준에 따라 파이프라인 분기 정보 포함 (오토인코더 또는 DBSCAN 재조정 필요 여부)
            return {
                "status": "success" if not self.pipeline_status['needs_autoencoder_adjustment'] else "warning",
                "message": "베이스 모델 학습 완료" + (" (일부 모델 성능 미달)" if self.pipeline_status['needs_autoencoder_adjustment'] else ""),
                "results": results,
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'create_base_model_error'
            self.pipeline_status['last_error'] = str(e)
            logger.error(f"베이스 모델 생성 중 오류 발생: {str(e)}")
            
            return {
                "status": "error",
                "message": f"베이스 모델 학습 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
    
    def check_performance_trend(self, path: str) -> Dict[str, Any]:
        """
        모델 성능 추이 검증 (DQM 문서 기준: 14일 연속 성능 저하 시 파이프라인 재실행)
        
        Args:
            path: 기본 경로
            
        Returns:
            dict: 검증 결과
        """
        try:
            logger.info("모델 성능 추이 검증 시작")
            # config에서 경로 가져오기
            paths = get_paths_dict()
            
            # 모델 디렉토리
            model_dir = os.path.join(paths.get('data', ''), "models")
            metrics_path = os.path.join(model_dir, "performance_metrics.json")
            
            # 이전 성능 지표 로드
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_history = json.load(f)
            else:
                metrics_history = {"patterns": {}, "all": {}, "consecutive_decline_days": 0, "last_updated": ""}
                
            # 현재 날짜와 모델 성능 기록
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 모델 성능 평가 (가장 최근 모델)
            latest_model_dir = os.path.join(model_dir, "2")
            
            # 모델 평가 결과 파일
            results_path = os.path.join(latest_model_dir, "model_evaluation_results.pkl")
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    latest_results = pickle.load(f)
                    
                # 성능 추이 업데이트
                if "all" in latest_results:
                    # ALL 모델 성능
                    all_r2 = latest_results["all"].get("r2", 0)
                    all_accuracy = latest_results["all"].get("accuracy", 0)
                    
                    # 이전 성능과 비교
                    prev_all_r2 = metrics_history["all"].get("r2", [0])[-1] if "r2" in metrics_history["all"] else 0
                    prev_all_accuracy = metrics_history["all"].get("accuracy", [0])[-1] if "accuracy" in metrics_history["all"] else 0
                    
                    # 성능 저하 확인
                    r2_decline = all_r2 < prev_all_r2
                    accuracy_decline = all_accuracy < prev_all_accuracy
                    
                    # 14일 연속 성능 저하 체크
                    if r2_decline and accuracy_decline:
                        metrics_history["consecutive_decline_days"] += 1
                        logger.warning(f"성능 저하 감지 - 연속일수: {metrics_history['consecutive_decline_days']}일")
                    else:
                        metrics_history["consecutive_decline_days"] = 0
                        logger.info("성능 저하 없음, 연속일수 초기화")
                    
                    # 성능 지표 업데이트
                    if "r2" not in metrics_history["all"]:
                        metrics_history["all"]["r2"] = []
                    if "accuracy" not in metrics_history["all"]:
                        metrics_history["all"]["accuracy"] = []
                        
                    metrics_history["all"]["r2"].append(all_r2)
                    metrics_history["all"]["accuracy"].append(all_accuracy)
                    
                    # 최대 지표 수 제한 (최근 30일만 유지)
                    if len(metrics_history["all"]["r2"]) > 30:
                        metrics_history["all"]["r2"] = metrics_history["all"]["r2"][-30:]
                    if len(metrics_history["all"]["accuracy"]) > 30:
                        metrics_history["all"]["accuracy"] = metrics_history["all"]["accuracy"][-30:]
                
                # 패턴별 성능 추이 업데이트
                for pattern in latest_results:
                    if pattern == "all":
                        continue
                        
                    if pattern not in metrics_history["patterns"]:
                        metrics_history["patterns"][pattern] = {"r2": [], "accuracy": []}
                        
                    r2 = latest_results[pattern].get("r2", 0)
                    accuracy = latest_results[pattern].get("accuracy", 0)
                    
                    metrics_history["patterns"][pattern]["r2"].append(r2)
                    metrics_history["patterns"][pattern]["accuracy"].append(accuracy)
                    
                    # 최대 지표 수 제한 (최근 30일만 유지)
                    if len(metrics_history["patterns"][pattern]["r2"]) > 30:
                        metrics_history["patterns"][pattern]["r2"] = metrics_history["patterns"][pattern]["r2"][-30:]
                    if len(metrics_history["patterns"][pattern]["accuracy"]) > 30:
                        metrics_history["patterns"][pattern]["accuracy"] = metrics_history["patterns"][pattern]["accuracy"][-30:]
            
            # 현재 날짜 추가
            metrics_history["last_updated"] = current_date
            
            # 기록 저장
            with open(metrics_path, 'w') as f:
                json.dump(metrics_history, f, indent=2)
                
            # DQM 기준: 14일 연속 성능 저하 시 파이프라인 재실행 필요
            needs_pipeline_restart = metrics_history["consecutive_decline_days"] >= 14
            
            if needs_pipeline_restart:
                logger.warning(f"경고: {metrics_history['consecutive_decline_days']}일 연속 성능 저하가 감지되었습니다. DQM 기준에 따라 전체 파이프라인 재실행이 필요합니다.")
                
            logger.info(f"성능 추이 검증 완료: 재실행 필요={needs_pipeline_restart}")
            
            return {
                "status": "warning" if needs_pipeline_restart else "success",
                "message": "모델 성능 추이 검증 완료" + (" - 재실행 필요" if needs_pipeline_restart else ""),
                "consecutive_decline_days": metrics_history["consecutive_decline_days"],
                "needs_pipeline_restart": needs_pipeline_restart,
                "last_checked": metrics_history["last_updated"]
            }
        except Exception as e:
            logger.error(f"성능 추이 분석 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "message": f"성능 추이 분석 중 오류 발생: {str(e)}"
            }
            
    def get_pipeline_status(self):
        """파이프라인 상태 반환"""
        return self.pipeline_status


# 이전 버전과의 호환성 유지를 위한 함수
def create_base_model(path, file):
    """
    베이스 모델 생성 함수 (이전 버전 호환용)
    """
    creator = BaseModelCreator()
    return creator.create_base_model(path, file)