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
import json
from app.config import get_paths_dict

class FeatureEvaluator:
    def __init__(self):
        self.base_dir = None
        self.results_dir = None
        self.results = {}
        self.paths = get_paths_dict()
        
        # DQM 기준에 맞게 임계값 설정
        self.vif_threshold = 15.0  # VIF 값 임계값
        self.importance_threshold = 0.005  # 중요도 임계값 (0.5%)
        self.r2_threshold = 0.5  # DQM 문서 기준: R2 Score > 0.5
        self.rmse_threshold = 30.0  # DQM 문서 기준: RMSE < 30
        
    def setup_directories(self):
        """경로 설정 및 필요한 디렉토리 생성"""
        self.base_dir = self.paths.get('data', '')
        self.results_dir = os.path.join(self.base_dir, "resources", "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def evaluate_feature(self, patterns=None):
        """
        특성 평가 메인 함수
        
        Args:
            patterns: 분석할 패턴 목록 (기본값: None, 파일에서 자동 감지)
            
        Returns:
            dict: 분석 결과
        """
        self.setup_directories()
        
        # 특성 파일 경로 설정 - 패턴별 파일 찾기
        processed_dir = os.path.join(self.base_dir, "resources", "processed")
        pattern_files = [f for f in os.listdir(processed_dir) 
                       if f.startswith('pattern_') and f.endswith('_features.csv')]
        
        if not pattern_files:
            return {"status": "error", "error": "패턴 파일이 존재하지 않습니다."}
        
        # 패턴을 자동으로 감지
        detected_patterns = [f.split('_')[1] for f in pattern_files]
        
        # patterns 매개변수가 지정되면 해당 패턴만 처리
        if patterns:
            patterns_to_process = [p for p in patterns if p in detected_patterns]
            if not patterns_to_process:
                print(f"경고: 지정된 패턴이 파일에 존재하지 않습니다.")
                return {"status": "error", "error": "패턴 파일을 찾을 수 없습니다."}
        else:
            patterns_to_process = detected_patterns
            
        # 모든 패턴 파일 로드 및 병합
        all_dfs = []
        
        for pattern in patterns_to_process:
            pattern_file = f"pattern_{pattern}_features.csv"
            pattern_path = os.path.join(processed_dir, pattern_file)
            
            if not os.path.exists(pattern_path):
                print(f"패턴 {pattern} 파일이 존재하지 않습니다.")
                continue
                
            # 패턴 파일 읽기
            try:
                df = pd.read_csv(pattern_path)
                df = df[df['amount'] > 0]  # amount가 0보다 큰 행만 유지
                all_dfs.append(df)
            except Exception as e:
                print(f"패턴 {pattern} 파일 로드 오류: {str(e)}")
                
        if not all_dfs:
            return {"status": "error", "error": "유효한 패턴 데이터가 없습니다."}
            
        # 모든 데이터프레임 병합
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # 시간 특성 변환
        merged_df['time_in_minutes'] = merged_df['hour'] * 60 + merged_df['minute']
        merged_df['time_sin'] = np.sin(2 * np.pi * merged_df['time_in_minutes'] / 1440)
        merged_df['time_cos'] = np.cos(2 * np.pi * merged_df['time_in_minutes'] / 1440)
        
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
        
        # 결과 저장용 딕셔너리 초기화
        self.results = {'all': {}}
        
        # GPU 데이터프레임으로 변환
        df_cudf = cudf.DataFrame.from_pandas(merged_df)
        
        # 전체 데이터에 대한 VIF 분석
        self._analyze_vif(df_cudf, features, 'all')
        
        # 전체 데이터에 대한 중요도 분석
        self._analyze_importance(df_cudf, features, 'all')
        
        # 패턴별 분석
        for pattern in patterns_to_process:
            self.results[pattern] = {}
            
            # 패턴별 데이터 필터링
            pattern_df = merged_df[merged_df['pattern'] == pattern]
            
            if len(pattern_df) == 0:
                print(f"경고: 패턴 '{pattern}'의 데이터가 없습니다.")
                continue
                
            # GPU 데이터프레임으로 변환
            try:
                pattern_df_cudf = cudf.DataFrame.from_pandas(pattern_df)
                
                # 패턴별 VIF 분석
                self._analyze_vif(pattern_df_cudf, features, pattern)
                
                # 패턴별 중요도 분석
                self._analyze_importance(pattern_df_cudf, features, pattern)
                
                print(f"패턴 {pattern} 평가 완료")
            except Exception as e:
                print(f"패턴 {pattern} 평가 오류: {str(e)}")
                self.results[pattern]['error'] = str(e)
        
        # 결과 저장
        self._save_results()
        
        return self.results
    
    def _analyze_vif(self, df_cudf, features, pattern_key):
        try:
            X_float64 = cp.array(df_cudf[features].values, dtype=cp.float64)
            X_centered = X_float64 - X_float64.mean(axis=0)
            Cov = (X_centered.T @ X_centered) / (X_float64.shape[0] - 1)
            std = cp.sqrt(cp.diag(Cov))
            corr_matrix = Cov / cp.outer(std, std)
            
            inv_corr = cp.linalg.inv(corr_matrix)
            vifs = cp.diag(inv_corr)
            vif_df = pd.DataFrame({
                'Feature': features,
                'VIF': cp.asnumpy(vifs)
            }).sort_values('VIF', ascending=False)
            
            self.results[pattern_key]['vif'] = vif_df
            
            # VIF 값이 임계치를 초과하는 특성 식별
            high_vif_features = vif_df[vif_df['VIF'] > self.vif_threshold]['Feature'].tolist()
            
            # 임계치를 초과하는 특성이 있으면 PCA 적용 필요성 기록
            if high_vif_features:
                self.results[pattern_key]['high_vif_features'] = high_vif_features
                self.results[pattern_key]['need_pca'] = True
                print(f"패턴 {pattern_key}에서 {len(high_vif_features)}개 특성이 VIF 임계값({self.vif_threshold})을 초과했습니다. PCA 적용이 권장됩니다.")
            else:
                self.results[pattern_key]['need_pca'] = False
            
        except Exception as e:
            print(f"VIF 분석 오류 ({pattern_key}): {str(e)}")
            self.results[pattern_key]['vif'] = pd.DataFrame({'Feature': features, 'VIF': [0] * len(features)})
    
    def _analyze_importance(self, df_cudf, features, pattern_key):
        try:
            X = df_cudf[features]
            y = df_cudf['amount']
            
            # XGBoost 모델 학습
            xgb_dtrain = xgb.DMatrix(X.values, label=y.values, feature_names=features)
            params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'random_state': 42
            }
            xgb_model = xgb.train(params, xgb_dtrain, num_boost_round=100)
            
            # 특성 중요도 추출
            xgb_gain = xgb_model.get_score(importance_type='gain')
            
            # 총 중요도 계산
            total_importance = sum(xgb_gain.values())
            
            # 중요도 비율 계산
            xgb_importance = pd.DataFrame({
                'feature': list(xgb_gain.keys()),
                'importance': list(xgb_gain.values()),
                'importance_ratio': [val/total_importance for val in xgb_gain.values()]
            }).sort_values('importance', ascending=False)
            
            self.results[pattern_key]['importance'] = xgb_importance
            
            # 중요도가 임계값 미만인 특성 식별
            low_importance_features = xgb_importance[xgb_importance['importance_ratio'] < self.importance_threshold]['feature'].tolist()
            
            if low_importance_features:
                self.results[pattern_key]['low_importance_features'] = low_importance_features
                print(f"패턴 {pattern_key}에서 {len(low_importance_features)}개 특성이 중요도 임계값({self.importance_threshold*100}%)보다 낮습니다. 제거 권장:")
                for feat in low_importance_features:
                    print(f"  - {feat}: {xgb_importance[xgb_importance['feature'] == feat]['importance_ratio'].values[0]*100:.4f}%")
            
            # 권장 특성 목록: 중요도가 높은 특성만 유지
            recommended_features = xgb_importance[xgb_importance['importance_ratio'] >= self.importance_threshold]['feature'].tolist()
            self.results[pattern_key]['recommended_features'] = recommended_features
            
        except Exception as e:
            print(f"특성 중요도 분석 오류 ({pattern_key}): {str(e)}")
            self.results[pattern_key]['importance'] = pd.DataFrame({'feature': features, 'importance': [0] * len(features)})
    
    # PCA 적용 함수 추가
    def apply_pca(self, df, high_vif_features, variance_explained=0.95):
        """
        VIF가 높은 특성들에 PCA를 적용하여 다중공선성 감소
        
        Args:
            df: 데이터프레임
            high_vif_features: VIF가 높은 특성 목록
            variance_explained: 유지할 분산 비율 (기본값: 95%)
            
        Returns:
            DataFrame: PCA가 적용된 데이터프레임
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # VIF가 높은 특성만 추출
        X_high_vif = df[high_vif_features].values
        
        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_high_vif)
        
        # PCA 적용
        pca = PCA(n_components=variance_explained, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)
        
        # 주성분 이름 생성
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        # PCA 결과를 데이터프레임으로 변환
        pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        # 원본 데이터프레임에서 VIF가 높은 특성 제거
        df_filtered = df.drop(columns=high_vif_features)
        
        # PCA 결과와 병합
        df_with_pca = pd.concat([df_filtered, pca_df], axis=1)
        
        print(f"PCA 적용 완료: {len(high_vif_features)}개 특성 → {len(pca_columns)}개 주성분 (분산 유지: {variance_explained*100}%)")
        print(f"누적 설명 분산 비율: {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%")
        
        return df_with_pca, pca, scaler
    
    def _save_results(self):
        try:
            # 패턴별 디렉토리 생성 및 결과 저장
            for pattern_key in self.results:
                pattern_dir = os.path.join(self.results_dir, pattern_key)
                os.makedirs(pattern_dir, exist_ok=True)
                
                # VIF 저장
                if 'vif' in self.results[pattern_key]:
                    self.results[pattern_key]['vif'].to_csv(os.path.join(pattern_dir, 'vif.csv'), index=False)
                
                # 중요도 저장
                if 'importance' in self.results[pattern_key]:
                    self.results[pattern_key]['importance'].to_csv(os.path.join(pattern_dir, 'importance.csv'), index=False)
                
                # 권장 특성 목록 저장
                if 'recommended_features' in self.results[pattern_key]:
                    with open(os.path.join(pattern_dir, 'recommended_features.json'), 'w') as f:
                        json.dump(self.results[pattern_key]['recommended_features'], f, indent=2)
            
            # 요약 정보 저장
            summary = {}
            for pattern_key in self.results:
                pattern_summary = {}
                
                # 상위 10개 중요 특성
                if 'importance' in self.results[pattern_key] and not self.results[pattern_key]['importance'].empty:
                    top_features = self.results[pattern_key]['importance'].head(10)['feature'].tolist()
                    pattern_summary['top_features'] = top_features
                    
                    # VIF 관련 정보
                    if 'high_vif_features' in self.results[pattern_key]:
                        pattern_summary['high_vif_features'] = self.results[pattern_key]['high_vif_features']
                        pattern_summary['need_pca'] = self.results[pattern_key]['need_pca']
                    
                    # 중요도 관련 정보
                    if 'low_importance_features' in self.results[pattern_key]:
                        pattern_summary['low_importance_features'] = self.results[pattern_key]['low_importance_features']
                    
                    # 권장 특성 목록
                    if 'recommended_features' in self.results[pattern_key]:
                        pattern_summary['recommended_features'] = self.results[pattern_key]['recommended_features']
                    
                    # 해당 특성의 VIF 값
                    if 'vif' in self.results[pattern_key]:
                        vif_values = {}
                        for feature in top_features:
                            vif_row = self.results[pattern_key]['vif'][self.results[pattern_key]['vif']['Feature'] == feature]
                            if not vif_row.empty:
                                vif_values[feature] = float(vif_row['VIF'].values[0])
                        pattern_summary['vif_values'] = vif_values
                
                summary[pattern_key] = pattern_summary
            
            # JSON으로 저장
            with open(os.path.join(self.results_dir, 'feature_evaluation_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"결과 저장 오류: {str(e)}")
            return False
    
    def get_top_features(self, pattern_key='all', n=10):
        if pattern_key in self.results and 'importance' in self.results[pattern_key]:
            return self.results[pattern_key]['importance'].head(n)['feature'].tolist()
        return []
    
    def get_feature_vif(self, pattern_key='all'):
        if pattern_key in self.results and 'vif' in self.results[pattern_key]:
            return dict(zip(self.results[pattern_key]['vif']['Feature'], self.results[pattern_key]['vif']['VIF']))
        return {}
    
    def get_feature_importance(self, pattern_key='all'):
        if pattern_key in self.results and 'importance' in self.results[pattern_key]:
            return dict(zip(self.results[pattern_key]['importance']['feature'], self.results[pattern_key]['importance']['importance']))
        return {}

# 이전 버전과의 호환성을 위한 함수
def evaluate_feature(patterns=None):
    evaluator = FeatureEvaluator()
    return evaluator.evaluate_feature(patterns)