# https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import pickle
from app.config import get_paths_dict

class EncoderModel(nn.Module):
    """오토인코더 PyTorch 모델"""
    def __init__(self, input_dim=48, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoencoderClassifier:
    """패턴 분류를 위한 오토인코더 기반 분류기"""
    def __init__(self):
        self.models = {}
        self.thresholds = {}
        self.patterns = ['A', 'B', 'C', 'D']
        self.paths = get_paths_dict()
        self.model_path = os.path.join(self.paths.get('models', ''), 'autoencoder')
        os.makedirs(self.model_path, exist_ok=True)
        
        # 임계값 조정 관련 설정 (DQM 문서 기준 적용)
        self.std_multipliers = [1.5, 1.8, 2.0, 2.3, 2.5]
        self.default_std_multiplier = 1.5  # DQM 문서 기준: Mean + 1.5*Std
        self.unknown_ratio_max = 0.05  # DQM 문서 기준: 5%
        self.max_retries = 3
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'threshold_history': {},
            'needs_dbscan_adjustment': False
        }
    
    def train_autoencoder(self, model, data, epochs=150, lr=1e-3):
        """단일 오토인코더 모델 훈련"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        inputs = torch.tensor(data, dtype=torch.float32)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
    
    def _load_pattern_data(self, patterns=None):
        """원본 패턴 데이터 로드 (특성 데이터가 아닌 원본 시계열)"""
        data_path = self.paths.get('data', '')
        
        # 패턴별 파일 찾기
        pattern_files = []
        if patterns:
            # 지정된 패턴만 찾기
            for pattern in patterns:
                file_path = os.path.join(data_path, f"pattern_{pattern}_30days.csv")
                if os.path.exists(file_path):
                    pattern_files.append(file_path)
        else:
            # 모든 패턴 파일 찾기
            all_files = [f for f in os.listdir(data_path) 
                       if f.startswith('pattern_') and f.endswith('_30days.csv')]
            pattern_files = [os.path.join(data_path, f) for f in all_files]
        
        if not pattern_files:
            return None
            
        # 모든 파일 로드 및 병합
        all_dfs = []
        for file_path in pattern_files:
            try:
                df = pd.read_csv(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"파일 로드 오류 {file_path}: {str(e)}")
                
        if not all_dfs:
            return None
            
        return pd.concat(all_dfs, ignore_index=True)
    
    def train_classifiers(self, patterns=None):
        """각 패턴별 오토인코더 학습"""
        # 파이프라인 상태 초기화
        self.pipeline_status = {
            'current_stage': 'training_started',
            'last_error': None,
            'threshold_history': {},
            'needs_dbscan_adjustment': False
        }
        
        # 패턴 데이터 로드
        raw_data = self._load_pattern_data(patterns)
        
        if raw_data is None:
            self.pipeline_status['current_stage'] = 'data_loading_failed'
            self.pipeline_status['last_error'] = "패턴 데이터 파일을 찾을 수 없습니다."
            return {"status": "error", "error": "패턴 데이터 파일을 찾을 수 없습니다."}
        
        # 패턴별 데이터 확인
        patterns_in_data = set(raw_data['pattern'].unique())
        patterns_to_train = [p for p in self.patterns if p in patterns_in_data]
        
        if not patterns_to_train:
            self.pipeline_status['current_stage'] = 'no_valid_patterns'
            self.pipeline_status['last_error'] = "훈련할 패턴 데이터가 없습니다."
            return {"status": "error", "error": "훈련할 패턴 데이터가 없습니다."}
        
        # 데이터 전처리 - 원본 시계열 데이터 사용
        X_5day, y_5day = self._prepare_data(raw_data, patterns_to_train)
        
        if len(X_5day) == 0:
            self.pipeline_status['current_stage'] = 'insufficient_data'
            self.pipeline_status['last_error'] = "처리할 데이터가 충분하지 않습니다."
            return {"status": "error", "error": "처리할 데이터가 충분하지 않습니다."}
        
        # 패턴별 모델 훈련
        for pattern in patterns_to_train:
            print(f"패턴 {pattern} 오토인코더 학습 중...")
            data = X_5day[y_5day == pattern]
            
            if len(data) == 0:
                print(f"패턴 {pattern}에 대한 데이터가 없습니다. 건너뜁니다.")
                continue
                
            model = EncoderModel()
            self.train_autoencoder(model, data)
            self.models[pattern] = model

            # 임계값 설정 (기본값으로 시작)
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(data, dtype=torch.float32)).numpy()
            mses = [mean_squared_error(x, y) for x, y in zip(data, outputs)]
            mse_mean = np.mean(mses)
            mse_std = np.std(mses)
            
            # 기본 임계값 설정 - DQM 문서 기준 (Mean + 1.5*Std)
            self.thresholds[pattern] = mse_mean + self.default_std_multiplier * mse_std
        
        self.pipeline_status['current_stage'] = 'training_completed'
        
        # 임계값 최적화 및 Unknown 비율 검증
        threshold_result = self._optimize_thresholds(X_5day, y_5day)
        
        if threshold_result["status"] == "error":
            print(f"경고: {threshold_result.get('message', '임계값 최적화 실패')}")
            print("기본 임계값을 사용합니다.")
            
            # 다음 단계에서 DBSCAN 재조정 필요 여부 설정
            if threshold_result.get("recommendation") == "DBSCAN 파라미터 재조정 필요":
                self.pipeline_status['needs_dbscan_adjustment'] = True
        
        # 모델 평가
        if len(self.models) > 0:
            self._evaluate_models(X_5day, y_5day, patterns_to_train)
            
        # 모델 저장
        self._save_models()
        
        # 파이프라인 연계를 위한 상태 추가
        return {
            "status": "success" if threshold_result["status"] == "success" else "warning",
            "message": f"{len(self.models)}개 패턴에 대한 오토인코더 학습 완료",
            "patterns_trained": list(self.models.keys()),
            "threshold_result": threshold_result,
            "pipeline_status": self.pipeline_status,
            "needs_dbscan_adjustment": self.pipeline_status['needs_dbscan_adjustment']
        }
    
    def _prepare_data(self, df, patterns, size=1000, day_range=30):
        """데이터 전처리 - 원본 시계열 데이터 사용"""
        X_5day = []
        y_5day = []
        
        # 시작 날짜 설정
        try:
            df['date'] = pd.to_datetime(df['date'])
            start_date = df['date'].min()
        except:
            # config에서 기본 날짜 가져오거나 현재 날짜 사용
            start_date = datetime.now() - timedelta(days=30)
        
        # 패턴별 처리
        for pattern in patterns:
            pattern_df = df[df['pattern'] == pattern]
            
            if len(pattern_df) == 0:
                continue
                
            # 사용자 ID 샘플링
            person_ids = pattern_df['person_id'].unique()
            if len(person_ids) > size:
                person_ids = np.random.choice(person_ids, size, replace=False)
            
            # 각 사용자별 데이터 처리
            for person_id in person_ids:
                person_df = pattern_df[pattern_df['person_id'] == person_id].sort_values(['date', 'hour', 'minute'])
                
                for i in range(0, day_range, 5):  # 0~4, 5~9, ..., 25~29
                    day_range_slice = (i, i + 5)
                    temp_df = person_df[
                        ((person_df['date'] - start_date).dt.days >= day_range_slice[0]) &
                        ((person_df['date'] - start_date).dt.days < day_range_slice[1])
                    ]

                    if len(temp_df) == 48 * 5:  # 5일치 전체 데이터가 있는지 확인
                        # 출수량 시계열로 변환 (시간별 정렬)
                        amount_series = temp_df['amount'].values
                        
                        # 정규화
                        max_amount = max(amount_series) if max(amount_series) > 0 else 1
                        norm_series = amount_series / max_amount
                        
                        # 하루 단위로 압축 (48 슬롯)
                        daily_pattern = []
                        for day in range(5):
                            day_start = day * 48
                            day_end = day_start + 48
                            day_data = norm_series[day_start:day_end]
                            
                            # 모든 날의 패턴을 합쳐서 하나의 패턴으로 만듦
                            if day == 0:
                                daily_pattern = day_data
                            else:
                                daily_pattern = daily_pattern + day_data
                                
                        # 일별 합산한 패턴을 다시 정규화
                        if max(daily_pattern) > 0:
                            daily_pattern = daily_pattern / max(daily_pattern)
                            
                        X_5day.append(daily_pattern)
                        y_5day.append(pattern)
        
        return np.array(X_5day), np.array(y_5day)
    
    def _evaluate_models(self, X_5day, y_5day, patterns):
        """모델 평가"""
        predicted_labels = []
        reconstruction_errors = []

        for sample in X_5day:
            pred, recon, mse = self.predict(sample)
            predicted_labels.append(pred)
            reconstruction_errors.append(mse)

        predicted_labels = np.array(predicted_labels)

        from collections import Counter

        print("\n📊 패턴별 예측 결과:")
        for pattern in patterns:
            idxs = np.where(y_5day == pattern)[0]
            if len(idxs) == 0:
                continue
                
            total = len(idxs)
            pred_counts = Counter(predicted_labels[idxs])
            unknown_count = pred_counts.get('Unknown', 0)
            correct_count = pred_counts.get(pattern, 0)
            acc = correct_count / total
            unknown_ratio = unknown_count / total

            print(f"패턴 {pattern} ▶ 정확도: {acc:.2%}, 미확인: {unknown_ratio:.2%}")

        # Unknown 제외 혼동 행렬
        valid_mask = predicted_labels != 'Unknown'
        if sum(valid_mask) > 0:
            valid_patterns = [p for p in patterns if p in self.models]
            cm = confusion_matrix(y_5day[valid_mask], predicted_labels[valid_mask], labels=valid_patterns)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_patterns)
            
            # 혼동 행렬 저장
            plt.figure(figsize=(8, 6))
            disp.plot(cmap='Blues')
            plt.title("혼동 행렬 (Unknown 제외)")
            plt.grid(False)
            plt.savefig(os.path.join(self.model_path, 'confusion_matrix.png'))
            plt.close()
    
    def _save_models(self):
        """모델 및 임계값 저장"""
        for pattern, model in self.models.items():
            model_path = os.path.join(self.model_path, f'autoencoder_{pattern}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"✅ 패턴 {pattern} 오토인코더 저장 완료: {model_path}")

        threshold_path = os.path.join(self.model_path, 'thresholds.pkl')
        with open(threshold_path, 'wb') as f:
            pickle.dump(self.thresholds, f)
            print(f"✅ 임계값 저장 완료: {threshold_path}")
    
    def load_models(self):
        """저장된 모델 및 임계값 로드"""
        self.models = {}
        self.thresholds = {}
        
        # 임계값 로드
        threshold_path = os.path.join(self.model_path, 'thresholds.pkl')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                self.thresholds = pickle.load(f)
        else:
            return {
                "status": "error",
                "error": "저장된 임계값 파일이 없습니다."
            }
        
        # 모델 로드
        models_loaded = 0
        for pattern in self.patterns:
            model_path = os.path.join(self.model_path, f'autoencoder_{pattern}.pt')
            if os.path.exists(model_path):
                model = EncoderModel()
                model.load_state_dict(torch.load(model_path))
                model.eval()
                self.models[pattern] = model
                models_loaded += 1
        
        if models_loaded == 0:
            return {
                "status": "error",
                "error": "저장된 모델 파일이 없습니다."
            }
            
        return {
            "status": "success", 
            "message": f"{models_loaded}개 패턴 오토인코더 로드 완료",
            "patterns_loaded": list(self.models.keys())
        }
    
    def predict(self, sample):
        """단일 샘플에 대한 패턴 예측 (내부용)"""
        if not self.models:
            return "Unknown", None, 0
            
        errors = {}
        for p, model in self.models.items():
            model.eval()
            with torch.no_grad():
                x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
                recon = model(x).squeeze(0).numpy()
                mse = mean_squared_error(sample, recon)
                errors[p] = (mse, recon)

        best_p = min(errors, key=lambda k: errors[k][0])
        best_mse = errors[best_p][0]

        if best_mse > self.thresholds.get(best_p, float('inf')):
            return "Unknown", errors[best_p][1], best_mse
        else:
            return best_p, errors[best_p][1], best_mse
            
    def predict_pattern(self, data):
        """패턴 예측 서비스 메서드"""
        # 먼저 모델 로드 시도
        if not self.models:
            load_result = self.load_models()
            if load_result["status"] == "error":
                return load_result
            
        # 데이터 전처리
        try:
            # 여기서는 이미 전처리된 데이터가 입력된다고 가정
            # 필요한 경우 추가 전처리 로직 구현
            pred_pattern, _, confidence = self.predict(data)
            
            return {
                "status": "success",
                "pattern": pred_pattern,
                "confidence": 1.0 - min(1.0, confidence)  # 오차를 신뢰도로 변환
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"분류 중 오류 발생: {str(e)}"
            }
    
    def _optimize_thresholds(self, X, y):
        """
        임계값을 최적화하여 Unknown 비율이 목표치 이하가 되도록 조정
        
        Args:
            X: 테스트 데이터
            y: 실제 레이블
            
        Returns:
            dict: 최적화 결과
        """
        if not self.models:
            self.pipeline_status['current_stage'] = 'threshold_optimization_failed'
            self.pipeline_status['last_error'] = "최적화할 모델이 없습니다."
            return {"status": "error", "error": "최적화할 모델이 없습니다."}
            
        # 초기 임계값 저장 (롤백 가능하도록)
        initial_thresholds = self.thresholds.copy()
        
        # 임계값 조정 시도
        for retry in range(self.max_retries):
            # 현재 임계값으로 Unknown 비율 계산
            unknown_count = 0
            
            for i in range(len(X)):
                pred, _, _ = self.predict(X[i])
                if pred == "Unknown":
                    unknown_count += 1
                    
            unknown_ratio = unknown_count / len(X)
            
            print(f"시도 {retry+1}/{self.max_retries}: Unknown 비율 = {unknown_ratio:.2%}, 목표 = {self.unknown_ratio_max:.2%}")
            
            # 현재 임계값 상태 기록
            self.pipeline_status['threshold_history'][f'attempt_{retry+1}'] = {
                'unknown_ratio': unknown_ratio,
                'thresholds': self.thresholds.copy()
            }
            
            # 목표 달성 - Unknown 비율이 목표 이하
            if unknown_ratio <= self.unknown_ratio_max:
                self.pipeline_status['current_stage'] = 'threshold_optimization_success'
                return {
                    "status": "success",
                    "message": f"임계값 최적화 완료 (시도 {retry+1}/{self.max_retries})",
                    "unknown_ratio": unknown_ratio,
                    "thresholds": self.thresholds.copy()
                }
            
            # 임계값 조정 - 다음 멀티플라이어 시도
            if retry < len(self.std_multipliers) - 1:
                next_multiplier = self.std_multipliers[retry + 1]
                
                # 모든 패턴의 임계값 조정
                for pattern in self.models.keys():
                    # 모델 평가
                    model = self.models[pattern]
                    pattern_data = X[y == pattern]
                    
                    if len(pattern_data) > 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(torch.tensor(pattern_data, dtype=torch.float32)).numpy()
                        mses = [mean_squared_error(x, y) for x, y in zip(pattern_data, outputs)]
                        mse_mean = np.mean(mses)
                        mse_std = np.std(mses)
                        
                        # 새 임계값 설정
                        self.thresholds[pattern] = mse_mean + next_multiplier * mse_std
                        print(f"패턴 {pattern} 임계값 조정: {next_multiplier} * std")
        
        # 최대 시도 횟수 초과 - 파이프라인 조정 필요
        self.pipeline_status['current_stage'] = 'threshold_optimization_failed'
        self.pipeline_status['needs_dbscan_adjustment'] = True
        
        # 원래 임계값으로 복원
        self.thresholds = initial_thresholds.copy()
        
        return {
            "status": "error",
            "message": f"최대 시도 횟수({self.max_retries})를 초과했지만 목표 Unknown 비율({self.unknown_ratio_max:.2%})에 도달하지 못했습니다.",
            "unknown_ratio": unknown_ratio,
            "thresholds": self.thresholds.copy(),
            "recommendation": "DBSCAN 파라미터 재조정 필요"
        }
        
    def get_pipeline_status(self):
        """현재 파이프라인 상태 반환"""
        return self.pipeline_status