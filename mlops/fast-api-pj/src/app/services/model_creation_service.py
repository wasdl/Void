from app.models.model_creation.create_base_model import BaseModelCreator
from app.models.model_creation.create_personal_model import PersonalModelCreator
from app.config import get_paths_dict
import os
import pandas as pd

class ModelCreationService:
    def __init__(self):
        """모델 생성 서비스 초기화"""
        self.base_model_creator = BaseModelCreator()
        self.personal_model_creator = PersonalModelCreator()
        self.paths = get_paths_dict()
        
        # 데이터 파일 경로 설정
        self.paths['data_file'] = os.path.join(self.paths.get('data', ''), 'all_features_merged.csv')
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'needs_dbscan_adjustment': False,
            'needs_autoencoder_adjustment': False,
            'performance_history': {}
        }
    
    def create_pattern(self, pattern_names, people_cnt):
        """
        패턴 데이터 생성 함수
        
        Args:
            pattern_names: 생성할 패턴 이름 목록
            people_cnt: 사용자 수
            
        Returns:
            dict: 결과 정보
        """
        try:
            from app.models.process_data.making_data import MakingData
            making_data = MakingData()
            result = making_data.create_pattern(pattern_names, people_cnt)
            
            return result
        except Exception as e:
            return {
                "status": "error",
                "error": f"패턴 생성 중 오류 발생: {str(e)}"
            }
    
    def added_pattern(self, pattern_names):
        """
        새로운 패턴 추가 및 모델 갱신
        
        Args:
            pattern_names: 추가할 패턴 이름 목록
            
        Returns:
            dict: 결과 정보
        """
        try:
            # 파이프라인 시작
            self.pipeline_status = {
                'current_stage': 'adding_patterns',
                'last_error': None,
                'needs_dbscan_adjustment': False,
                'needs_autoencoder_adjustment': False,
                'performance_history': {}
            }
            
            # DBSCAN 이상치 제거 실행
            from app.models.process_data.outlier_detection import OutlierDetection
            outlier_detector = OutlierDetection()
            
            path = self.paths.get('data', '')
            outlier_result = outlier_detector.process_multi_patterns(path, pattern_names)
            
            # DBSCAN 결과 확인
            if outlier_result["status"] == "error" or outlier_result.get("needs_pipeline_adjustment", False):
                self.pipeline_status['current_stage'] = 'dbscan_failed'
                self.pipeline_status['last_error'] = "DBSCAN 이상치 제거 실패"
                self.pipeline_status['needs_dbscan_adjustment'] = True
                
                return {
                    "status": "error",
                    "error": "패턴에 대한 이상치 제거 실패, DBSCAN 파라미터 조정 필요",
                    "details": outlier_result,
                    "pipeline_status": self.pipeline_status
                }
            
            # 오토인코더 분류기 학습
            from app.services.classifier_service import ClassifierService
            classifier_service = ClassifierService()
            classifier_result = classifier_service.train_classifiers(pattern_names)
            
            # 오토인코더 결과 확인
            if classifier_result["status"] == "error" or classifier_result.get("needs_dbscan_adjustment", False):
                self.pipeline_status['current_stage'] = 'autoencoder_failed'
                self.pipeline_status['last_error'] = "오토인코더 분류기 학습 실패"
                self.pipeline_status['needs_autoencoder_adjustment'] = True
                
                return {
                    "status": "error",
                    "error": "패턴에 대한 오토인코더 학습 실패, 임계값 조정 필요",
                    "details": classifier_result,
                    "pipeline_status": self.pipeline_status
                }
            
            # 베이스 모델 생성
            file_path = self.paths.get('data_file', '')
            base_model_result = self.base_model_creator.create_base_model(path, file_path)
            
            # 베이스 모델 결과 확인
            if base_model_result["status"] == "error":
                self.pipeline_status['current_stage'] = 'base_model_failed'
                self.pipeline_status['last_error'] = base_model_result.get("message", "베이스 모델 생성 실패")
                
                if base_model_result.get("pipeline_status", {}).get("needs_autoencoder_adjustment", False):
                    self.pipeline_status['needs_autoencoder_adjustment'] = True
                
                return {
                    "status": "error",
                    "error": "패턴에 대한 베이스 모델 생성 실패",
                    "details": base_model_result,
                    "pipeline_status": self.pipeline_status
                }
            
            # 전체 파이프라인 완료
            self.pipeline_status['current_stage'] = 'patterns_added_successfully'
            
            return {
                "status": "success",
                "message": f"{len(pattern_names)}개 패턴에 대한 모델이 갱신되었습니다.",
                "details": {
                    "outlier_detection": outlier_result,
                    "classifier": classifier_result,
                    "base_model": base_model_result
                },
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'pattern_addition_error'
            self.pipeline_status['last_error'] = str(e)
            
            return {
                "status": "error",
                "error": f"패턴 추가 및 모델 갱신 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
    
    def update_models(self):
        """
        모델 자동 업데이트 함수 (DQM 문서 기준: 7일 주기)
        
        Returns:
            dict: 업데이트 결과
        """
        # 파이프라인 시작
        self.pipeline_status = {
            'current_stage': 'updating_models',
            'last_error': None,
            'needs_dbscan_adjustment': False,
            'needs_autoencoder_adjustment': False,
            'performance_history': {}
        }
        
        try:
            # 1. 14일 성능 추이 확인 (DQM 문서 기준)
            path = self.paths.get('data', '')
            trend_result = self.base_model_creator.check_performance_trend(path)
            
            # 성능 추이 확인 결과 처리
            if trend_result["status"] == "warning" and trend_result.get("needs_pipeline_restart", False):
                self.pipeline_status['current_stage'] = 'performance_decline_detected'
                
                print(f"14일 연속 성능 저하 감지. 전체 파이프라인 재실행이 필요합니다.")
                print(f"연속 성능 저하 일수: {trend_result.get('consecutive_decline_days', 0)}일")
                
                # 전체 파이프라인 재실행 (DQM 문서 기준: 14일 연속 성능 저하 시)
                from app.services.process_data_service import ProcessDataService
                process_data_service = ProcessDataService()
                
                # 1. 특성 생성
                file_path = self.paths.get('data_file', '')
                process_result = process_data_service.init_process([file_path])
                
                # 실패 시
                if process_result["status"] == "error":
                    self.pipeline_status['current_stage'] = 'process_data_failed'
                    self.pipeline_status['last_error'] = process_result.get("error", "데이터 처리 실패")
                    
                    return {
                        "status": "error",
                        "message": "14일 연속 성능 저하로 파이프라인 재실행 시도했으나 실패했습니다.",
                        "details": process_result,
                        "pipeline_status": self.pipeline_status
                    }
                
                # 성공 시
                self.pipeline_status['current_stage'] = 'full_pipeline_restarted'
                
                return {
                    "status": "success",
                    "message": "14일 연속 성능 저하로 전체 파이프라인을 재실행했습니다.",
                    "details": process_result,
                    "pipeline_status": self.pipeline_status
                }
            
            # 2. 정기 업데이트 (DQM 문서 기준: 7일 주기)
            path = self.paths.get('data', '')
            file_path = self.paths.get('data_file', '')
            
            # 베이스 모델 업데이트
            base_model_result = self.base_model_creator.create_base_model(path, file_path)
            
            # 베이스 모델 업데이트 실패 시
            if base_model_result["status"] == "error":
                self.pipeline_status['current_stage'] = 'base_model_update_failed'
                self.pipeline_status['last_error'] = base_model_result.get("message", "베이스 모델 업데이트 실패")
                
                if base_model_result.get("pipeline_status", {}).get("needs_autoencoder_adjustment", False):
                    self.pipeline_status['needs_autoencoder_adjustment'] = True
                
                return {
                    "status": "error",
                    "message": "베이스 모델 업데이트 실패",
                    "details": base_model_result,
                    "pipeline_status": self.pipeline_status
                }
            
            # 3. 개인화 모델 업데이트 - 사용자별 처리
            personal_models_results = {}
            
            # 사용자 목록 가져오기
            try:
                df = pd.read_csv(file_path)
                user_ids = df['person_id'].unique()
                
                # 사용자별 개인화 모델 업데이트
                for user_id in user_ids:
                    personal_model_result = self.personal_model_creator.create_personal_model(user_id, path, file_path)
                    personal_models_results[f"user_{user_id}"] = personal_model_result
            except Exception as e:
                self.pipeline_status['current_stage'] = 'personal_models_update_failed'
                self.pipeline_status['last_error'] = str(e)
                
                return {
                    "status": "warning",
                    "message": f"베이스 모델은 업데이트 됐으나 개인화 모델 업데이트 중 오류 발생: {str(e)}",
                    "details": {
                        "base_model": base_model_result,
                        "personal_models": personal_models_results
                    },
                    "pipeline_status": self.pipeline_status
                }
            
            # 전체 업데이트 성공
            self.pipeline_status['current_stage'] = 'update_models_completed'
            
            return {
                "status": "success",
                "message": "모델 정기 업데이트 완료 (7일 주기)",
                "details": {
                    "performance_trend": trend_result,
                    "base_model": base_model_result,
                    "personal_models": personal_models_results
                },
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'update_models_error'
            self.pipeline_status['last_error'] = str(e)
            
            return {
                "status": "error",
                "message": f"모델 업데이트 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
    
    def personalize_models(self):
        """
        개인화 모델 자동 업데이트 함수 (DQM 문서 기준: 7일 주기)
        
        Returns:
            dict: 업데이트 결과
        """
        # 파이프라인 시작
        self.pipeline_status = {
            'current_stage': 'updating_personal_models',
            'last_error': None,
            'performance_history': {}
        }
        
        try:
            path = self.paths.get('data', '')
            file_path = self.paths.get('data_file', '')
            
            # 데이터 파일 확인
            if not os.path.exists(file_path):
                self.pipeline_status['current_stage'] = 'data_file_not_found'
                self.pipeline_status['last_error'] = f"데이터 파일이 존재하지 않습니다: {file_path}"
                
                return {
                    "status": "error",
                    "message": f"데이터 파일이 존재하지 않습니다: {file_path}",
                    "pipeline_status": self.pipeline_status
                }
                
            # 데이터에서 사용자 목록 가져오기
            data = pd.read_csv(file_path)
            user_ids = data['person_id'].unique()
            
            if len(user_ids) == 0:
                self.pipeline_status['current_stage'] = 'no_users_found'
                self.pipeline_status['last_error'] = "데이터에 사용자가 없습니다."
                
                return {
                    "status": "warning",
                    "message": "데이터에 사용자가 없습니다.",
                    "pipeline_status": self.pipeline_status
                }
                
            # 사용자별 개인화 모델 업데이트
            results = {}
            for user_id in user_ids:
                try:
                    result = self.personal_model_creator.create_personal_model(user_id, path, file_path)
                    results[str(user_id)] = result
                    self.pipeline_status['performance_history'][f"user_{user_id}"] = result.get("pipeline_status", {}).get("performance_history", {})
                except Exception as e:
                    results[str(user_id)] = {
                        "status": "error",
                        "message": f"사용자 {user_id} 모델 업데이트 중 오류 발생: {str(e)}"
                    }
                    
            # 결과 요약
            success_count = sum(1 for r in results.values() if r.get("status") == "success")
            
            self.pipeline_status['current_stage'] = 'personal_models_update_completed'
            
            return {
                "status": "success",
                "message": f"{success_count}/{len(user_ids)} 사용자의 개인화 모델이 업데이트되었습니다.",
                "details": results,
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'personal_models_update_error'
            self.pipeline_status['last_error'] = str(e)
            
            return {
                "status": "error",
                "message": f"개인화 모델 업데이트 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
            
    def get_pipeline_status(self):
        """파이프라인 상태 반환"""
        return self.pipeline_status
