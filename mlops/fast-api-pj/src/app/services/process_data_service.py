from app.models.process_data.making_data import MakingData
from app.models.process_data.extend_feature import FeatureExtender
from app.models.process_data.evaluate_feature import FeatureEvaluator
from app.models.classifier.autoencoder import AutoencoderClassifier
from app.models.model_creation.create_base_model import BaseModelCreator
from app.services.classifier_service import ClassifierService
from app.config import get_paths_dict
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing.log')
    ]
)
logger = logging.getLogger('data_processing')

class ProcessDataService:
    def __init__(self):
        self.making_data = MakingData()
        self.feature_extender = FeatureExtender()
        self.feature_evaluator = FeatureEvaluator()
        self.autoencoder = AutoencoderClassifier()
        self.base_model_creator = BaseModelCreator()
        self.classifier_service = ClassifierService()
        self.paths = get_paths_dict()
        
        # 파이프라인 상태 추적
        self.pipeline_status = {
            'current_stage': 'init',
            'last_error': None,
            'dbscan_params': {},
            'autoencoder_params': {},
            'base_model_params': {},
            'retry_count': {}
        }

    def create_pattern(self, pattern_names, people_cnt):
        """패턴 데이터 생성"""
        logger.info(f"패턴 생성 시작: {pattern_names}, 인원 수: {people_cnt}")
        return self.making_data.create_pattern(pattern_names, people_cnt)

    def init_process(self, file_names):
        """
        DQM 프로세스에 따른 데이터 처리 및 모델 학습 전체 파이프라인
        
        Args:
            file_names: 처리할 파일 이름 목록
            
        Returns:
            dict: 처리 결과
        """
        # 파이프라인 상태 초기화
        self.pipeline_status = {
            'current_stage': 'init_process_started',
            'last_error': None,
            'dbscan_params': {},
            'autoencoder_params': {},
            'base_model_params': {},
            'retry_count': {
                'dbscan': 0,
                'autoencoder': 0,
                'base_model': 0
            }
        }
        
        logger.info(f"데이터 처리 및 모델 학습 파이프라인 시작: {file_names}")
        
        try:
            # A. DBSCAN 군집화 및 이상치 제거 단계
            self.pipeline_status['current_stage'] = 'dbscan_processing'
            
            # 1. 특성 생성 (시계열 데이터 → 특성 벡터)
            logger.info("특성 생성 시작...")
            pattern_files = self.feature_extender.create_feature(file_names)
            
            if not pattern_files:
                self.pipeline_status['current_stage'] = 'feature_creation_failed'
                self.pipeline_status['last_error'] = "특성 생성 실패"
                logger.error("특성 생성 실패")
                return {
                    "status": "error",
                    "error": "특성 생성에 실패했습니다.",
                    "pipeline_status": self.pipeline_status
                }
            
            # 2. 이상치 제거 (DBSCAN)
            logger.info("DBSCAN 이상치 제거 시작...")
            from app.models.process_data.outlier_detection import OutlierDetection
            outlier_detector = OutlierDetection()
            
            outlier_result = outlier_detector.process_multi_patterns(self.paths.get('data', ''))
            self.pipeline_status['dbscan_params'] = outlier_detector.get_pipeline_status()
            
            # 이상치 제거 결과 확인
            if outlier_result["status"] == "error" or outlier_result.get("needs_pipeline_adjustment", False):
                self.pipeline_status['current_stage'] = 'dbscan_failed'
                self.pipeline_status['last_error'] = "DBSCAN 이상치 제거 실패"
                logger.error(f"DBSCAN 이상치 제거 실패: {outlier_result.get('message', '')}")
                return {
                    "status": "error",
                    "error": "DBSCAN 이상치 제거에 실패했습니다. 파라미터 조정이 필요합니다.",
                    "details": outlier_result,
                    "pipeline_status": self.pipeline_status
                }
            
            # B. 오토인코더 학습 단계
            self.pipeline_status['current_stage'] = 'autoencoder_training'
            
            # 3. 특성 평가 (VIF, 중요도 기반)
            logger.info("특성 평가 시작...")
            evaluation_results = self.feature_evaluator.evaluate_feature()
            
            # 4. 오토인코더 분류기 학습
            logger.info("오토인코더 분류기 학습 시작...")
            classifier_results = self.classifier_service.train_classifiers()
            self.pipeline_status['autoencoder_params'] = self.autoencoder.get_pipeline_status()
            
            # 오토인코더 학습 결과 확인
            if classifier_results["status"] == "error" or classifier_results.get("needs_dbscan_adjustment", False):
                # DQM 기준: 오토인코더 실패 시 DBSCAN 파라미터 재조정 필요
                self.pipeline_status['current_stage'] = 'autoencoder_failed'
                self.pipeline_status['last_error'] = "오토인코더 분류기 학습 실패"
                self.pipeline_status['retry_count']['dbscan'] += 1
                
                logger.error(f"오토인코더 학습 실패: {classifier_results.get('message', '')}")
                logger.info(f"DBSCAN 파라미터 재조정 필요. 재시도 횟수: {self.pipeline_status['retry_count']['dbscan']}")
                
                # 재시도 횟수 제한 확인
                if self.pipeline_status['retry_count']['dbscan'] >= 3:
                    return {
                        "status": "error",
                        "error": "DBSCAN 파라미터 재조정 최대 시도 횟수를 초과했습니다.",
                        "details": classifier_results,
                        "pipeline_status": self.pipeline_status
                    }
                
                # DBSCAN 파라미터 조정 후 처음부터 다시 시작
                # 실제 구현에서는 이전 파라미터를 기반으로 조정해야 함
                # 지금은 예시로 파라미터를 변경하는 대신 다시 시작
                return self.init_process(file_names)
            
            # C. XGBoost ALL 모델 학습 단계
            self.pipeline_status['current_stage'] = 'all_model_training'
            
            # 5. 베이스 모델 학습 (ALL 모델)
            logger.info("베이스 모델 학습 시작...")
            base_model_result = self.base_model_creator.create_base_model(
                self.paths.get('data', ''),
                self.paths.get('data_file', '')
            )
            self.pipeline_status['base_model_params'] = self.base_model_creator.get_pipeline_status()
            
            # ALL 모델 학습 결과 확인
            if base_model_result["status"] == "error" or base_model_result.get("pipeline_status", {}).get("needs_autoencoder_adjustment", False):
                # DQM 기준: ALL 모델 실패 시 오토인코더 임계값 재조정 필요
                self.pipeline_status['current_stage'] = 'all_model_failed'
                self.pipeline_status['last_error'] = "베이스 ALL 모델 학습 실패"
                self.pipeline_status['retry_count']['autoencoder'] += 1
                
                logger.error(f"ALL 모델 학습 실패: {base_model_result.get('message', '')}")
                logger.info(f"오토인코더 임계값 재조정 필요. 재시도 횟수: {self.pipeline_status['retry_count']['autoencoder']}")
                
                # 재시도 횟수 제한 확인
                if self.pipeline_status['retry_count']['autoencoder'] >= 3:
                    return {
                        "status": "error",
                        "error": "오토인코더 임계값 재조정 최대 시도 횟수를 초과했습니다.",
                        "details": base_model_result,
                        "pipeline_status": self.pipeline_status
                    }
                    
                # 오토인코더 임계값 조정 후 오토인코더 단계부터 다시 시작
                # 현재는 단순 재시도만 구현
                self.pipeline_status['current_stage'] = 'restarting_from_autoencoder'
                logger.info("오토인코더 단계부터 재시작...")
                
                # B. 오토인코더 재학습 (임계값 조정)
                classifier_results = self.classifier_service.train_classifiers()
                
                # 오토인코더 재학습 후 다시 C, D 단계 진행
                return self.init_process(file_names)
            
            # D. XGBoost 타입별 모델 학습 단계
            # 타입별 모델 학습 결과는 이미 base_model_result에 포함되어 있음
            if base_model_result.get("status") == "warning" or base_model_result.get("pipeline_status", {}).get("needs_autoencoder_adjustment", False):
                # DQM 기준: 타입별 모델 실패 시 오토인코더 임계값 재조정 필요
                self.pipeline_status['current_stage'] = 'pattern_models_partially_failed'
                self.pipeline_status['last_error'] = "일부 패턴별 모델 학습 실패"
                self.pipeline_status['retry_count']['autoencoder'] += 1
                
                logger.warning(f"일부 패턴별 모델 학습 실패: {base_model_result.get('message', '')}")
                logger.info(f"오토인코더 임계값 재조정 필요. 재시도 횟수: {self.pipeline_status['retry_count']['autoencoder']}")
                
                # 재시도 횟수 제한 확인
                if self.pipeline_status['retry_count']['autoencoder'] >= 3:
                    return {
                        "status": "warning",
                        "message": "일부 패턴별 모델 학습이 실패했으나, 최대 재시도 횟수를 초과했습니다.",
                        "details": base_model_result,
                        "pipeline_status": self.pipeline_status
                    }
                    
                # 오토인코더 임계값 조정 후 오토인코더 단계부터 다시 시작
                # 현재는 단순 경고만 반환
                logger.warning("일부 패턴 모델 학습 실패를 감지했지만, 전체 파이프라인은 완료되었습니다.")
            
            # 전체 파이프라인 완료
            self.pipeline_status['current_stage'] = 'init_process_completed'
            logger.info("데이터 처리 및 모델 학습 파이프라인 완료")
            
            return {
                "status": "success",
                "message": "데이터 처리, 특성 생성 및 모델 학습이 완료되었습니다.",
                "details": {
                    "feature_creation": pattern_files,
                    "outlier_detection": outlier_result,
                    "feature_evaluation": evaluation_results,
                    "classifier": classifier_results,
                    "base_model": base_model_result
                },
                "pipeline_status": self.pipeline_status
            }
            
        except Exception as e:
            self.pipeline_status['current_stage'] = 'init_process_error'
            self.pipeline_status['last_error'] = str(e)
            logger.error(f"파이프라인 실행 중 오류 발생: {str(e)}")
            
            return {
                "status": "error",
                "error": f"데이터 처리 및 모델 학습 중 오류 발생: {str(e)}",
                "pipeline_status": self.pipeline_status
            }
    
    def get_pipeline_status(self):
        """파이프라인 상태 반환"""
        return self.pipeline_status

