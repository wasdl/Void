from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_ERROR
from app.services.model_creation_service import ModelCreationService
from app.utils.helpers import get_timestamp
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'scheduler_{get_timestamp()}.log')
    ]
)
logger = logging.getLogger('mlops_scheduler')

# 서비스 인스턴스 생성
model_creation_service = ModelCreationService()

# 스케줄러 설정
scheduler = BackgroundScheduler()

# 작업 실패 시 오류 처리 및 알림 함수
def job_error_listener(event):
    """작업 실패 시 호출될 콜백 함수"""
    if event.exception:
        job = scheduler.get_job(event.job_id)
        logger.error(f"작업 실패: {job.name}, 오류: {str(event.exception)}")
        
        # 오류 발생 시 다음 단계 결정
        if job.name == 'check_performance_trend_job':
            # 성능 추이 검증 실패 - 관리자 알림 필요
            logger.error("성능 추이 검증 작업이 실패했습니다. 관리자 확인이 필요합니다.")
        
        elif job.name == 'update_models_job':
            # 모델 업데이트 실패 - 개인화 모델 업데이트는 건너뛰기
            logger.error("베이스 모델 업데이트 작업이 실패했습니다. 개인화 모델 업데이트를 건너뜁니다.")
            
            # 개인화 모델 업데이트 작업을 일시적으로 비활성화
            scheduler.pause_job('personalize_models_job')
            logger.info("개인화 모델 업데이트 작업이 일시 중지되었습니다.")

# 성능 추이 검증 함수
def check_performance_trend():
    """모델 성능 추이를 검증하고 필요시 파이프라인 재실행"""
    logger.info("성능 추이 검증 시작...")
    
    try:
        # 기본 모델 성능 추이 검증 (DQM 문서 기준: 14일 연속 성능 저하 시 파이프라인 재실행)
        result = model_creation_service.update_models()
        
        if result.get("status") == "success":
            logger.info(f"성능 추이 검증 완료: {result.get('message')}")
            
            # 개인화 모델 업데이트 작업이 일시 중지된 경우 재개
            if scheduler.get_job('personalize_models_job').paused:
                scheduler.resume_job('personalize_models_job')
                logger.info("개인화 모델 업데이트 작업이 재개되었습니다.")
                
        else:
            logger.warning(f"성능 추이 검증 경고: {result.get('message')}")
            
            # 파이프라인 상태 확인
            pipeline_status = result.get('pipeline_status', {})
            if pipeline_status.get('needs_dbscan_adjustment', False):
                logger.warning("DBSCAN 파라미터 조정이 필요합니다.")
            
            if pipeline_status.get('needs_autoencoder_adjustment', False):
                logger.warning("오토인코더 임계값 조정이 필요합니다.")
                
    except Exception as e:
        logger.error(f"성능 추이 검증 중 오류 발생: {str(e)}")
        raise  # 에러 리스너에서 처리할 수 있도록 예외 전파

# 모델 업데이트 함수
def update_models():
    """정기적인 모델 업데이트 (7일 주기)"""
    logger.info("모델 정기 업데이트 시작...")
    
    try:
        # 베이스 모델 업데이트 실행 (오토인코더 포함)
        result = model_creation_service.update_models()
        
        if result.get("status") == "success":
            logger.info(f"모델 업데이트 완료: {result.get('message')}")
        else:
            logger.warning(f"모델 업데이트 경고: {result.get('message')}")
            
            # 파이프라인 상태 확인
            pipeline_status = result.get('pipeline_status', {})
            current_stage = pipeline_status.get('current_stage', '')
            
            if 'performance_decline_detected' in current_stage:
                logger.warning("14일 연속 성능 저하가 감지되어 전체 파이프라인을 재실행합니다.")
            
    except Exception as e:
        logger.error(f"모델 업데이트 중 오류 발생: {str(e)}")
        raise  # 에러 리스너에서 처리할 수 있도록 예외 전파

# 개인화 모델 업데이트 함수
def personalize_models():
    """개인화 모델 업데이트 (7일 주기)"""
    logger.info("개인화 모델 업데이트 시작...")
    
    try:
        # 개인화 모델 업데이트 실행
        result = model_creation_service.personalize_models()
        
        if result.get("status") == "success":
            logger.info(f"개인화 모델 업데이트 완료: {result.get('message')}")
        else:
            logger.warning(f"개인화 모델 업데이트 경고: {result.get('message')}")
    
    except Exception as e:
        logger.error(f"개인화 모델 업데이트 중 오류 발생: {str(e)}")
        raise  # 에러 리스너에서 처리할 수 있도록 예외 전파

def start_scheduler():
    """스케줄러를 시작하고 모델 관리 작업을 등록합니다."""
    # 에러 리스너 등록
    scheduler.add_listener(job_error_listener, EVENT_JOB_ERROR)
    
    # 1일마다 모델 성능 추이 확인 (DQM 문서 기준: 14일 연속 저하 시 파이프라인 재실행)
    scheduler.add_job(
        check_performance_trend,
        trigger=IntervalTrigger(days=1),
        id='check_performance_trend_job',
        name='check_performance_trend_job',
        replace_existing=True
    )
    
    # 7일마다 모델 업데이트 실행 (DQM 문서 기준)
    scheduler.add_job(
        update_models,
        trigger=IntervalTrigger(days=7),
        id='update_models_job',
        name='update_models_job',
        replace_existing=True
    )
    
    # 7일마다 개인화 모델 업데이트 실행
    scheduler.add_job(
        personalize_models,
        trigger=IntervalTrigger(days=7),
        id='personalize_models_job',
        name='personalize_models_job',
        replace_existing=True
    )
    
    # 스케줄러 시작
    scheduler.start()
    logger.info("스케줄러가 시작되었습니다.")

def stop_scheduler():
    """스케줄러를 중지합니다."""
    scheduler.shutdown()
    logger.info("스케줄러가 중지되었습니다.") 