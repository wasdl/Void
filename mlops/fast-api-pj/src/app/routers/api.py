from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.process_data_service import ProcessDataService
from app.services.model_creation_service import ModelCreationService
from app.services.classifier_service import ClassifierService
from app.models.classifier.dbscan import DBSCAN
import numpy as np

router = APIRouter(prefix="/api", tags=["API"])

# 서비스 인스턴스 생성
process_data_service = ProcessDataService()
model_creation_service = ModelCreationService()
classifier_service = ClassifierService()

@router.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {"message": "머신러닝 API에 오신 것을 환영합니다"}


class patternRequest(BaseModel):
    pattern_names: List[str]
    people_cnt: int


class userDataRequest(BaseModel):
    file_names: List[str]


class dataRequest(BaseModel):
    file_names: List[str]


class addedPatternRequest(BaseModel):
    pattern_names: List[str]
    
    
class PatternDataRequest(BaseModel):
    data: List[float]


class ClusteringRequest(BaseModel):
    data: List[List[float]]
    eps: Optional[float] = 0.9
    min_samples: Optional[int] = 3


@router.post("/create_pattern")
async def create_pattern(request: patternRequest):
    """패턴 데이터를 생성합니다"""
    if request.pattern_names is None:
        raise HTTPException(status_code=400, detail="pattern_names is required")
    if request.people_cnt is None:
        request.people_cnt = 100
    if request.people_cnt < 100:
        raise HTTPException(status_code=400, detail="people_cnt must be greater than 100")
    if request.people_cnt > 10000:
        raise HTTPException(status_code=400, detail="people_cnt must be less than 10000")
    
    result = model_creation_service.create_pattern(request.pattern_names, request.people_cnt)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.post("/init_process")
async def init_process(request: dataRequest):
    """DQM 프로세스에 따라 데이터 처리 및 모델 학습을 수행합니다"""
    if request.file_names is None:
        raise HTTPException(status_code=400, detail="file_names is required")
    
    result = process_data_service.init_process(request.file_names)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return result
    

@router.post("/added_pattern")
async def added_pattern(request: addedPatternRequest):
    """새로운 패턴을 추가하고 모델을 갱신합니다"""
    if request.pattern_names is None:
        raise HTTPException(status_code=400, detail="pattern_names is required")
    
    result = model_creation_service.added_pattern(request.pattern_names)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return result


@router.get("/pipeline_status")
async def get_pipeline_status(service: str = Query("process_data", description="확인할 서비스 (process_data, model_creation)")):
    """현재 파이프라인 상태를 확인합니다"""
    if service == "process_data":
        return process_data_service.get_pipeline_status()
    elif service == "model_creation":
        return model_creation_service.get_pipeline_status()
    else:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 서비스: {service}")


@router.post("/classify_pattern")
async def classify_pattern(request: PatternDataRequest):
    """사용자 데이터 패턴을 분류합니다 (미할당 사용자 처리용)"""
    if not request.data or len(request.data) != 48:
        raise HTTPException(status_code=400, detail="data must contain 48 values (daily usage pattern)")
    
    result = classifier_service.classify_pattern(request.data)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result


@router.post("/personalize_model/{user_id}")
async def personalize_model(user_id: int):
    """단일 사용자에 대한 개인화 모델을 생성합니다"""
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    from app.models.model_creation.create_personal_model import PersonalModelCreator
    personal_model_creator = PersonalModelCreator()
    
    result = personal_model_creator.create_personal_model(user_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result


@router.get("/model_performance")
async def get_model_performance(pattern: Optional[str] = None, user_id: Optional[int] = None):
    """모델 성능 지표를 확인합니다 (특정 패턴 또는 사용자별)"""
    # 베이스 모델 성능
    base_performance = model_creation_service.base_model_creator.check_performance_trend(
        model_creation_service.paths.get('data', '')
    )
    
    # 요청에 따라 특정 패턴 또는 사용자 성능 필터링
    if pattern:
        # 패턴별 필터링
        if "patterns" in base_performance and pattern in base_performance["patterns"]:
            base_performance = {
                "pattern": pattern,
                "performance": base_performance["patterns"][pattern]
            }
        else:
            base_performance = {"pattern": pattern, "performance": "No data available"}
    
    # 사용자별 개인화 모델 성능 (요청된 경우)
    if user_id:
        from app.models.model_creation.create_personal_model import PersonalModelCreator
        personal_model_creator = PersonalModelCreator()
        
        # 파일 위치 찾기 (실제 구현에서는 더 효율적인 방법 필요)
        import os
        ensemble_dir = os.path.join(model_creation_service.paths.get('data', ''), 
                                    "VoID_WaterPurifier", "models", "ensemble")
        
        summary_path = os.path.join(ensemble_dir, f'ensemble_summary_user_{user_id}.csv')
        
        if os.path.exists(summary_path):
            import pandas as pd
            user_performance = pd.read_csv(summary_path).to_dict(orient='records')
        else:
            user_performance = {"user_id": user_id, "performance": "No data available"}
            
        return {
            "base_model": base_performance,
            "personal_model": user_performance
        }
    
    return {"performance": base_performance}


@router.post("/update_models")
async def update_models():
    """모델을 수동으로 업데이트합니다 (7일 주기 자동 업데이트와 동일)"""
    result = model_creation_service.update_models()
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return result


@router.post("/clustering/dbscan")
async def run_dbscan_clustering(request: ClusteringRequest):
    """DBSCAN 군집화를 실행합니다."""
    if not request.data:
        raise HTTPException(status_code=400, detail="데이터가 필요합니다.")
    
    if len(request.data) < 100:
        raise HTTPException(status_code=400, detail="최소 100개 이상의 데이터 포인트가 필요합니다.")
    
    try:
        # DBSCAN 클래스 인스턴스 생성
        dbscan = DBSCAN()
        
        # 파라미터 수동 설정 (요청에서 전달된 경우)
        if request.eps != 0.9:
            dbscan.eps = request.eps
        if request.min_samples != 3:
            dbscan.min_samples = request.min_samples
        
        # DBSCAN 군집화 실행
        result = dbscan.fit_transform(np.array(request.data))
        
        return {
            "status": result["status"],
            "message": result["message"],
            "cluster_counts": {str(label): int((result["labels"] == label).sum()) 
                              for label in np.unique(result["labels"])},
            "silhouette_score": float(result["silhouette_score"]),
            "dbscan_params": result["dbscan_params"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DBSCAN 군집화 중 오류 발생: {str(e)}")

