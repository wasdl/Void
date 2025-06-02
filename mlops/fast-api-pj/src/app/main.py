from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from app.routers.api import router as api_router
from pathlib import Path
from app.scheduler import start_scheduler, stop_scheduler

app = FastAPI(
    title="머신러닝 API",
    description="FastAPI를 이용한 머신러닝 모델 서비스",
    version="0.1.0"
)

# 라우터 등록
app.include_router(api_router)

# 정적 파일 경로 설정
STATIC_DIR = Path(__file__).parent / "static"

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 애플리케이션 시작 시 스케줄러 시작
@app.on_event("startup")
async def startup_event():
    start_scheduler()

# 애플리케이션 종료 시 스케줄러 중지
@app.on_event("shutdown")
async def shutdown_event():
    stop_scheduler()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 