# MLOps 정수기 데이터 분석 프로젝트

## 프로젝트 소개
이 프로젝트는 정수기 사용 데이터를 수집, 분석하고 머신러닝 모델을 통해 예측하는 MLOps 시스템입니다. 
DQM(Data Quality Management) 프로세스를 적용하여 지속적인 데이터 품질 관리와 모델 개선을 자동화합니다.

## 주요 특징
- DBSCAN 기반 군집화를 통한 패턴 분류
- 오토인코더 기반 비정상 패턴 감지
- XGBoost 기반 예측 모델링
- 자동화된 모델 학습 및 개선 파이프라인
- FastAPI 기반 웹 서비스

## 요구사항
Poetry 외에 추가적으로 필요한 라이브러리:
- cupy
- cudf
- numba
- torch
- CUDA가 설치되어 있어야 함

## 주요 구성 요소

### 1. 데이터 처리 파이프라인
- `outlier_detection.py`: DBSCAN 기반 이상치 탐지
- `making_data.py`: 패턴 데이터 생성
- `extend_feature.py`: 특성 확장
- `evaluate_feature.py`: 특성 평가

### 2. 머신러닝 모델
- `autoencoder.py`: 패턴 분류를 위한 오토인코더
- `dbscan.py`: 클러스터링을 위한 DBSCAN 구현
- `create_base_model.py`: XGBoost 기반 기본 모델
- `create_personal_model.py`: 개인화 모델

### 3. 서비스 레이어
- `process_data_service.py`: 데이터 처리 서비스
- `model_creation_service.py`: 모델 생성 서비스
- `classifier_service.py`: 분류 서비스

### 4. API 및 시스템
- `api.py`: FastAPI 라우터
- `main.py`: 애플리케이션 엔트리포인트
- `scheduler.py`: 작업 스케줄러
- `config.py`: 시스템 설정
- `helpers.py`: 유틸리티 함수

## DQM 파이프라인
프로젝트는 DQM_Process.md에 명시된 프로세스를 구현합니다:

1. DBSCAN 군집화 및 검증 (실루엣 계수 ≥ 0.6)
2. 오토인코더 학습 및 패턴 분류 (Unknown 비율 ≤ 5%)
3. 베이스 모델 학습 (XGBoost ALL, R2 Score ≥ 0.6)
4. 패턴별 모델 학습 (정확도 ≥ 60%)
5. 자동 성능 모니터링 및 재학습 (7일 주기)
6. 성능 저하 감지 및 파이프라인 재실행 (14일 연속 저하 시)
