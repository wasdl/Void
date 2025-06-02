import pandas as pd
import os
file_path = ""

# 1 초기 파이프라인
if os.path.exists(f"{file_path}/thresholds.pkl"):
    # 1-1. 데이터 생성(A,B,C)
    from makingData import *;

    makingData(file_path,1000,1)

    # 1-2. 데이터 이상치 제거
    from outlierDetection import *;
    dfA = pd.read_csv(f"{file_path}/water_consumption_2022_02_01_A.csv")
    dfB = pd.read_csv("{file_path}/water_consumption_2022_02_01_B.csv")
    dfC = pd.read_csv("{file_path}/water_consumption_2022_02_01_C.csv")

    for df, label in zip([dfA, dfB, dfC], ['A', 'B', 'C']):
        process_pattern_dbscan(df,file_path, label)

    # 1-3 데이터 증강(30일치)
    from extendData import *
    dfA,dfB,dfC=extendData(file_path,30)

    # 1-4 데이터 패턴별 classifier 생성
    from classifier import *
    classifier(dfA,dfB,dfC)

    # 1-5 파운데이션 모델을 위한 feature 생성
    from createFeature import create_feature
    files = [
        'pattern_A_30days.csv',
        'pattern_B_30days.csv',
        'pattern_C_30days.csv'
    ]
    df_one,df_all=create_feature(file_path, files, 0)

    #1-6 feature 특성 확인

    from evaluateFeature import evaluate_feature

    file = '1_day_feature.csv'
    # evaluate_feature(file_path, file)

    # 1-7 파운데이션 모델 학습
    from createBaseModel import create_base_model
    # 베이스 모델 만들기
    create_base_model(file_path, file)

# 2 개인화모델 생성
userId=23 # 개인화 ID 설정
from createPeronalModel import create_personal_model
create_personal_model(userId,file_path)

# 3 새로운 패턴 생성

# 3-1 D 패턴 생성 및 Autoencoder 생성
generateD(1000,30,file_path)

# 3-2 D 패턴용 feature 생성
new_files=[
    'pattern_D_30days.csv'
    ]
df_one,df_all=create_feature(file_path, new_files, 0)

file = '1_day_feature.csv'

# 3-3 D 패턴용 파운데이션 모델 생성
create_base_model(file_path, file)
