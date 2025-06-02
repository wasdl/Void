# 이 파일은 데이터(A,B,C) 30일치 데이터 증강 파일입니다.
# 결과는 https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing

# CSV 저장
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 물 섭취량 생성 함수
def generate_usage_with_fixed_count(pattern_type, fixed_count):
    usage = np.zeros(48)
    start_hours = []
    eating_hours = []

    if 'A' in pattern_type:
        start_hour2 = int(np.random.uniform(34, 38))
        eating_hour2 = int(np.random.uniform(6, 10))
        boost_hours = [list(range(start_hour2, start_hour2 + eating_hour2 + 1))]
        start_hours.append(start_hour2)
        eating_hours.append(eating_hour2)

    elif 'B' in pattern_type:
        start_hour1 = int(np.random.uniform(20, 26))
        eating_hour1 = int(np.random.uniform(6, 10))
        start_hour2 = int(np.random.uniform(34, 38))
        eating_hour2 = int(np.random.uniform(6, 10))
        boost_hours = [
            list(range(start_hour1, start_hour1 + eating_hour1 + 1)),
            list(range(start_hour2, start_hour2 + eating_hour2 + 1))
        ]
        start_hours.extend([start_hour1, start_hour2])
        eating_hours.extend([eating_hour1, eating_hour2])

    elif 'C' in pattern_type:
        start_hour1 = int(np.random.uniform(20, 26))
        eating_hour1 = int(np.random.uniform(6, 10))
        boost_hours = [list(range(start_hour1, start_hour1 + eating_hour1 + 1))]
        start_hours.append(start_hour1)
        eating_hours.append(eating_hour1)

    # 기존 boost 처리 동일
    waterForm = [(0, 0), (450, 500), (200, 250), (150, 200), (80, 130)]
    for i in range(len(boost_hours)):
        count = fixed_count[i] if i < len(fixed_count) else 0
        if count == 0:
            continue
        times = np.random.choice(boost_hours[i], size=count, replace=False)
        for hour in times:
            usage[hour] += int(np.random.uniform(*waterForm[count])) // 10 * 10

    max_drinks = 8 - sum(fixed_count)
    base_hours = list(set(range(48)) - set().union(*boost_hours))
    base_drinks = np.random.randint(min(3, max_drinks), max(3, max_drinks) + 1)
    if base_hours:
        times = np.random.choice(base_hours, size=base_drinks, replace=False)
        for hour in times:
            usage[hour] += int(np.random.uniform(80.0, 130.0)) // 10 * 10

    return usage, start_hours, eating_hours


# 30일치 데이터 생성
def generate_30_days(pattern, user_count_dict,start_date,num_days):
    records = []
    for person_id, counts in user_count_dict.items():
        for day_offset in range(num_days):
            date = (start_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            usage, start_hours, eating_hours = generate_usage_with_fixed_count(pattern, counts)

            for hour in range(48):
                records.append({
                    'person_id': person_id,
                    'date': date,
                    'hour': hour // 2,
                    'minute': (hour % 2) * 30,
                    'amount': usage[hour],
                    'pattern': pattern,
                    'start_hours': str(start_hours),
                    'eating_hours': str(eating_hours),
                    'counts': str(counts)
                })
    return pd.DataFrame(records)


# 생성 시작
def extendData(file_path,
               num_days = 30
               ):

    # 원본 1일치 파일 불러오기 (counts 포함되어 있는 버전)
    dfA_one = pd.read_csv(f"{file_path}/filtered_pattern_A.csv")
    dfB_one = pd.read_csv(f"{file_path}/filtered_pattern_B.csv")
    dfC_one = pd.read_csv(f"{file_path}/filtered_pattern_C.csv")

    # 유저별 counts 추출 (중복 제거) 및 문자열 → 리스트 변환
    dfA_counts = dfA_one[['person_id', 'counts']].drop_duplicates()
    dfB_counts = dfB_one[['person_id', 'counts']].drop_duplicates()
    dfC_counts = dfC_one[['person_id', 'counts']].drop_duplicates()

    dfA_counts['counts'] = dfA_counts['counts'].apply(eval)
    dfB_counts['counts'] = dfB_counts['counts'].apply(eval)
    dfC_counts['counts'] = dfC_counts['counts'].apply(eval)

    # 딕셔너리로 변환
    user_counts = {
        'A': dfA_counts.set_index('person_id')['counts'].to_dict(),
        'B': dfB_counts.set_index('person_id')['counts'].to_dict(),
        'C': dfC_counts.set_index('person_id')['counts'].to_dict(),
    }

    # 날짜 설정
    start_date = datetime(2022, 2, 1)

    dfA = generate_30_days('A', user_counts['A'],start_date,num_days)
    dfB = generate_30_days('B', user_counts['B'],start_date,num_days)
    dfC = generate_30_days('C', user_counts['C'],start_date,num_days)

    # 저장하고 싶다면:
    dfA.to_csv(f"{file_path}/pattern_A_30days.csv", index=False)
    dfB.to_csv(f"{file_path}/pattern_B_30days.csv", index=False)
    dfC.to_csv(f"{file_path}/pattern_C_30days.csv", index=False)
    return dfA,dfB,dfC