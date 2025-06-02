# 이 파일은 데이터(A,B,C) 생성파일입니다.
# 결과는 https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from mlops.main import file_path


# 수정된 데이터 생성 함수
def generate_human_like_day_strong_pattern(pattern_type):
    usage = np.zeros(48)

    meta_info = {
        'count': [],
        'start_hours': [],
        'eating_hours': []
    }

    if 'A' in pattern_type:
        start_hour2 = int(np.random.uniform(34, 38))
        eating_hour2 = int(np.random.uniform(6, 10))
        boost_hours = [list(range(start_hour2, start_hour2 + eating_hour2 + 1))]
        count = [np.random.randint(1, 5)]

        meta_info['start_hours'].append(start_hour2)
        meta_info['eating_hours'].append(eating_hour2)
        meta_info['count'] = count

    elif 'B' in pattern_type:
        start_hour1 = int(np.random.uniform(20, 26))
        eating_hour1 = int(np.random.uniform(6, 10))
        start_hour2 = int(np.random.uniform(34, 38))
        eating_hour2 = int(np.random.uniform(6, 10))

        boost_hours = [
            list(range(start_hour1, start_hour1 + eating_hour1 + 1)),
            list(range(start_hour2, start_hour2 + eating_hour2 + 1))
        ]
        x = np.random.randint(1, 5)
        if x != 4:
            count = [x, np.random.randint(1, 5)]
        else:
            count = [x, np.random.randint(1, 4)]

        meta_info['start_hours'] = [start_hour1, start_hour2]
        meta_info['eating_hours'] = [eating_hour1, eating_hour2]
        meta_info['count'] = count

    elif 'C' in pattern_type:
        start_hour1 = int(np.random.uniform(20, 26))
        eating_hour1 = int(np.random.uniform(6, 10))
        boost_hours = [list(range(start_hour1, start_hour1 + eating_hour1 + 1))]
        count = [np.random.randint(1, 5)]

        meta_info['start_hours'].append(start_hour1)
        meta_info['eating_hours'].append(eating_hour1)
        meta_info['count'] = count

    waterForm = [(0, 0), (450, 500), (200, 250), (150, 200), (80, 130)]

    for i in range(len(boost_hours)):
        pattern_drink_times = np.random.choice(boost_hours[i], size=count[i], replace=False)
        for hour in pattern_drink_times:
            usage[hour] += int(np.random.uniform(waterForm[count[i]][0], waterForm[count[i]][1])) // 10 * 10

    max_drinks = 8 - sum(count)
    if len(boost_hours) == 2:
        base_hours = list(set(range(48)) - set(boost_hours[0]) - set(boost_hours[1]))
    else:
        base_hours = list(set(range(48)) - set(boost_hours[0]))

    base_drinks = np.random.randint(min(3, max_drinks), max(3, max_drinks) + 1)
    drink_times = np.random.choice(base_hours, size=base_drinks, replace=False)

    for hour in drink_times:
        drink_usage = int(np.random.uniform(80.0, 130.0)) // 10 * 10
        usage[hour] += drink_usage

    return usage, meta_info

def makingData(path,
               num_people = 1000,
               num_days = 1
               ):


    # 날짜 설정
    start_date = datetime(2022, 2, 1)

    # 데이터 생성 및 저장
    for pattern in ['A', 'B', 'C']:
        records = []
        for person_id in range(1, num_people + 1):
            if pattern == 'A':
                pid = person_id
            elif pattern == 'B':
                pid = person_id + num_people
            else:
                pid = person_id + 2 * num_people

            for day_offset in range(num_days):
                date = start_date + timedelta(days=day_offset)
                usage, meta = generate_human_like_day_strong_pattern(pattern)

                for hour in range(48):
                    records.append({
                        'person_id': pid,
                        'date': date.strftime('%Y-%m-%d'),
                        'hour': hour // 2,
                        'minute': hour % 2 * 30,
                        'amount': usage[hour],
                        'pattern': pattern,
                        'start_hours': str(meta['start_hours']),
                        'eating_hours': str(meta['eating_hours']),
                        'counts': str(meta['count'])
                    })

            df = pd.DataFrame(records)

            csv_path = f"{path}/water_consumption_2022_02_01_{pattern}.csv"
            df.to_csv(csv_path, index=False)
            print(f"{pattern} 저장 완료 → {csv_path}")

def generate_pattern_d_days(num_people=1000, num_days=30):
    records = []
    start_date = datetime(2022, 2, 1)

    for person_id in range(1, num_people + 1):
        pid = 30000 + person_id  # 다른 패턴과 겹치지 않도록

        count = np.random.randint(1, 5)

        for day_offset in range(num_days):
            date = start_date + timedelta(days=day_offset)
            usage = np.zeros(48)
            start_hour = int(np.random.uniform(10, 16))
            eating_hour = int(np.random.uniform(6, 10))

            boost_hours = list(range(start_hour, start_hour + eating_hour + 1))
            waterForm = [(0, 0), (450, 500), (200, 250), (150, 200), (80, 130)]

            pattern_drink_times = np.random.choice(boost_hours, size=count, replace=False)
            for hour in pattern_drink_times:
                usage[hour] += int(np.random.uniform(waterForm[count][0], waterForm[count][1])) // 10 * 10

            max_drinks = 8 - count
            base_hours = list(set(range(48)) - set(boost_hours))
            base_drinks = np.random.randint(min(3, max_drinks), max(3, max_drinks) + 1)
            drink_times = np.random.choice(base_hours, size=base_drinks, replace=False)
            for hour in drink_times:
                usage[hour] += int(np.random.uniform(80.0, 130.0)) // 10 * 10

            for hour in range(48):
                records.append({
                    'person_id': pid,
                    'date': date.strftime('%Y-%m-%d'),
                    'hour': hour // 2,
                    'minute': hour % 2 * 30,
                    'amount': usage[hour],
                    'pattern': 'D'
                })
    return pd.DataFrame(records)

def generateD(num_people=1000, num_days=30,file_path=''):
    df_D = generate_pattern_d_days(num_people,num_days)
    df_D['date'] = pd.to_datetime(df_D['date'])
    X_D = []

    unique_ids = df_D['person_id'].unique()

    for pid in unique_ids:
        person_df = df_D[df_D['person_id'] == pid].sort_values('date')

        for i in range(0, 30, 5):
            start_date = person_df['date'].min()
            temp_df = person_df[
                ((person_df['date'] - start_date).dt.days >= i) &
                ((person_df['date'] - start_date).dt.days < i + 5)
                ]

            if len(temp_df) == 48 * 5:
                temp_df = temp_df.copy()
                temp_df['slot'] = temp_df['hour'] * 2 + temp_df['minute'] // 30

                daily_vectors = []
                for day in temp_df['date'].unique():
                    slot_sum = temp_df[temp_df['date'] == day].groupby('slot')['amount'].sum().sort_index().values
                    if len(slot_sum) == 48:
                        daily_vectors.append(slot_sum)

                if len(daily_vectors) == 5:
                    avg_vector = np.mean(daily_vectors, axis=0)
                    norm_vector = avg_vector / np.max(avg_vector) if np.max(avg_vector) != 0 else avg_vector
                    X_D.append(norm_vector)
    from classifier import Autoencoder,train_autoencoder
    import torch
    from sklearn.metrics import mean_squared_error

    X_D = np.array(X_D)
    model_D = Autoencoder()
    train_autoencoder(model_D, X_D)
    model_D.eval()

    with torch.no_grad():
        outputs = model_D(torch.tensor(X_D, dtype=torch.float32)).numpy()
    mses = [mean_squared_error(x, y) for x, y in zip(X_D, outputs)]
    threshold_D = np.mean(mses) + 1.5 * np.std(mses)


    df_D = pd.DataFrame(df_D)
    df_D.to_csv(f'{file_path}/pattern_D_1.csv', index=False)

    # ⬇ D 모델 저장
    torch.save(model_D.state_dict(), f'{file_path}/autoencoder_D.pt')
    print("✅ Saved AE model for Pattern D.")

    # # ⬇ Threshold 저장 (Pickle)
    import pickle

    # 기존 thresholds.pkl 불러오기
    threshold_path = f'{file_path}/thresholds.pkl'
    with open(threshold_path, 'rb') as f:
        thresholds = pickle.load(f)

    # threshold_D 추가
    thresholds['D'] = threshold_D  # 또는 적절한 키 사용

    # 다시 저장
    with open(threshold_path, 'wb') as f:
        pickle.dump(thresholds, f)
        print(f"✅ Updated thresholds with 'D' and saved to {threshold_path}")
