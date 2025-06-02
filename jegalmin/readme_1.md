## 1. 원본 데이터 분석 (EDA)

### 1.1 데이터 설명

분석에 사용된 데이터셋 : `water_purifier_event_history.csv`

3일치 event history 데이터

- `uuid`: 사용자 고유 ID
    - uuid 중 a, b는 한 기기에서 2명의 사용자를 구별하기 위한 uuid
- `eventTime`: 출수 시간
- `desiredcapacity`: 출수량(ml)
- `desiredtype`: 요청된 물의 온도 유형 (hotwater, ambientwater, coldwater)

### Water Usage Data

| id | uuid | eventTime | desiredCapacity | desiredType |
|----|------|-----------|-----------------|-------------|
| 0 | 1-a | 2024-02-01 0:13 | 140 | ambientwater |
| 1 | 1-a | 2024-02-01 4:05 | 1000 | ambientwater |
| 2 | 1-a | 2024-02-01 8:02 | 120 | hotwater |
| 3 | 1-a | 2024-02-01 8:29 | 120 | coldwater |
| 4 | 1-a | 2024-02-01 9:11 | 140 | hotwater |
| ... | ... | ... | ... | ... |
| 999995 | 999-b | 2024-02-03 19:44 | 260 | hotwater |
| 999996 | 999-b | 2024-02-03 19:52 | 500 | coldwater |
| 999997 | 999-b | 2024-02-03 20:24 | 120 | hotwater |
| 999998 | 999-b | 2024-02-03 20:52 | 260 | ambientwater |
| 999999 | 999-b | 2024-02-03 21:27 | 120 | ambientwater |


### DataFrame Information

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000000 entries, 0 to 999999
Data columns (total 4 columns):
 #   Column           Non-Null Count    Dtype  
---  ------           --------------    -----  
 0   uuid             1000000 non-null  object 
 1   eventTime        1000000 non-null  object 
 2   desiredcapacity  1000000 non-null  int64  
 3   desiredtype      1000000 non-null  object 
dtypes: int64(1), object(3)
memory usage: 30.5+ MB
```


### 1.2 물 용량 분석

물 용량(desiredcapacity) 파이 차트로 시각화

![alt text](<../assets/물 용량.png>)

### 1.3 물 온도 선호도 분석

물 온도(desiredtype) 선호도 파이 차트로 시각화

![alt text](<../assets/물온도.png>)


### 1.4 시간별 물 사용 패턴 분석

정수기 사용량을 시간대별로 분석하여 1시간 단위로 사용 패턴 시각화


#### 1.4.1 시간별 전체 사용 수 (1시간 단위)

![시간별 전체 사용 수 (1시간 단위).png](<../assets/시간별 전체 사용 수 (1시간 단위).png>)

```python
# 'eventTime'을 datetime 형식으로 변환
data['eventTime'] = pd.to_datetime(data['eventTime'])

# 1시간 단위로 전체 사용 수 계산
total_usage = data.set_index('eventTime').resample('1h').size()

# X축 라벨 형식 변경 (일-시 형태로 변환)
formatted_labels = [f"{d.day}-{d.hour}" for d in total_usage.index]

# 막대 그래프 생성 (전체 사용 수)
plt.figure(figsize=(12, 6))
plt.bar(formatted_labels, total_usage.values, width=0.8, color='gray')
plt.ylabel("사용 수")
plt.title("시간별 전체 사용 수 (1시간 단위)")
plt.xticks(rotation=45)
plt.show()
```

#### 1.4.2 온도별 사용 수 분석

각 온도 타입별로 시간대별 사용 패턴

##### 온수(Hot Water) 사용 패턴

![시간별 온수 사용 수](<../assets/시간별 hotwater 사용 수 (1시간 단위).png>)

##### 미온수(Ambient Water) 사용 패턴

![시간별 미온수 사용 수](<../assets/시간별 ambientwater 사용 수 (1시간 단위).png>)

##### 냉수(Cold Water) 사용 패턴

![시간별 냉수 사용 수](<../assets/시간별 coldwater 사용 수 (1시간 단위).png>)


#### 1.4.3 시간별 온도 비율 분석

특정 사용자의 시간대별 온도 선호도를 분석했습니다.

![시간별 사용 수 및 온도별 비율 - A](<../assets/시간별 사용 수 및 온도별 비율 (1시간 단위) - 1 - a.png>)
![시간별 사용 수 및 온도별 비율 - B](<../assets/시간별 사용 수 및 온도별 비율 (1시간 단위) - 1- b.png>)

#### 1.4.4 시간별 전체 사용량 (1시간 단위)



### 1.5 물 용량 및 온도 조합 분석

물 용량과 온도의 조합 분석을 통해 사용자가 선호하는 조합을 파악했습니다.
![시간별 사용 수 및 온도별 비율 - B](<../assets/물용량&물온도조합.png>)

```python
# 물 용량 기준 설정 (정렬 순서)
capacity_order = ['120', '140', '260', '500', '1000']

# 'desiredcapacity'와 'desiredtype'을 조합하여 새로운 범주 생성
data['combined'] = data['desiredcapacity'].astype(str) + " - " + data['desiredtype']

# 각 조합의 개수 세기
combined_counts = data['combined'].value_counts()

# 정렬: 물 용량 순서에 맞춰 정렬
sorted_combined_counts = sorted(
    combined_counts.items(), 
    key=lambda x: (capacity_order.index(x[0].split(" - ")[0]), x[0])
)

# 정렬된 데이터를 분리
sorted_labels = [x[0] for x in sorted_combined_counts]
sorted_values = [x[1] for x in sorted_combined_counts]

# 색상 매핑
bar_colors = [colors[label.split(" - ")[1]] for label in sorted_labels]

# 막대 그래프 (물 용량 & 물 온도 조합)
plt.figure(figsize=(12, 6))
plt.bar(sorted_labels, sorted_values, color=bar_colors)
plt.xlabel("물 용량 - 물 온도 조합")
plt.ylabel("수량")
plt.title("물 용량 & 물 온도 조합 (Bar Chart)")
plt.xticks(rotation=45)
plt.show()
```

### 1.6 사용자별 사용량 분석

사용자별 총 사용량(desiredcapacity 합계)과 온도별 사용량을 분석했습니다.



### 1.7 전체 시간별 사용량 분석

전체 및 온도별 시간대별 사용량(desiredcapacity 합계)을 분석했습니다.
![alt text](image.png)

### 1.8 사용자별 온도 선호도 분석

개별 사용자의 온도 선호도를 파이 차트로 시각화했습니다.

### 1.9 단시간 내 재요청 패턴 분석

1분 이내 정수기 재요청 시 동일한 온도를 선택하는지 분석했습니다.

![시간별 사용 수 및 온도별 비율 - B](<../assets/1분내 정수기 재요청.png>)
```python
# eventTime을 datetime 형식으로 변환
data['eventTime'] = pd.to_datetime(data['eventTime'])

# 다음 행과의 시간 차이를 분 단위로 계산
data['timediff'] = (data['eventTime'].shift(-1) - data['eventTime']).dt.total_seconds() / 60

# 그룹 번호 할당
data['group'] = 0
group_id = 0

# 그룹화 진행
for i in range(len(data)):
    data.loc[i, 'group'] = group_id
    if data.loc[i, 'timediff'] > 1:
        group_id += 1

# 그룹별 desiredtype 일관성 확인 함수
def check_group_consistency(data):
    return data.groupby('group')['desiredtype'].nunique() == 1

# 일관성 확인 및 파이 차트 생성
group_consistency = check_group_consistency(data)
true_count = group_consistency.sum()
false_count = len(group_consistency) - true_count

plt.figure(figsize=(7, 7))
plt.pie(
    [true_count, false_count], 
    labels=['일관됨 (True)', '일관되지않음 (False)'], 
    autopct='%1.1f%%', 
    colors=['green', 'red'], 
    startangle=140, 
    wedgeprops={'edgecolor': 'black'}
)
plt.title("1분내 정수기 재요청이 똑같은 온도를 하였는가?")
plt.show()
```

## 2. 주요 발견점

### 2.1 시간대별 사용 패턴

### 2.2 온도별 사용 패턴


### 2.3 출수량 분석


## 3. 결론


**추후 작성**
