# 이 파일은 데이터(A,B,C) 이상치 제거 파일입니다.
# 결과는 https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from dtaidistance import dtw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- 전처리 함수들 -------------------

def cumulative_minmax(vec):
    cum = np.cumsum(vec)
    vmin, vmax = cum.min(), cum.max()
    return (cum - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(cum)

def normalize_each_day_minmax(vectors):
    normalized = []
    for vec in vectors:
        vmin, vmax = np.min(vec), np.max(vec)
        normalized.append((vec - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vec))
    return np.array(normalized)

# ------------------- 메인 함수 -------------------

def process_pattern_dbscan(df,file_path, label=''):
    df['date'] = pd.to_datetime(df['date'])

    # 하루 단위 벡터 생성
    daily_vectors = []
    for (pid, date), group in df.groupby(['person_id', 'date']):
        sorted_group = group.sort_values(by=['hour', 'minute'])
        vec = sorted_group['amount'].values
        if len(vec) == 48:
            daily_vectors.append(vec)
    daily_vectors = np.array(daily_vectors)

    if len(daily_vectors) == 0:
        print(f"[{label}] No valid daily vectors.")
        return

    # 정규화
    daily_vectors_norm = normalize_each_day_minmax(daily_vectors)

    # DBSCAN 군집화
    dbscan = DBSCAN(eps=0.9, min_samples=3)

    labels = dbscan.fit_predict(daily_vectors_norm)

    # PCA 시각화를 위한 변환
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(daily_vectors_norm)

    # 유효 벡터만 (노이즈 제외)
    valid_indices = labels != -1
    valid_vectors = daily_vectors_norm[valid_indices]
    valid_labels = labels[valid_indices]

    if len(valid_vectors) == 0:
        print(f"[{label}] No clusters found.")
        return

    # ------------------- (1) Silhouette Score 계산 -------------------
    sil_samples = silhouette_samples(valid_vectors, valid_labels)
    sil_scores = {
        cluster_id: np.mean(sil_samples[valid_labels == cluster_id])
        for cluster_id in np.unique(valid_labels)
    }

    # 표로 출력
    sil_table = pd.DataFrame({
        'Cluster': list(sil_scores.keys()),
        'Silhouette Score': list(sil_scores.values())
    }).sort_values(by='Silhouette Score', ascending=False)

    print(f"\n[{label}] Cluster-wise Silhouette Scores:")

    # ------------------- (2) 가장 응집도 높은 클러스터 → 대표 벡터 -------------------
    best_cluster = sil_table.iloc[0]['Cluster']
    best_vectors = valid_vectors[valid_labels == best_cluster]

    # 가장 대표적인 실제 벡터 찾기
    dtw_matrix = np.zeros((len(best_vectors), len(best_vectors)))
    for i in range(len(best_vectors)):
        for j in range(i + 1, len(best_vectors)):
            dist = dtw.distance(best_vectors[i], best_vectors[j])
            dtw_matrix[i, j] = dist
            dtw_matrix[j, i] = dist
    avg_dtw = dtw_matrix.mean(axis=1)
    center_idx = np.argmin(avg_dtw)
    center_vector = best_vectors[center_idx]

    # 전체 데이터에 대한 DTW 계산
    cum_center = cumulative_minmax(center_vector)
    dtw_cum_distances = [dtw.distance(cum_center, cumulative_minmax(vec)) for vec in daily_vectors_norm]
    dtw_plain_distances = [dtw.distance(center_vector, vec) for vec in daily_vectors_norm]

    # ------------------- (3) PCA 시각화 -------------------
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = labels == lab
        label_name = f'Cluster {lab}' if lab != -1 else 'Noise'
        color = 'red' if lab == best_cluster else None
        plt.scatter(pca_result[idx, 0], pca_result[idx, 1], label=label_name, alpha=0.5, c=color)

    plt.title(f"[{label}] PCA of DBSCAN Clusters (Best Cluster Highlighted)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------- (4) 대표 벡터 시계열 -------------------
    plt.figure(figsize=(8, 3))
    plt.plot(center_vector, label='Representative Pattern', linewidth=2)
    plt.title(f"[{label}] Most Representative Pattern")
    plt.xlabel("Time Slot")
    plt.ylabel("Normalized Amount")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------- (5) DTW 히스토그램 -------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(dtw_cum_distances, bins=40, color='orange')
    plt.title(f"[{label}] DTW (Cumulative) to Representative (ALL Data)")
    plt.xlabel("DTW Distance")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(dtw_plain_distances, bins=40, color='skyblue')
    plt.title(f"[{label}] DTW (Plain) to Representative (ALL Data)")
    plt.xlabel("DTW Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    # ------------------- 이상치 제거 (IQR 방식) -------------------
    q1 = np.percentile(dtw_cum_distances, 25)
    q3 = np.percentile(dtw_cum_distances, 75)
    iqr = q3 - q1
    upper_bound = q3 + 0.5 * iqr

    # 이상치 마스크 및 인덱스 추출
    outlier_mask = np.array(dtw_cum_distances) > upper_bound

    # ------------------- 이상치 제거 (상위 5% 제거 방식) -------------------

    # threshold = np.percentile(dtw_cum_distances, 95)
    # outlier_mask = np.array(dtw_cum_distances) > threshold

    # ------------------- (3-2) 전체 점 + DTW 이상치 시각화 -------------------
    outlier_indices = np.where(outlier_mask)[0]

    plt.figure(figsize=(8, 6))

    # ① 전체 점 (연한 파랑)
    plt.scatter(pca_result[:, 0], pca_result[:, 1],
                alpha=0.3, color='skyblue', label='All Data')

    # ② 이상치 (선명한 빨강 + X 마커)
    if len(outlier_indices) > 0:
        plt.scatter(pca_result[outlier_indices, 0], pca_result[outlier_indices, 1],
                    color='red', marker='x', s=70, label='DTW Outliers (IQR)')

    plt.title(f"[{label}] PCA with DTW Outliers Highlighted")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    non_outlier_indices = np.where(~outlier_mask)[0]

    print(f"[{label}] IQR 기준 이상치 개수: {outlier_mask.sum()}개 / 전체 {len(dtw_cum_distances)}개")

    # 이상치 제외한 데이터 추출
    filtered_vectors = daily_vectors[non_outlier_indices]  # 원본 벡터 사용 (정규화 X)
    person_day_info = []

    # 각 벡터가 어떤 사람/날짜에 해당하는지 매칭
    idx = 0
    for (pid, date), group in df.groupby(['person_id', 'date']):
        if len(group) == 48:
            if idx in non_outlier_indices:
                start_hours = group['start_hours'].iloc[0]
                eating_hours = group['eating_hours'].iloc[0]
                counts = group['counts'].iloc[0]
                person_day_info.append((pid, date, start_hours, eating_hours, counts))
            idx += 1

    # 저장용 데이터프레임 생성
    records = []
    for (pid, date, start_hours, eating_hours, counts), vec in zip(person_day_info, filtered_vectors):
        for i, amount in enumerate(vec):
            hour = (i * 30) // 60
            minute = (i * 30) % 60
            records.append({
                'person_id': pid,
                'date': date,
                'hour': hour,
                'minute': minute,
                'amount': amount,
                'start_hours': start_hours,
                'eating_hours': eating_hours,
                'counts': counts
            })

    df_filtered = pd.DataFrame(records)
    df_filtered.to_csv(f"{file_path}/filtered_pattern_{label}.csv", index=False)
    print(f"[{label}] 이상치 제거된 데이터 저장 완료 → filtered_pattern_{label}.csv")



