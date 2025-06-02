# https://colab.research.google.com/drive/1UTG0jsTU6SMeLqPHQ0GZybEAuCiNk2LI?usp=sharing


import os

# CSV 저장
os.chdir("/content/drive/MyDrive/Colab Notebooks/VoID_WaterPurifier")
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim=48, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(model, data, epochs=150, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    inputs = torch.tensor(data, dtype=torch.float32)

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

def classifier(dfA,dfB,dfC,file_path,size=1000,day_range=30):
    # 원본 1일치 파일 불러오기 (counts 포함되어 있는 버전)
    df = pd.concat([dfA, dfB, dfC], ignore_index=True)

    X_5day = []
    y_5day = []

    # 패턴별 1000명 추출
    sampled_ids = (
        df.groupby('pattern')['person_id']
        .unique()
        .apply(lambda x: np.random.choice(x,  size, replace=False))
    )

    start_date = datetime(2022, 2, 1)
    df['date'] = pd.to_datetime(df['date'])
    # 패턴별 유저 1000명씩만 처리
    for pattern in ['A', 'B', 'C']:
        for person_id in sampled_ids[pattern]:
            person_df = df[df['person_id'] == person_id].sort_values('date')

            for i in range(0, day_range, 5):  # 0~4, 5~9, ..., 25~29
                day_range = (i, i + 5)
                temp_df = person_df[
                    ((person_df['date'] - start_date).dt.days >= day_range[0]) &
                    ((person_df['date'] - start_date).dt.days < day_range[1])
                    ]

                if len(temp_df) == 48 * 5:
                    temp_df = temp_df.copy()
                    temp_df['slot'] = temp_df['hour'] * 2 + temp_df['minute'] // 30

                    slot_sum = temp_df.groupby('slot')['amount'].sum().sort_index().values

                    if len(slot_sum) == 48:
                        norm_slot_sum = slot_sum / np.max(slot_sum) if np.max(slot_sum) != 0 else slot_sum
                        X_5day.append(norm_slot_sum)
                        y_5day.append(pattern)

    X_5day = np.array(X_5day)
    y_5day = np.array(y_5day)




    patterns = ['A', 'B', 'C']
    models = {}
    thresholds = {}

    for pattern in patterns:
        print(f"Training AE for Pattern {pattern}")
        data = X_5day[y_5day == pattern]
        model = Autoencoder()
        train_autoencoder(model, data)
        models[pattern] = model

        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(data, dtype=torch.float32)).numpy()
        mses = [mean_squared_error(x, y) for x, y in zip(data, outputs)]
        thresholds[pattern] = np.mean(mses) + 1.5 * np.std(mses) # 1.5 87%, 2 90%

    # 4. Prediction function
    def predict(sample):
        errors = {}
        for p, model in models.items():
            model.eval()
            with torch.no_grad():
                x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
                recon = model(x).squeeze(0).numpy()
                mse = mean_squared_error(sample, recon)
                errors[p] = (mse, recon)

        best_p = min(errors, key=lambda k: errors[k][0])
        best_mse = errors[best_p][0]

        if best_mse > thresholds[best_p]:
            return "Unknown", errors[best_p][1], best_mse
        else:
            return best_p, errors[best_p][1], best_mse


    predicted_labels = []
    reconstruction_errors = []

    for sample in X_5day:
        pred, recon, mse = predict(sample)
        predicted_labels.append(pred)
        reconstruction_errors.append(mse)

    predicted_labels = np.array(predicted_labels)

    from collections import Counter

    print("\n📊 패턴별 예측 결과:")
    for pattern in patterns:
        idxs = np.where(y_5day == pattern)[0]
        total = len(idxs)
        pred_counts = Counter(predicted_labels[idxs])
        unknown_count = pred_counts.get('Unknown', 0)
        correct_count = pred_counts.get(pattern, 0)
        acc = correct_count / total
        unknown_ratio = unknown_count / total

        print(f"Pattern {pattern} ▶ Accuracy: {acc:.2%}, Unknown: {unknown_ratio:.2%}")


    # Unknown 제외하고 평가
    valid_mask = predicted_labels != 'Unknown'
    cm = confusion_matrix(y_5day[valid_mask], predicted_labels[valid_mask], labels=patterns)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=patterns)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (excluding Unknown)")
    plt.grid(False)
    plt.show()

    import pickle

    for pattern, model in models.items():
        model_path = f'{file_path}/autoencoder_{pattern}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved AE for Pattern {pattern} to {model_path}")

    threshold_path = f'{file_path}/thresholds.pkl'
    with open(threshold_path, 'wb') as f:
        pickle.dump(thresholds, f)
        print(f"✅ Saved thresholds to {threshold_path}")
    return models, thresholds