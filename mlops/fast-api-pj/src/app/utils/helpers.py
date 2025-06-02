"""
DQM 파이프라인에서 사용되는 공통 유틸리티 함수
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from app.config import get_paths_dict

def ensure_dir(directory):
    """디렉토리가 없으면 생성"""
    os.makedirs(directory, exist_ok=True)

def save_model_results(results, file_path, format='pickle'):
    """모델 평가 결과 저장"""
    ensure_dir(os.path.dirname(file_path))
    
    if format == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return True

def load_model_results(file_path, format='pickle'):
    """모델 평가 결과 로드"""
    if not os.path.exists(file_path):
        return None
        
    if format == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def calculate_model_accuracy(y_true, y_pred, tolerance=0.1):
    """
    DQM 기준에 따른 정확도 계산: |실제값-예측값|/실제값 < 0.1 비율
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 0으로 나누기 방지
    mask = y_true != 0
    
    if not np.any(mask):
        return 0.0
        
    # 상대 오차 계산
    relative_errors = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    
    # 허용 오차 내 예측 비율
    accuracy = np.mean(relative_errors < tolerance) * 100
    
    return accuracy

def normalize_vector(vec):
    """벡터 정규화 (0-1 범위)"""
    min_val = np.min(vec)
    max_val = np.max(vec)
    
    if max_val > min_val:
        return (vec - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(vec)

def plot_confusion_matrix(cm, labels, save_path=None):
    """혼동 행렬 시각화 및 저장"""
    from sklearn.metrics import ConfusionMatrixDisplay
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close()
        return save_path
    else:
        plt.show()
        return None

def get_timestamp():
    """현재 시간 타임스탬프 (파일명용)"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
