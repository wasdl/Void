from app.models.classifier.autoencoder import AutoencoderClassifier
from app.config import get_paths_dict

class ClassifierService:
    """분류기 서비스 클래스"""
    def __init__(self):
        self.autoencoder = AutoencoderClassifier()
        self.paths = get_paths_dict()
        
    def train_classifiers(self, patterns=None):
        """오토인코더 분류기 학습 서비스"""
        return self.autoencoder.train_classifiers(patterns)
        
    def load_classifiers(self):
        """저장된 오토인코더 분류기 로드"""
        return self.autoencoder.load_models()
        
    def classify_pattern(self, data):
        """패턴 분류 서비스"""
        return self.autoencoder.predict_pattern(data)
