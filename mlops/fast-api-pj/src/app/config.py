from pathlib import Path
import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 프로젝트 루트 경로 (fast-api-pj 디렉토리)
ROOT_DIR = Path(__file__).parent.parent.parent

# 앱 디렉토리 경로 (src/app)
APP_DIR = Path(__file__).parent

# 디렉토리 생성 유틸리티 함수
def ensure_dir(directory):
    """디렉토리가 없으면 생성"""
    os.makedirs(directory, exist_ok=True)

# 데이터 경로 (수정된 폴더 구조에 맞게 변경)
RESOURCES_DIR = APP_DIR / 'resources'
DATA_DIR = RESOURCES_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = RESOURCES_DIR / 'models'

# 정적 파일 및 시각화 경로
STATIC_DIR = APP_DIR / 'static'
VISUALIZATION_PATH = STATIC_DIR / 'images'  # 시각화 이미지 저장 경로

# 필요한 디렉토리 자동 생성
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, VISUALIZATION_PATH]:
    ensure_dir(directory)

# 데이터 관련 설정
RANDOM_SEED = 42    # 랜덤 시드

# 시각화 관련 설정
DPI = 300  # 그래프 해상도

# 경로 사전 반환 함수
def get_paths_dict():
    """경로 사전 반환"""
    return {
        'root': ROOT_DIR,
        'app': APP_DIR,
        'resources': RESOURCES_DIR,
        'data': DATA_DIR,
        'raw': RAW_DATA_DIR,
        'processed': PROCESSED_DATA_DIR,
        'models': MODELS_DIR,
        'static': STATIC_DIR,
        'static_images': VISUALIZATION_PATH
    }

# 폰트 설정
def set_font():
    # 운영체제별 기본 폰트 설정
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux 등 다른 환경
        # 나눔고딕 등 한글 폰트가 있는지 확인
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        
        # 이용 가능한 한글 폰트 탐색
        for font_path in font_list:
            font = fm.FontProperties(fname=font_path).get_name()
            if any(keyword in font.lower() for keyword in ['gothic', 'gulim', 'batang', 'malgun']):
                plt.rcParams['font.family'] = font
                break
    try:
        plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지
    except:
        pass
