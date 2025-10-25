# -*- coding: utf-8 -*-
import os
import sys

def get_project_root():
    """
    ریشه اصلی پروژه را پیدا می‌کند.
    این تابع فرض می‌کند که این فایل در src/utils/ قرار دارد.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# مسیر ریشه پروژه
ROOT_DIR = get_project_root()

# افزودن ریشه پروژه به sys.path تا ماژول‌ها به درستی پیدا شوند
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# تعریف مسیرهای اصلی پروژه
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
SRC_DIR = os.path.join(ROOT_DIR, 'src')

# تعریف مسیر فایل‌های کلیدی
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, 'model_config.yaml')
SAM_MODEL_CHECKPOINT = os.path.join(MODELS_DIR, 'sam_vit_h_4b8939.pth')

def ensure_dirs_exist():
    """
    اطمینان حاصل می‌کند که پوشه‌های ضروری وجود دارند.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

# در زمان ایمپورت، از وجود پوشه‌ها اطمینان حاصل می‌شود
ensure_dirs_exist()