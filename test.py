# -*- coding: utf-8 -*-
import os
import sys
import shutil

# اضافه کردن مسیر پروژه به sys.path از طریق ماژول متمرکز
# این کار باید قبل از ایمپورت‌های دیگر پروژه انجام شود
try:
    from src.utils import paths
except ImportError:
    # اگر ماژول یافت نشد، به روش قدیمی مسیر را اضافه می‌کنیم
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.utils import paths


def test_installation():
    """
    نصب و صحت فایل‌های پروژه را به طور کامل بررسی می‌کند.
    این اسکریپت وجود پایتون، کتابخانه‌ها، مدل‌ها و ابزارهای خارجی را چک می‌کند.
    """
    
    print("=" * 60)
    print("🔬 شروع بررسی پیش‌نیازهای پروژه طرح فرش")
    print("=" * 60)
    
    all_ok = True

    # ۱. بررسی نسخه پایتون
    print(f"\n🐍 ۱. بررسی پایتون...")
    print(f"   - نسخه پایتون: {sys.version}")
    if sys.version_info < (3, 8):
        print("   - ❌ هشدار: نسخه پایتون شما قدیمی است. پیشنهاد می‌شود از پایتون ۳.۸ یا بالاتر استفاده کنید.")
        all_ok = False
    else:
        print("   - ✅ نسخه پایتون مناسب است.")

    # ۲. بررسی PyTorch و CUDA
    print(f"\n🔥 ۲. بررسی PyTorch و CUDA...")
    try:
        import torch
        print(f"   - نسخه PyTorch: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   - قابلیت CUDA فعال است؟ {'✅ بله' if cuda_available else '⚠️ خیر (پردازش روی CPU انجام خواهد شد)'}")
        if cuda_available:
            print(f"   - نام GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   - ❌ بحرانی: PyTorch نصب نیست! لطفاً `requirements.txt` را نصب کنید.")
        return False

    # ۳. بررسی کتابخانه‌های اصلی
    print("\n📚 ۳. بررسی سایر کتابخانه‌ها...")
    required_packages = {
        'transformers': 'transformers',
        'diffusers': 'diffusers',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'cairosvg': 'CairoSVG',
        'yaml': 'PyYAML' # yaml برای خواندن کانفیگ استفاده می‌شود
    }
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"   - ✅ {package_name}")
        except ImportError:
            print(f"   - ❌ {package_name} نصب نیست. لطفاً `requirements.txt` را نصب کنید.")
            all_ok = False
    
    # ۴. بررسی مدل SAM
    print("\n🤖 ۴. بررسی فایل مدل SAM...")
    sam_path = paths.SAM_MODEL_CHECKPOINT
    if os.path.exists(sam_path):
        size_mb = os.path.getsize(sam_path) / (1024 * 1024)
        print(f"   - ✅ فایل مدل SAM یافت شد ({size_mb:.2f} مگابایت).")
        if size_mb < 2400:
            print(f"   - ⚠️ هشدار: حجم فایل مدل SAM کمتر از حد انتظار است (~2560 MB).")
            print(f"      ممکن است دانلود ناقص باشد. توصیه می‌شود فایل را دوباره دانلود کنید.")
        else:
            print(f"   - ✅ حجم فایل صحیح به نظر می‌رسد.")
    else:
        print(f"   - ❌ فایل مدل SAM در مسیر '{sam_path}' یافت نشد.")
        print(f"      لطفاً فایل sam_vit_h_4b8939.pth را از آدرس زیر دانلود کنید:")
        print(f"      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print(f"      و آن را در پوشه `models` قرار دهید.")
        all_ok = False

    # ۵. بررسی ساختار پروژه
    print("\n📁 ۵. بررسی ساختار پوشه‌ها و فایل‌ها...")
    required_dirs = [paths.SRC_DIR, paths.CONFIG_DIR, paths.MODELS_DIR]
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"   - ✅ پوشه: {os.path.basename(dir_path)}/")
        else:
            print(f"   - ❌ پوشه ضروری '{os.path.basename(dir_path)}/' یافت نشد.")
            all_ok = False
            
    required_files = ['main.py', 'gui_improved.py', paths.DEFAULT_CONFIG_PATH, 'requirements.txt']
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   - ✅ فایل: {os.path.relpath(file_path, paths.ROOT_DIR)}")
        else:
            print(f"   - ❌ فایل ضروری '{os.path.relpath(file_path, paths.ROOT_DIR)}' یافت نشد.")
            all_ok = False
            
    # ۶. بررسی وابستگی‌های خارجی (برای وکتورسازی)
    print("\n✒️  ۶. بررسی ابزارهای خارجی (اختیاری اما مهم)...")
    vectorizers = ['vtracer', 'potrace']
    vectorizer_found = False
    for vz in vectorizers:
        if shutil.which(vz):
            print(f"   - ✅ ابزار '{vz}' یافت شد.")
            vectorizer_found = True
        else:
            print(f"   - ⚠️ ابزار '{vz}' یافت نشد. قابلیت وکتورسازی با این متد غیرفعال خواهد بود.")
    if not vectorizer_found:
        print("   - ❌ هیچ ابزار وکتورسازی یافت نشد. قابلیت وکتورسازی به طور کامل غیرفعال است.")
        # این مرحله را اختیاری در نظر می‌گیریم

    # نتیجه نهایی
    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 بررسی با موفقیت به پایان رسید. تمام پیش‌نیازهای اصلی نصب هستند.")
        print("\nبرای شروع برنامه:")
        print("  - رابط گرافیکی: python gui_improved.py")
        print("  - خط فرمان:    python main.py --input <path_to_image>")
    else:
        print("❌ بررسی ناموفق بود! لطفاً خطاها و هشدارهای بالا را برطرف کنید.")
    print("=" * 60)
    
    return all_ok

if __name__ == '__main__':
    try:
        success = test_installation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n\n🚨 یک خطای غیرمنتظره در حین اجرای اسکریپت تست رخ داد: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)