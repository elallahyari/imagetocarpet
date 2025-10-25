# -*- coding: utf-8 -*-
import os
import subprocess
import shutil
import numpy as np
from PIL import Image

class Vectorizer:
    """
    کلاس تبدیل تصاویر رستر به وکتور (SVG, PDF).
    این کلاس به ابزارهای خط فرمان خارجی مانند vtracer یا potrace نیاز دارد.
    """
    
    def __init__(self, method='vtracer'):
        """
        سازنده کلاس.
        در همان ابتدا وجود ابزار مورد نیاز را بررسی می‌کند.
        Args:
            method (str): متد وکتورسازی ('vtracer' یا 'potrace').
        Raises:
            EnvironmentError: اگر ابزار مورد نیاز در PATH سیستم یافت نشود.
        """
        self.method = method
        self.executable_path = shutil.which(self.method)
        
        if not self.executable_path:
            error_message = (
                f"ابزار '{self.method}' در سیستم شما یافت نشد.\n"
                "لطفاً آن را نصب کرده و در مسیر (PATH) سیستم قرار دهید.\n"
                "راهنمای نصب:\n"
                " - vtracer: https://github.com/visioncortex/vtracer (پیشنهادی برای تصاویر رنگی)\n"
                " - potrace: http://potrace.sourceforge.net/ (برای تصاویر سیاه و سفید)"
            )
            raise EnvironmentError(error_message)
        
        print(f"✅ ابزار وکتورسازی '{self.method}' در مسیر '{self.executable_path}' یافت شد.")
        
    def vectorize_vtracer(self, image, output_path, **kwargs):
        """وکتوری‌سازی با vtracer با قابلیت تنظیم پارامترها."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        temp_png = output_path.replace('.svg', '_temp.png')
        image.save(temp_png)
        
        cmd = [
            self.executable_path,
            '--input', temp_png,
            '--output', output_path,
            '--colormode', 'color',
            '--hierarchical', 'stacked',
            # --- پارامترهای قابل تنظیم ---
            '--filter_speckle', str(kwargs.get('filter_speckle', 4)),
            '--color_precision', str(kwargs.get('color_precision', 6)),
            '--corner_threshold', str(kwargs.get('corner_threshold', 60)),
        ]
        
        print(f"   - اجرای دستور vtracer با پارامترها: {kwargs}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            print(f"   - vtracer output: {result.stdout}")
            return output_path
        finally:
            if os.path.exists(temp_png):
                os.remove(temp_png)

    def vectorize(self, image, output_path, **kwargs):
        """
        اجرای وکتورسازی با متد انتخاب شده در سازنده کلاس.
        """
        print(f"🔄 در حال وکتورسازی تصویر با استفاده از {self.method}...")
        try:
            if self.method == 'vtracer':
                return self.vectorize_vtracer(image, output_path, **kwargs)
            else:
                raise ValueError(f"متد وکتورسازی '{self.method}' پشتیبانی نمی‌شود.")
        except subprocess.CalledProcessError as e:
            error_details = f"اجرای دستور {self.method} با خطا مواجه شد.\n" \
                            f"Stderr: {e.stderr}\nStdout: {e.stdout}"
            print(f"❌ {error_details}")
            raise RuntimeError(error_details) from e
        except Exception as e:
            print(f"❌ یک خطای غیرمنتظره در حین وکتورسازی رخ داد: {e}")
            raise e

    def svg_to_pdf(self, svg_path, output_path):
        """
        تبدیل فایل SVG به PDF با استفاده از کتابخانه cairosvg.
        """
        try:
            import cairosvg
            print(f"   - در حال تبدیل SVG به PDF...")
            cairosvg.svg2pdf(url=svg_path, write_to=output_path)
            print(f"   - فایل PDF با موفقیت در '{output_path}' ذخیره شد.")
            return output_path
        except ImportError:
            msg = "کتابخانه 'cairosvg' نصب نیست. لطفاً با `pip install cairosvg` آن را نصب کنید."
            print(f"❌ {msg}")
            raise ImportError(msg)
        except Exception as e:
            msg = f"خطا در تبدیل SVG به PDF: {e}"
            print(f"❌ {msg}")
            raise RuntimeError(msg) from e