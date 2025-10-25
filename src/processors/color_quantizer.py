# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class ColorQuantizer:
    """کلاس کاهش و کوانتیزه کردن رنگ‌های تصویر با متدهای مختلف."""
    
    def __init__(self, n_colors=10):
        self.n_colors = n_colors
        self.palette = None

    def quantize_with_dithering(self, image):
        """
        کاهش رنگ با استفاده از دیترینگ برای حفظ حداکثری بافت و جزئیات.
        این متد برای طرح‌های فرش بسیار مناسب است.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # استفاده از متد داخلی و بهینه کتابخانه PIL
        # MEDIANCUT یک الگوریتم خوب برای انتخاب پالت است
        # FLOYDSTEINBERG بهترین الگوریتم دیترینگ برای حفظ جزئیات است
        dithered_image = image.quantize(
            colors=self.n_colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG
        )
        
        # تصویر کوانتیزه شده دارای پالت است، آن را به RGB تبدیل می‌کنیم
        dithered_image = dithered_image.convert("RGB")
        
        # استخراج پالت از تصویر خروجی
        palette_raw = dithered_image.getpalette()
        palette = [palette_raw[i:i+3] for i in range(0, self.n_colors * 3, 3)]
        self.palette = np.array(palette, dtype=np.uint8)
        
        return dithered_image, self.palette
    
    def apply_palette_with_dithering(self, image, custom_palette):
        """
        اعمال یک پالت سفارشی به تصویر با استفاده از دیترینگ.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # ساخت یک تصویر پالت پایه برای اعمال به تصویر اصلی
        palette_img = Image.new("P", (1, 1))
        # تبدیل پالت به فرمت مورد نیاز PIL (لیست تخت)
        palette_flat = [value for color in custom_palette for value in color]
        palette_img.putpalette(palette_flat)
        
        # اعمال پالت با دیترینگ
        dithered_image = image.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
        dithered_image = dithered_image.convert("RGB")
        
        self.palette = np.array(custom_palette, dtype=np.uint8)
        return dithered_image, self.palette

    def extract_palette(self, image, max_samples=20000):
        """استخراج پالت با K-Means (برای استخراج رنگ از تصویر اولیه مناسب است)."""
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        pixels = image.reshape(-1, 3)
        
        if len(pixels) > max_samples:
            pixels_sample = shuffle(pixels, random_state=42, n_samples=max_samples)
        else:
            pixels_sample = pixels
        
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_sample)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        return palette

    def create_palette_visualization(self, palette=None, size=(600, 80)):
        if palette is None:
            palette = self.palette
        
        if palette is None or len(palette) == 0:
            img = Image.new('RGB', size, 'white')
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            draw.text((10, 10), "پالت رنگی برای نمایش وجود ندارد.", fill="black", font=font)
            return img

        n_colors = len(palette)
        width, height = size
        color_width = width // n_colors
        
        palette_img_np = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i, color in enumerate(palette):
            start_col = i * color_width
            end_col = (i + 1) * color_width if i < n_colors - 1 else width
            palette_img_np[:, start_col:end_col] = color
        
        return Image.fromarray(palette_img_np)