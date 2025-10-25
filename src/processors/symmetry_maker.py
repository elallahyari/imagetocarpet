# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageOps

class SymmetryMaker:
    """کلاس ایجاد تقارن و الگوهای تکرارشونده برای طرح فرش."""
    
    def __init__(self):
        pass

    def create_mirror_horizontal(self, image):
        """ایجاد آینه‌ای افقی از نیمه چپ تصویر."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        width, height = image.size
        left_half = image.crop((0, 0, width // 2, height))
        right_half_mirrored = ImageOps.mirror(left_half)
        
        result = Image.new('RGB', (width, height))
        result.paste(left_half, (0, 0))
        result.paste(right_half_mirrored, (width // 2, 0))
        
        return result

    def create_four_way_mirror(self, image):
        """
        ایجاد تقارن چهار طرفه از ربع بالا-چپ تصویر.
        این عمل برای ساخت مدالیون‌های مرکزی فرش بسیار متداول است.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        width, height = image.size
        q_width, q_height = width // 2, height // 2
        
        # برش ربع بالا-چپ
        top_left = image.crop((0, 0, q_width, q_height))
        
        # ایجاد سایر ربع‌ها با استفاده از عملیات آینه‌ای
        top_right = ImageOps.mirror(top_left)
        bottom_left = ImageOps.flip(top_left)
        bottom_right = ImageOps.mirror(bottom_left)
        
        # چسباندن ۴ ربع در کنار هم برای ساخت تصویر کامل
        result = Image.new('RGB', (width, height))
        result.paste(top_left, (0, 0))
        result.paste(top_right, (q_width, 0))
        result.paste(bottom_left, (0, q_height))
        result.paste(bottom_right, (q_width, q_height))
        
        return result

    def create_medallion_layout(self, center_element, canvas_size=(2048, 2048), background_color=(245, 240, 230)):
        """
        ایجاد چیدمان کلاسیک فرش با یک مدالیون در مرکز.
        
        Args:
            center_element (PIL.Image): المان مرکزی (معمولاً با تقارن چهارطرفه).
            canvas_size (tuple): اندازه کل طرح فرش (width, height).
            background_color (tuple): رنگ پس‌زمینه فرش به صورت (R, G, B).
            
        Returns:
            PIL.Image: طرح کامل فرش با مدالیون مرکزی.
        """
        width, height = canvas_size
        
        # ایجاد پس‌زمینه با رنگ مشخص شده
        result = Image.new('RGB', canvas_size, background_color)
        
        if isinstance(center_element, np.ndarray):
            center_element = Image.fromarray(center_element)
        
        # تغییر اندازه مدالیون به نصف ابعاد کوچکتر کانvas
        medallion_size = min(width, height) // 2
        medallion = center_element.resize((medallion_size, medallion_size), Image.LANCZOS)
        
        # محاسبه موقعیت مرکز برای چسباندن مدالیون
        center_x = (width - medallion_size) // 2
        center_y = (height - medallion_size) // 2
        
        # استفاده از ماسک آلفا در صورت وجود برای چسباندن نرم
        paste_mask = medallion if medallion.mode == 'RGBA' else None
        result.paste(medallion, (center_x, center_y), mask=paste_mask)
        
        return result