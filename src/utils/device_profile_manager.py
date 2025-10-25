# -*- coding: utf-8 -*-
import os
import json
from . import paths

class DeviceProfileManager:
    """
    کلاسی برای مدیریت پروفایل‌های دستگاه.
    هر پروفایل شامل شانه، تار و پالت رنگی است.
    این پروفایل‌ها در یک فایل JSON ذخیره می‌شوند.
    """
    def __init__(self):
        self.profiles_path = os.path.join(paths.CONFIG_DIR, 'device_profiles.json')
        self.profiles = self._load_profiles()

    def _load_profiles(self):
        """
        پروفایل‌ها را از فایل JSON بارگذاری می‌کند.
        اگر فایل وجود نداشته باشد، یک دیکشنری خالی برمی‌گرداند.
        """
        try:
            if os.path.exists(self.profiles_path):
                with open(self.profiles_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading device profiles: {e}")
        return {}

    def _save_profiles(self):
        """
        پروفایل‌های فعلی را در فایل JSON ذخیره می‌کند.
        """
        try:
            with open(self.profiles_path, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving device profiles: {e}")

    def get_profile_names(self):
        """
        لیستی از نام تمام پروفایل‌های موجود را برمی‌گرداند.
        """
        return sorted(list(self.profiles.keys()))

    def get_profile(self, name):
        """
        اطلاعات یک پروفایل مشخص را بر اساس نام آن برمی‌گرداند.
        """
        return self.profiles.get(name)

    def save_profile(self, name, shaneh, tar, palette):
        """
        یک پروفایل جدید را ذخیره کرده یا یک پروفایل موجود را به‌روزرسانی می‌کند.
        """
        if not name:
            raise ValueError("نام پروفایل نمی‌تواند خالی باشد.")
            
        # --- رفع خطا: تبدیل پالت به نوع داده سازگار با JSON ---
        # مقادیر رنگی (که ممکن است از نوع numpy.uint8 باشند) به int تبدیل می‌شوند.
        converted_palette = [tuple(map(int, color)) for color in palette]

        self.profiles[name] = {
            'shaneh': shaneh,
            'tar': tar,
            'palette': converted_palette  # استفاده از پالت تبدیل‌شده
        }
        self._save_profiles()

    def delete_profile(self, name):
        """
        یک پروفایل را بر اساس نام آن حذف می‌کند.
        """
        if name in self.profiles:
            del self.profiles[name]
            self._save_profiles()
            return True
        return False