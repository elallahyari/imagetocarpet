# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from PIL import Image

class EdgeDetector:
    """کلاس تشخیص لبه‌ها و خطوط با قابلیت بارگذاری تنبل (Lazy Loading)."""
    
    def __init__(self, method="HED", device="cuda"):
        """
        مقداردهی اولیه پارامترها بدون بارگذاری مدل.
        """
        self.method = method
        self.device = device
        self.detector = None # مدل در اینجا None است

    def _lazy_load_detector(self):
        """
        مدل تشخیص لبه را تنها در صورت نیاز و در اولین فراخوانی بارگذاری می‌کند.
        """
        if self.detector is not None:
            return

        # این ایمپورت سنگین فقط در صورت نیاز انجام می‌شود
        from controlnet_aux import HEDdetector, PidiNetDetector
        
        print(f"🔄 در حال بارگذاری مدل تشخیص لبه ({self.method})...")
        
        if self.method == "HED":
            self.detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
        elif self.method == "PiDiNet":
            self.detector = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        elif self.method == "Canny":
            # Canny نیازی به مدل ندارد
            self.detector = self._detect_edges_canny_internal
        else:
            raise ValueError(f"روش تشخیص لبه نامعتبر است: {self.method}")
            
        # انتقال به دستگاه فقط در صورتی که مدل واقعی باشد
        if hasattr(self.detector, 'to'):
            self.detector.to(self.device)
        
        print(f"✅ مدل تشخیص لبه ({self.method}) آماده استفاده است.")

    def _detect_edges_canny_internal(self, image_np, low_threshold=50, high_threshold=150):
        """پیاده‌سازی داخلی Canny برای سازگاری با ساختار کلاس."""
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return Image.fromarray(edges)

    def detect_edges(self, image, **kwargs):
        """
        تشخیص لبه با روش انتخاب شده. ابتدا از بارگذاری مدل اطمینان حاصل می‌کند.
        """
        # --- Lazy Loading Trigger ---
        self._lazy_load_detector()

        print(f"🔍 تشخیص لبه‌ها با روش {self.method}...")
        
        if isinstance(image, np.ndarray):
            # مدل‌های controlnet_aux به PIL Image نیاز دارند
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        if self.method in ["HED", "PiDiNet"]:
            if self.method == "PiDiNet":
                return self.detector(image_pil, safe=kwargs.get('safe', True))
            else:
                return self.detector(image_pil)
        elif self.method == "Canny":
            # Canny روی numpy array کار می‌کند
            image_np = np.array(image_pil)
            return self.detector(image_np, **kwargs)
        else:
            raise ValueError(f"روش نامعتبر: {self.method}")

    def refine_edges(self, edges, kernel_size=3, iterations=1):
        """
        بهبود و پالایش لبه‌ها با استفاده از عملیات مورفولوژی.
        """
        if isinstance(edges, Image.Image):
            edges = np.array(edges)
        
        if len(edges.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
        
        _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # بستن شکاف‌های کوچک
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # حذف نویز و نقاط کوچک
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        return opened