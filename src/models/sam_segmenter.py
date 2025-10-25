# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
import cv2
import os

class SAMSegmenter:
    """کلاس جداسازی عناصر تصویر با SAM با قابلیت بارگذاری تنبل (Lazy Loading)."""
    
    def __init__(self, model_type="vit_h", checkpoint_path=None, device="cuda"):
        """
        مقداردهی اولیه پارامترها بدون بارگذاری مدل.
        """
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        self.sam = None
        self.mask_generator = None
        self.predictor = None

    def _lazy_load_model(self):
        """
        مدل SAM و ابزارهای آن را تنها در صورت نیاز و در اولین فراخوانی بارگذاری می‌کند.
        """
        if self.sam is not None:
            return

        # ایمپورت‌های سنگین فقط در صورت نیاز انجام می‌شوند
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        
        print(f"🔄 در حال بارگذاری مدل SAM ({self.model_type}). این فرآیند ممکن است زمان‌بر باشد...")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"فایل مدل SAM در مسیر '{self.checkpoint_path}' یافت نشد. لطفاً آن را دانلود و در این مسیر قرار دهید.")
            
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)
        
        # ساخت ابزارهای جانبی پس از بارگذاری مدل
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        self.predictor = SamPredictor(self.sam)
        print("✅ مدل SAM با موفقیت بارگذاری شد.")
    
    def segment_automatic(self, image):
        """
        جداسازی خودکار تمام عناصر تصویر.
        ابتدا از بارگذاری مدل اطمینان حاصل می‌کند.
        """
        # --- Lazy Loading Trigger ---
        self._lazy_load_model()
        
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        print("🔍 در حال جداسازی خودکار عناصر تصویر با SAM...")
        masks = self.mask_generator.generate(image)
        print(f"✅ {len(masks)} عنصر شناسایی شد.")
        
        # مرتب‌سازی ماسک‌ها بر اساس مساحت (بزرگترین اول)
        if masks:
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def segment_with_point(self, image):
        """
        جداسازی سریع شیء مرکزی با استفاده از یک نقطه در مرکز تصویر.
        """
        # --- Lazy Loading Trigger ---
        self._lazy_load_model()

        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image

        print("⚡️ در حال جداسازی سریع شیء مرکزی با نقطه...")
        self.predictor.set_image(image_np)
        
        # نقطه مرکزی تصویر
        input_point = np.array([[image_np.shape[1] // 2, image_np.shape[0] // 2]])
        input_label = np.array([1]) # 1 برای پیش‌زمینه

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # بهترین ماسک با بالاترین امتیاز را انتخاب می‌کنیم
        best_mask = masks[np.argmax(scores)]
        print(f"✅ ماسک شیء مرکزی با امتیاز {np.max(scores):.2f} استخراج شد.")
        return best_mask

    def extract_main_object(self, image, fast_mode=False):
        """
        استخراج شیء اصلی (بزرگ‌ترین عنصر) از تصویر.
        اگر fast_mode فعال باشد، از یک نقطه در مرکز استفاده می‌کند.
        """
        if fast_mode:
            return self.segment_with_point(image)
        else:
            masks = self.segment_automatic(image)
            if masks:
                print("   - بزرگترین عنصر به عنوان شیء اصلی انتخاب شد.")
                return masks[0]['segmentation']
            print("   - هیچ عنصری برای استخراج یافت نشد.")
            return None
    
    def apply_mask_to_image(self, image, mask, background_color=(255, 255, 255)):
        """
        اعمال یک ماسک باینری به تصویر برای حذف پس‌زمینه.
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else:
            image_np = image
            
        # اطمینان از 3 کاناله بودن تصویر خروجی
        if len(image_np.shape) < 3:
             image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        # ایجاد یک پس‌زمینه با رنگ مشخص
        background = np.full(image_np.shape, background_color, dtype=np.uint8)
        
        # اطمینان از اینکه ماسک به صورت 3 بعدی برای broadcast صحیح است
        mask_3d = np.stack([mask]*3, axis=-1)
        
        # ترکیب پیش‌زمینه و پس‌زمینه بر اساس ماسک
        result = np.where(mask_3d > 0, image_np, background)
        
        return Image.fromarray(result)