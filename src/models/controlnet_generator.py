import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image

class ControlNetGenerator:
    """کلاس تولید طرح فرش با ControlNet"""
    
    def __init__(self, base_model, controlnet_model, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        print(f"🔄 در حال بارگذاری ControlNet...")
        print(f"   Base Model: {base_model}")
        print(f"   ControlNet: {controlnet_model}")
        
        # بارگذاری ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=dtype
        )
        
        # بارگذاری پایپلاین
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=dtype,
            safety_checker=None
        )
        
        # بهینه‌سازی
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # بهینه‌سازی‌های اختیاری با مدیریت خطا
        try:
            self.pipe.enable_model_cpu_offload()
            print("   - بهینه‌سازی Model CPU Offload فعال شد.")
        except Exception as e:
            print(f"   - ⚠️ امکان فعال‌سازی Model CPU Offload وجود ندارد: {e}")

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("   - بهینه‌سازی xFormers Memory Efficient Attention فعال شد.")
        except ImportError:
            print("   - ⚠️ کتابخانه xformers نصب نیست. بهینه‌سازی حافظه مربوطه غیرفعال است.")
        except Exception as e:
            print(f"   - ⚠️ امکان فعال‌سازی xFormers وجود ندارد: {e}")

        self.pipe = self.pipe.to(device)
        
        print("✅ ControlNet بارگذاری شد")
    
    def generate(
        self,
        control_image,
        prompt,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        seed=None,
        num_images=1,
        width=None,
        height=None
    ):
        """
        تولید طرح فرش
        
        Args:
            control_image: تصویر کنترل (لبه‌ها)
            prompt: پرامپت مثبت
            negative_prompt: پرامپت منفی
            num_inference_steps: تعداد مراحل
            guidance_scale: مقیاس راهنمایی
            controlnet_conditioning_scale: شدت کنترل
            seed: seed تصادفی
            num_images: تعداد تصاویر تولیدی
            width: عرض خروجی
            height: ارتفاع خروجی
            
        Returns:
            list: لیست تصاویر تولید شده
        """
        if isinstance(control_image, np.ndarray):
            control_image = Image.fromarray(control_image)
        
        # تنظیم seed
        if seed is not None and seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # تنظیم ابعاد
        if width is None:
            width = control_image.width
        if height is None:
            height = control_image.height
        
        # گرد کردن به مضرب 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        print(f"🎨 تولید طرح فرش...")
        print(f"   ابعاد: {width}x{height}")
        print(f"   Steps: {num_inference_steps}")
        print(f"   Guidance Scale: {guidance_scale}")
        print(f"   ControlNet Scale: {controlnet_conditioning_scale}")
        
        # تولید
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            num_images_per_prompt=num_images,
            width=width,
            height=height
        )
        
        print(f"✅ {len(output.images)} تصویر تولید شد")
        
        return output.images
    
    def generate_batch(
        self,
        control_image,
        prompts_list,
        negative_prompt="",
        **kwargs
    ):
        """
        تولید دسته‌ای با پرامپت‌های مختلف
        
        Args:
            control_image: تصویر کنترل
            prompts_list: لیست پرامپت‌ها
            negative_prompt: پرامپت منفی
            **kwargs: پارامترهای اضافی
            
        Returns:
            list: لیست تمام تصاویر تولید شده
        """
        all_images = []
        
        for i, prompt in enumerate(prompts_list):
            print(f"\n📝 پرامپت {i+1}/{len(prompts_list)}: {prompt[:50]}...")
            
            images = self.generate(
                control_image=control_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                **kwargs
            )
            
            all_images.extend(images)
        
        return all_images
    
    def generate_with_variations(
        self,
        control_image,
        base_prompt,
        variations,
        **kwargs
    ):
        """
        تولید با تنوع در پرامپت
        
        Args:
            control_image: تصویر کنترل
            base_prompt: پرامپت پایه
            variations: لیست تغییرات
            **kwargs: پارامترهای اضافی
            
        Returns:
            list: لیست تصاویر با تنوع
        """
        prompts = [f"{base_prompt}, {var}" for var in variations]
        return self.generate_batch(control_image, prompts, **kwargs)