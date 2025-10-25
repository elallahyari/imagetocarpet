# -*- coding: utf-8 -*-
import os
import sys
import argparse
from PIL import Image
import yaml

# اضافه کردن مسیر پروژه به sys.path از طریق ماژول متمرکز
try:
    from src.utils import paths
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.utils import paths

from src.pipeline.carpet_pipeline import CarpetDesignPipeline

def main():
    """
    تابع اصلی برای اجرای پایپلاین از طریق خط فرمان.
    """
    parser = argparse.ArgumentParser(
        description='🧶 سیستم هوشمند تبدیل تصویر به طرح صنعتی فرش (نسخه خط فرمان)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # آرگومان‌های اصلی
    parser.add_argument('--input', '-i', type=str, required=True, help='مسیر تصویر ورودی.')
    parser.add_argument('--output', '-o', type=str, default=paths.OUTPUT_DIR, help='مسیر پوشه خروجی.')
    parser.add_argument('--config', '-c', type=str, default=paths.DEFAULT_CONFIG_PATH, help='مسیر فایل تنظیمات YAML.')
    
    # تنظیمات مشخصات فرش
    carpet_group = parser.add_argument_group('📏 مشخصات فرش')
    carpet_group.add_argument('--width', type=int, help='عرض فرش (سانتی‌متر). مقدار پیش‌فرض از کانفیگ خوانده می‌شود.')
    carpet_group.add_argument('--height', type=int, help='طول فرش (سانتی‌متر).')
    carpet_group.add_argument('--shaneh', type=int, help='تعداد شانه فرش.')
    carpet_group.add_argument('--tar', type=int, help='تعداد تار در هر شانه.')
    
    # فعال/غیرفعال کردن مراحل پردازش
    process_group = parser.add_argument_group('🔄 مراحل پردازش')
    process_group.add_argument('--no-background-removal', action='store_false', dest='remove_background', help='غیرفعال کردن حذف پس‌زمینه.')
    process_group.add_argument('--no-edge-detection', action='store_false', dest='detect_edges', help='غیرفعال کردن تشخیص لبه.')
    process_group.add_argument('--no-ai-generation', action='store_false', dest='generate_design', help='غیرفعال کردن تولید طرح با AI.')
    process_group.add_argument('--no-symmetry', action='store_false', dest='apply_symmetry', help='غیرفعال کردن اعمال تقارن.')
    process_group.add_argument('--no-quantize', action='store_false', dest='quantize_colors', help='غیرفعال کردن کاهش رنگ.')
    process_group.add_argument('--vectorize', action='store_true', default=False, help='فعال کردن وکتوری‌سازی (نیاز به vtracer دارد).')
    
    # پارامترهای پیشرفته
    advanced_group = parser.add_argument_group('🔧 پارامترهای پیشرفته')
    advanced_group.add_argument('--n-colors', type=int, help='تعداد رنگ‌ها در حالت کوانتیزاسیون خودکار.')
    advanced_group.add_argument('--edge-method', type=str, choices=['HED', 'Canny', 'PiDiNet'], help='متد تشخیص لبه.')
    advanced_group.add_argument('--controlnet-model', type=str, help='نام یا مسیر مدل ControlNet برای استفاده.')
    advanced_group.add_argument('--controlnet-scale', type=float, help='میزان تاثیرپذیری از تصویر کنترل (لبه‌ها).')
    advanced_group.add_argument('--steps', type=int, help='تعداد مراحل نمونه‌برداری در Stable Diffusion.')
    advanced_group.add_argument('--seed', type=int, help='عدد seed برای تکرارپذیری نتایج.')

    args = parser.parse_args()
    
    try:
        # ۱. ساخت پایپلاین
        print("⏳ در حال ساخت پایپلاین پردازش...")
        pipeline = CarpetDesignPipeline(config_path=args.config)
        
        # ۲. اعمال تنظیمات از آرگومان‌های خط فرمان به کانفیگ پایپلاین
        if args.n_colors:
            pipeline.config['processing']['color_quantization']['n_colors'] = args.n_colors
        if args.edge_method:
            pipeline.config['models']['edge_detection']['method'] = args.edge_method
        if args.controlnet_scale:
            pipeline.config['generation']['controlnet_scale'] = args.controlnet_scale
        if args.steps:
            pipeline.config['generation']['steps'] = args.steps
        if args.seed:
            pipeline.config['generation']['seed'] = args.seed
            
        # ۳. تنظیم مشخصات فرش
        pipeline.carpet_specs = {
            'width_cm': args.width or 200,
            'height_cm': args.height or 300,
            'shaneh': args.shaneh or 50,
            'tar': args.tar or 12
        }
        
        # ۴. بارگذاری تصویر ورودی
        print(f"🖼️ در حال بارگذاری تصویر از: {args.input}")
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"فایل ورودی یافت نشد: {args.input}")
        input_image = Image.open(args.input).convert('RGB')
        
        # ۵. اجرای پردازش
        print("\n🚀 شروع پردازش تصویر...")
        
        # تبدیل آرگومان‌ها به دیکشنری برای run_config
        run_config_dict = vars(args)
        
        results = pipeline.process_image(
            input_image=input_image,
            output_dir=args.output,
            run_config=run_config_dict 
        )
        
        print("\n✨ پردازش با موفقیت کامل شد!")
        print(f"📁 نتایج در پوشه زیر ذخیره شدند:\n{results.get('output_path', 'N/A')}")
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ یک خطای بحرانی در حین اجرا رخ داد:")
        print(f"   نوع خطا: {type(e).__name__}")
        print(f"   پیام خطا: {e}")
        print("="*80)
        sys.exit(1)

if __name__ == '__main__':
    main()