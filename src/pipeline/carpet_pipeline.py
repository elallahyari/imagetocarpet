# -*- coding: utf-8 -*-
import os
import yaml
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import json

from ..models.sam_segmenter import SAMSegmenter
from ..models.edge_detector import EdgeDetector
from ..models.controlnet_generator import ControlNetGenerator
from ..processors.color_quantizer import ColorQuantizer
from ..processors.symmetry_maker import SymmetryMaker
from ..processors.vectorizer import Vectorizer
from ..utils.paths import DEFAULT_CONFIG_PATH, SAM_MODEL_CHECKPOINT

class ProcessingCancelledError(Exception):
    """این خطا زمانی که پردازش توسط کاربر لغو می‌شود، فراخوانی می‌گردد."""
    pass

class CarpetDesignPipeline:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        print("=" * 80)
        print("🧶 سیستم هوشمند تبدیل تصویر به طرح صنعتی فرش (پایپلاین نسخه ۲.۷)")
        print("=" * 80)
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # --- منطق اصلاح‌شده و بهبودیافته برای انتخاب دستگاه ---
        config_device = self.config.get('models', {}).get('device', 'auto')
        if config_device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config_device
        
        print(f"🖥️  دستگاه پردازشی انتخاب شده: {self.device}")
        
        self._sam = None
        self._edge_detector = None
        self._controlnet_instances = {}
        
        self.color_quantizer = ColorQuantizer()
        self.symmetry_maker = SymmetryMaker()
        self.vectorizer = None
        self.custom_palette = None
        self.carpet_specs = None

    def _lazy_load_controlnet(self, base_model_path, controlnet_path):
        instance_key = (base_model_path, controlnet_path)
        if instance_key not in self._controlnet_instances:
            self.log_callback(f"⏳ در حال بارگذاری مدل ControlNet + Stable Diffusion...")
            self.log_callback(f"   - مدل پایه: {base_model_path}")
            self.log_callback(f"   - مدل کنترل: {controlnet_path}")
            self.log_callback("   (این مرحله ممکن است بسیار زمان‌بر باشد و به حافظه VRAM بالایی نیاز دارد)")
            
            generator_instance = ControlNetGenerator(
                base_model=base_model_path,
                controlnet_model=controlnet_path,
                device=self.device
            )
            self._controlnet_instances[instance_key] = generator_instance
            self.log_callback(f"✅ مدل ControlNet با موفقیت بارگذاری شد.")
        return self._controlnet_instances[instance_key]

    def _lazy_load_sam(self):
        if self._sam is None:
            self.log_callback("⏳ در حال بارگذاری مدل SAM (ممکن است کمی طول بکشد)...")
            sam_model_type = self.config.get('models', {}).get('sam', {}).get('model_type', 'vit_h')
            self._sam = SAMSegmenter(
                model_type=sam_model_type,
                checkpoint_path=SAM_MODEL_CHECKPOINT,
                device=self.device
            )
            self.log_callback("✅ مدل SAM با موفقیت بارگذاری شد.")
        return self._sam

    def _lazy_load_edge_detector(self):
        if self._edge_detector is None:
            method = self.config.get('processing', {}).get('edge_detection', {}).get('method', 'HED')
            self.log_callback(f"⏳ در حال بارگذاری مدل تشخیص لبه ({method})...")
            self._edge_detector = EdgeDetector(method=method, device=self.device)
            self.log_callback("✅ مدل تشخیص لبه با موفقیت بارگذاری شد.")
        return self._edge_detector

    def _check_for_cancel(self, cancel_event):
        if cancel_event and cancel_event.is_set():
            raise ProcessingCancelledError("عملیات توسط کاربر لغو شد.")

    def _update_progress(self, current_step, total_steps, cancel_event):
        self._check_for_cancel(cancel_event)
        if self.progress_callback:
            self.progress_callback(current_step, total_steps)
            
    def process_image(self, input_image, output_dir='output', run_config=None, cancel_event=None, log_callback=print, progress_callback=None):
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        
        total_steps = sum(1 for step in [
            'remove_background', 'detect_edges', 'generate_design', 
            'quantize_colors', 'apply_symmetry', 'vectorize'
        ] if run_config.get(step))
        current_step = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, timestamp)
        os.makedirs(output_path, exist_ok=True)
        self.log_callback(f"📁 پوشه خروجی برای این اجرا: {output_path}\n")
        self._check_for_cancel(cancel_event)
        image = input_image.copy()
        self.log_callback(f"📷 تصویر ورودی با ابعاد {image.width}x{image.height} دریافت شد.")
        results = {'original': image, 'output_path': output_path}
        self.log_callback("\n" + "="*40 + "\nمرحله ۰: محاسبه ابعاد نقشه گره\n" + "="*40)
        spec = self.carpet_specs
        width_px = int((spec['width_cm'] / 10) * spec['shaneh'])
        height_px = int((spec['height_cm'] / 10) * spec['tar'])
        self.log_callback(f"   - ابعاد محاسبه شده برای دستگاه: {width_px} x {height_px} پیکسل (گره)")
        image = image.resize((width_px, height_px), Image.LANCZOS)
        self.log_callback("   - تصویر ورودی به ابعاد نقشه گره تغییر اندازه یافت.")
        if run_config.get('save_intermediate'):
            image.save(os.path.join(output_path, '01_knot_resolution.png'))

        processed_image = image
        if run_config.get('remove_background'):
            current_step += 1
            self.log_callback("\n" + "="*40 + f"\nمرحله {current_step}/{total_steps}: حذف پس‌زمینه با SAM\n" + "="*40)
            self._update_progress(current_step, total_steps, cancel_event)
            sam_model = self._lazy_load_sam()
            main_mask = sam_model.extract_main_object(image, fast_mode=run_config.get('sam_fast_mode', False))
            if main_mask is not None:
                processed_image = sam_model.apply_mask_to_image(image, main_mask)
                if run_config.get('save_intermediate'):
                    processed_image.save(os.path.join(output_path, '02_background_removed.png'))
                self.log_callback("✅ پس‌زمینه با موفقیت حذف شد.")
            else:
                self.log_callback("⚠️ هیچ شیء غالبی یافت نشد. از تصویر اصلی استفاده می‌شود.")

        refined_edges = None
        if run_config.get('detect_edges'):
            current_step += 1
            self.log_callback("\n" + "="*40 + f"\nمرحله {current_step}/{total_steps}: تشخیص لبه‌ها\n" + "="*40)
            self._update_progress(current_step, total_steps, cancel_event)
            edge_model = self._lazy_load_edge_detector()
            edges = edge_model.detect_edges(processed_image)
            refined_edges_np = edge_model.refine_edges(edges, kernel_size=3)
            refined_edges = Image.fromarray(refined_edges_np)
            if run_config.get('save_intermediate'):
                refined_edges.save(os.path.join(output_path, '03_edges.png'))
            self.log_callback("✅ لبه‌ها با موفقیت تشخیص داده شدند.")
            
        working_image = processed_image
        if run_config.get('generate_design'):
            current_step += 1
            self.log_callback("\n" + "="*40 + f"\nمرحله {current_step}/{total_steps}: تولید طرح فرش با AI\n" + "="*40)
            self._update_progress(current_step, total_steps, cancel_event)
            
            base_model_path = run_config.get('base_model_path')
            controlnet_path = run_config.get('controlnet_path')

            if not base_model_path or not controlnet_path:
                self.log_callback("❌ مدل پایه یا مدل کنترل مشخص نشده است. این مرحله رد می‌شود.")
            else:
                control_image_for_ai = None
                if 'tile' in controlnet_path.lower():
                    self.log_callback("ℹ️ از حالت ControlNet-Tile استفاده می‌شود. تصویر اصلی به عنوان ورودی کنترل خواهد بود.")
                    control_image_for_ai = processed_image
                else:
                    if refined_edges:
                        control_image_for_ai = refined_edges
                    else:
                        self.log_callback("⚠️ تیک 'تشخیص لبه' فعال نیست. ورودی برای این مدل کنترل وجود ندارد. این مرحله رد می‌شود.")

                if control_image_for_ai:
                    controlnet_model = self._lazy_load_controlnet(base_model_path, controlnet_path)
                    gen_config = self.config['generation']
                    enhanced_prompt = gen_config['prompts']['positive']
                    if self.carpet_specs:
                        enhanced_prompt += f", carpet design for {self.carpet_specs['width_cm']}x{self.carpet_specs['height_cm']}cm, {self.carpet_specs['shaneh']} raj density"
                    
                    output_width, output_height = width_px, height_px

                    generated_images = controlnet_model.generate(
                        control_image=control_image_for_ai,
                        prompt=enhanced_prompt,
                        negative_prompt=gen_config['prompts']['negative'],
                        num_inference_steps=gen_config['steps'],
                        guidance_scale=gen_config['guidance_scale'],
                        controlnet_conditioning_scale=gen_config['controlnet_scale'],
                        seed=gen_config['seed'] if gen_config['seed'] != -1 else None,
                        width=output_width,
                        height=output_height
                    )
                    working_image = generated_images[0]
                    if run_config.get('save_intermediate'):
                        working_image.save(os.path.join(output_path, '04_ai_generated.png'))
                    self.log_callback("✅ طرح جدید با هوش مصنوعی تولید شد.")
        
        if run_config.get('quantize_colors'):
            current_step += 1
            self.log_callback("\n" + "="*40 + f"\nمرحله {current_step}/{total_steps}: کاهش رنگ‌ها\n" + "="*40)
            self._update_progress(current_step, total_steps, cancel_event)
            if self.custom_palette is not None and len(self.custom_palette) > 0:
                self.log_callback(f"🎨 استفاده از پالت رنگی سفارشی با {len(self.custom_palette)} رنگ.")
                quantized_image, palette = self.color_quantizer.apply_palette_with_dithering(working_image, self.custom_palette)
            else:
                n_colors = self.config['processing']['color_quantization']['n_colors']
                self.log_callback(f"🎨 کوانتیزه کردن خودکار به {n_colors} رنگ.")
                self.color_quantizer.n_colors = n_colors
                quantized_image, palette = self.color_quantizer.quantize_with_dithering(working_image)
            
            palette_viz = self.color_quantizer.create_palette_visualization(palette)
            
            if run_config.get('save_intermediate'):
                quantized_image.save(os.path.join(output_path, '05_quantized.png'))
                palette_viz.save(os.path.join(output_path, '05_palette.png'))
            self.save_color_info(palette, output_path)
            working_image = quantized_image
            self.log_callback("✅ رنگ‌های تصویر با موفقیت کاهش یافت.")
        
        if run_config.get('apply_symmetry') and not run_config.get('is_full_design'):
            current_step += 1
            self.log_callback("\n" + "="*40 + f"\nمرحله {current_step}/{total_steps}: ایجاد تقارن و چیدمان\n" + "="*40)
            self._update_progress(current_step, total_steps, cancel_event)
            four_way = self.symmetry_maker.create_four_way_mirror(working_image)
            if run_config.get('save_intermediate'):
                four_way.save(os.path.join(output_path, '06_four_way_symmetry.png'))
            
            background_color = tuple(self.config['output'].get('medallion_background_color', [245, 240, 230]))
            
            medallion_layout = self.symmetry_maker.create_medallion_layout(
                center_element=four_way,
                canvas_size=(width_px, height_px),
                background_color=background_color
            )
            if run_config.get('save_intermediate'):
                medallion_layout.save(os.path.join(output_path, '07_medallion_layout.png'))
            working_image = medallion_layout
            self.log_callback("✅ تقارن و چیدمان مدالیون اعمال شد.")
        elif run_config.get('is_full_design'):
             self.log_callback("\n" + "="*40 + "\nℹ️ مرحله تقارن و چیدمان رد شد (ورودی یک طرح کامل است).\n" + "="*40)

        if run_config.get('vectorize'):
            current_step += 1
            self.log_callback("\n" + "="*40 + f"\nمرحله {current_step}/{total_steps}: وکتوری‌سازی\n" + "="*40)
            self._update_progress(current_step, total_steps, cancel_event)
            try:
                self.vectorizer = Vectorizer(method='vtracer')
                svg_path = os.path.join(output_path, 'final_design.svg')
                
                vector_kwargs = {
                    'filter_speckle': run_config.get('vector_speckle', 4),
                    'color_precision': run_config.get('vector_color_precision', 6),
                    'corner_threshold': run_config.get('vector_corner_threshold', 60)
                }

                svg_result = self.vectorizer.vectorize(working_image, svg_path, **vector_kwargs)
                if svg_result:
                    pdf_path = os.path.join(output_path, 'final_design.pdf')
                    self.vectorizer.svg_to_pdf(svg_path, pdf_path)
                    self.log_callback("✅ وکتورسازی با موفقیت انجام شد.")
            except Exception as e:
                self.log_callback(f"❌ خطا در وکتورسازی: {e}")

        final_image = working_image
        final_path = os.path.join(output_path, 'final_design.png')
        final_image.save(final_path)
        results['final_png'] = final_path
        self.log_callback(f"\n✅ نتیجه نهایی با ابعاد دقیق {final_image.width}x{final_image.height} ذخیره شد: {final_path}")

        if self.carpet_specs:
            self.save_carpet_specs(output_path)
            
        return results

    def save_color_info(self, palette, output_path):
        color_info = {
            'palette': [
                {'index': i+1, 'rgb': list(map(int, color)), 'hex': '#%02x%02x%02x' % tuple(map(int, color))}
                for i, color in enumerate(palette)
            ],
            'total_colors': len(palette)
        }
        info_path = os.path.join(output_path, 'color_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(color_info, f, indent=4, ensure_ascii=False)
        self.log_callback(f"   - اطلاعات رنگی در فایل color_info.json ذخیره شد.")

    def save_carpet_specs(self, output_path):
        if not self.carpet_specs:
            return

        density = (self.carpet_specs['shaneh'] / 10) * self.carpet_specs['tar']
        width_m = self.carpet_specs['width_cm'] / 100
        height_m = self.carpet_specs['height_cm'] / 100
        area_m2 = width_m * height_m
        total_knots = (width_m * self.carpet_specs['shaneh'] * 10) * (height_m * self.carpet_specs['tar'] * 10)
        
        specs = {
            'dimensions': { 'width_cm': self.carpet_specs['width_cm'], 'height_cm': self.carpet_specs['height_cm'], 'width_m': width_m, 'height_m': height_m },
            'weaving': { 'shaneh_per_10cm': self.carpet_specs['shaneh'], 'tar_per_10cm': self.carpet_specs['tar'], 'density_per_dm2': int(density), 'density_per_m2': int(density * 100), 'raj': self.carpet_specs['shaneh'] },
            'production_estimate': { 'area_m2': round(area_m2, 2), 'total_knots': int(total_knots) }
        }
        
        specs_path_json = os.path.join(output_path, 'carpet_specifications.json')
        with open(specs_path_json, 'w', encoding='utf-8') as f:
            json.dump(specs, f, indent=4, ensure_ascii=False)

        specs_path_txt = os.path.join(output_path, 'carpet_specifications.txt')
        with open(specs_path_txt, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\nمشخصات فنی فرش\n" + "=" * 60 + "\n\n")
            f.write(f"ابعاد:\n")
            f.write(f"  - عرض: {specs['dimensions']['width_cm']} سانتی‌متر ({specs['dimensions']['width_m']:.2f} متر)\n")
            f.write(f"  - طول: {specs['dimensions']['height_cm']} سانتی‌متر ({specs['dimensions']['height_m']:.2f} متر)\n")
            f.write(f"  - مساحت: {specs['production_estimate']['area_m2']:.2f} متر مربع\n\n")
            f.write(f"بافت:\n")
            f.write(f"  - شانه: {specs['weaving']['shaneh_per_10cm']} (گره در عرض ۱۰ سانتی‌متر)\n")
            f.write(f"  - تراکم طولی (تار): {specs['weaving']['tar_per_10cm']} (گره در طول ۱۰ سانتی‌متر)\n")
            f.write(f"  - تراکم کل: {specs['weaving']['density_per_m2']} گره در متر مربع\n\n")
            f.write(f"برآورد تولید:\n")
            f.write(f"  - تعداد کل گره‌ها: {specs['production_estimate']['total_knots']:,}\n")

        self.log_callback(f"   - مشخصات فرش در فایل‌های JSON و TXT ذخیره شد.")