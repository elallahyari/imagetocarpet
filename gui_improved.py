# -*- coding: utf-8 -*-
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, colorchooser, simpledialog
from PIL import Image, ImageTk
import threading
import numpy as np
import shutil
import json
import yaml

try:
    from src.utils import paths
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import paths

from src.pipeline.carpet_pipeline import CarpetDesignPipeline, ProcessingCancelledError
from src.processors.color_quantizer import ColorQuantizer
from src.utils.palette_manager import PaletteManager
from src.utils.device_profile_manager import DeviceProfileManager

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

class CarpetDesignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧶 سیستم تبدیل تصویر به طرح فرش - نسخه ۲.۷")
        self.root.geometry("1400x950")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.pipeline = None
        self.palette_manager = PaletteManager()
        self.profile_manager = DeviceProfileManager()

        self.input_image_path = None
        self.input_image = None
        self.results = None
        self.custom_palette = []
        
        self.processing_thread = None
        self.cancel_event = threading.Event()
        
        self.config = self.load_app_config()
        self.model_profiles = self.config.get('model_profiles', [])
        self.setup_ui()

    def load_app_config(self):
        try:
            with open(paths.DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror("خطا در بارگذاری کانفیگ", f"فایل model_config.yaml یافت نشد یا خراب است.\n{e}")
            return {}

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="⚙️ تنظیمات اصلی")
        
        profiles_tab = ttk.Frame(notebook)
        notebook.add(profiles_tab, text="💻 پروفایل‌های دستگاه")

        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="🔧 پیشرفته")
        
        color_tab = ttk.Frame(notebook)
        notebook.add(color_tab, text="🎨 پالت رنگی")
        
        self.setup_main_tab(main_tab)
        self.setup_profiles_tab(profiles_tab)
        self.setup_advanced_tab(advanced_tab)
        self.setup_color_tab(color_tab)
        
        self.status_bar = ttk.Label(main_frame, text="آماده به کار...", anchor=tk.W, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

    def setup_main_tab(self, parent):
        io_frame = ttk.LabelFrame(parent, text="📁 ورودی و خروجی", padding="10")
        io_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_path_var = tk.StringVar()
        ttk.Label(io_frame, text="تصویر ورودی:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(io_frame, textvariable=self.input_path_var, width=70).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="انتخاب...", command=self.select_input_file).grid(row=0, column=2, padx=5)
        self.preview_button = ttk.Button(io_frame, text="مشاهده", command=self.preview_input, state=tk.DISABLED)
        self.preview_button.grid(row=0, column=3, padx=5)
        
        self.output_path_var = tk.StringVar(value=os.path.abspath(paths.OUTPUT_DIR))
        ttk.Label(io_frame, text="پوشه خروجی:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(io_frame, textvariable=self.output_path_var, width=70).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(io_frame, text="انتخاب...", command=self.select_output_folder).grid(row=1, column=2, padx=5)
        ttk.Button(io_frame, text="باز کردن", command=self.open_output_folder).grid(row=1, column=3, padx=5)

        carpet_frame = ttk.LabelFrame(parent, text="📏 مشخصات فرش", padding="10")
        carpet_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(carpet_frame, text="عرض (cm):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.carpet_width_var = tk.IntVar(value=200)
        ttk.Spinbox(carpet_frame, from_=50, to=1000, textvariable=self.carpet_width_var, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(carpet_frame, text="طول (cm):").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(20,0))
        self.carpet_height_var = tk.IntVar(value=300)
        ttk.Spinbox(carpet_frame, from_=50, to=1000, textvariable=self.carpet_height_var, width=15).grid(row=0, column=3, padx=5)
        
        ttk.Label(carpet_frame, text="شانه:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.shaneh_var = tk.IntVar(value=50)
        ttk.Spinbox(carpet_frame, from_=20, to=100, textvariable=self.shaneh_var, width=15).grid(row=1, column=1, padx=5)
        
        ttk.Label(carpet_frame, text="تار:").grid(row=1, column=2, sticky=tk.W, pady=5, padx=(20,0))
        self.tar_var = tk.IntVar(value=12)
        ttk.Spinbox(carpet_frame, from_=6, to=20, textvariable=self.tar_var, width=15).grid(row=1, column=3, padx=5)
        
        self.density_label = ttk.Label(carpet_frame, text="تراکم: 600 گره/dm²", font=('Arial', 10, 'bold'))
        self.density_label.grid(row=2, column=0, columnspan=4, pady=10)
        
        self.shaneh_var.trace('w', self.update_density)
        self.tar_var.trace('w', self.update_density)
        
        process_frame = ttk.LabelFrame(parent, text="🔄 مراحل پردازش", padding="10")
        process_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.remove_bg_var = tk.BooleanVar(value=True)
        self.edge_detect_var = tk.BooleanVar(value=True)
        self.ai_generate_var = tk.BooleanVar(value=True)
        self.symmetry_var = tk.BooleanVar(value=True)
        self.quantize_var = tk.BooleanVar(value=True)
        self.vectorize_var = tk.BooleanVar(value=False)
        self.is_full_design_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(process_frame, text="حذف پس‌زمینه", variable=self.remove_bg_var).grid(row=0, column=0, sticky=tk.W, pady=2, padx=5)
        ttk.Checkbutton(process_frame, text="تشخیص لبه", variable=self.edge_detect_var).grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Checkbutton(process_frame, text="تولید با AI", variable=self.ai_generate_var).grid(row=1, column=0, sticky=tk.W, pady=2, padx=5)
        ttk.Checkbutton(process_frame, text="اعمال تقارن", variable=self.symmetry_var).grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Checkbutton(process_frame, text="کاهش رنگ", variable=self.quantize_var).grid(row=2, column=0, sticky=tk.W, pady=2, padx=5)
        cb_vector = ttk.Checkbutton(process_frame, text="وکتوری‌سازی", variable=self.vectorize_var)
        cb_vector.grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)
        Tooltip(cb_vector, "برای این گزینه باید ابزار vtracer نصب و در PATH سیستم باشد.")
        
        full_design_cb = ttk.Checkbutton(process_frame, text="ورودی یک طرح کامل است (رد کردن چیدمان)", variable=self.is_full_design_var)
        full_design_cb.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        
        preview_log_frame = ttk.Frame(parent)
        preview_log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        preview_frame = ttk.LabelFrame(preview_log_frame, text="🖼️ پیش‌نمایش", padding="10")
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.preview_label = ttk.Label(preview_frame, text="تصویری انتخاب نشده", background='#f0f0f0', anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        log_frame = ttk.LabelFrame(preview_log_frame, text="📋 گزارش", padding="10")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        main_control_frame = ttk.Frame(button_frame)
        main_control_frame.pack(side=tk.LEFT)

        self.start_button = ttk.Button(main_control_frame, text="🚀 شروع پردازش", command=self.start_processing, width=25)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(main_control_frame, text="🛑 لغو پردازش", command=self.cancel_processing, width=20, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        sub_control_frame = ttk.Frame(button_frame)
        sub_control_frame.pack(side=tk.RIGHT)
        
        ttk.Button(sub_control_frame, text="📥 بارگذاری تنظیمات", command=self.load_settings_from_file, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(sub_control_frame, text="💾 ذخیره تنظیمات", command=self.save_settings_to_file, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(sub_control_frame, text="❌ خروج", command=self.on_closing, width=15).pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(parent, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

    def setup_profiles_tab(self, parent):
        profiles_frame = ttk.LabelFrame(parent, text="🎛️ مدیریت پروفایل‌های دستگاه", padding="10")
        profiles_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(profiles_frame, text="انتخاب پروفایل:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.profile_var = tk.StringVar()
        self.profiles_combobox = ttk.Combobox(profiles_frame, textvariable=self.profile_var, width=40, state='readonly')
        self.profiles_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.populate_profiles_combobox()
        
        buttons_frame = ttk.Frame(profiles_frame)
        buttons_frame.grid(row=0, column=2, padx=10)
        
        ttk.Button(buttons_frame, text="✅ اعمال پروفایل", command=self.apply_selected_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="💾 ذخیره پروفایل جدید", command=self.save_new_profile).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="🗑️ حذف پروفایل", command=self.delete_selected_profile).pack(side=tk.LEFT, padx=5)

        info_label = ttk.Label(parent, 
            text="در این بخش می‌توانید تنظیمات مربوط به یک دستگاه خاص (شانه، تار و پالت رنگی) را ذخیره و بازیابی کنید.\n"
                 "برای ذخیره یک پروفایل جدید، ابتدا تنظیمات شانه و تار را در تب 'تنظیمات اصلی' و پالت رنگی را در تب 'پالت رنگی' مشخص کنید،\n"
                 "سپس دکمه 'ذخیره پروفایل جدید' را بزنید.",
            justify=tk.RIGHT, wraplength=800)
        info_label.pack(pady=20, padx=10)

    def setup_advanced_tab(self, parent):
        ai_frame = ttk.LabelFrame(parent, text="🤖 تنظیمات AI", padding="10")
        ai_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(ai_frame, text="پروفایل مدل:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_profile_var = tk.StringVar()
        profile_names = [p['name'] for p in self.model_profiles]
        self.model_profile_combo = ttk.Combobox(ai_frame, textvariable=self.model_profile_var, values=profile_names, width=35, state='readonly')
        self.model_profile_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(ai_frame, text="روش کنترل:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.controlnet_name_var = tk.StringVar()
        self.controlnet_combo = ttk.Combobox(ai_frame, textvariable=self.controlnet_name_var, width=35, state='readonly')
        self.controlnet_combo.grid(row=1, column=1, sticky=tk.W, padx=5)

        # --- رفع خطا: ویجت راهنما را قبل از فراخوانی event ها ایجاد می‌کنیم ---
        self.tile_hint_label = ttk.Label(ai_frame, text="", foreground="blue")
        self.tile_hint_label.grid(row=1, column=2, padx=5, sticky=tk.W)
        
        self.model_profile_combo.bind("<<ComboboxSelected>>", self._on_model_profile_selected)
        self.controlnet_combo.bind("<<ComboboxSelected>>", self._on_controlnet_selected)
        
        if profile_names:
            self.model_profile_combo.set(profile_names[0])
            self._on_model_profile_selected(None)

        ttk.Label(ai_frame, text="میزان شباهت:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.controlnet_scale_var = tk.DoubleVar(value=self.config['generation']['controlnet_scale'])
        scale_widget = ttk.Scale(ai_frame, from_=0.1, to=2.0, variable=self.controlnet_scale_var, orient=tk.HORIZONTAL, length=300)
        scale_widget.grid(row=2, column=1, padx=5)
        Tooltip(scale_widget, "میزان وفاداری طرح نهایی به ورودی کنترل (لبه یا تصویر).\nمقادیر بالاتر = شباهت بیشتر.")
        
        self.scale_label = ttk.Label(ai_frame, text=f"{self.controlnet_scale_var.get():.2f}")
        self.scale_label.grid(row=2, column=2, padx=5)
        self.controlnet_scale_var.trace('w', lambda *args: self.scale_label.config(text=f"{self.controlnet_scale_var.get():.2f}"))
        
        ttk.Label(ai_frame, text="تعداد مراحل:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.steps_var = tk.IntVar(value=self.config['generation']['steps'])
        steps_widget = ttk.Spinbox(ai_frame, from_=10, to=100, textvariable=self.steps_var, width=10)
        steps_widget.grid(row=3, column=1, sticky=tk.W, padx=5)
        Tooltip(steps_widget, "تعداد مراحل تولید تصویر توسط AI.\nمقادیر بالاتر کیفیت را افزایش می‌دهد اما زمان‌برتر است.")
        
        ttk.Label(ai_frame, text="Guidance:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.guidance_var = tk.DoubleVar(value=self.config['generation']['guidance_scale'])
        guidance_widget = ttk.Spinbox(ai_frame, from_=1.0, to=20.0, textvariable=self.guidance_var, width=10, increment=0.5)
        guidance_widget.grid(row=4, column=1, sticky=tk.W, padx=5)
        Tooltip(guidance_widget, "میزان پیروی مدل از پرامپت متنی.\nمقادیر بالاتر باعث پیروی دقیق‌تر از متن می‌شود.")
        
        edge_frame = ttk.LabelFrame(parent, text="✏️ تشخیص لبه", padding="10")
        edge_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(edge_frame, text="روش:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.edge_method_var = tk.StringVar(value=self.config['processing']['edge_detection']['method'])
        ttk.Combobox(edge_frame, textvariable=self.edge_method_var, values=["HED", "Canny", "PiDiNet"], width=15, state='readonly').grid(row=0, column=1, sticky=tk.W, padx=5)
        
        bg_frame = ttk.LabelFrame(parent, text="🖼️ حذف پس‌زمینه", padding="10")
        bg_frame.pack(fill=tk.X, padx=10, pady=5)
        self.sam_fast_mode_var = tk.BooleanVar(value=False)
        sam_cb = ttk.Checkbutton(bg_frame, text="حالت سریع (تمرکز روی مرکز تصویر)", variable=self.sam_fast_mode_var)
        sam_cb.pack(anchor=tk.W)
        Tooltip(sam_cb, "در این حالت، مدل SAM فقط تلاش می‌کند شیء موجود در مرکز تصویر را پیدا کند.\nاین کار سریع‌تر است اما ممکن است برای سوژه‌های خارج از مرکز دقت کمتری داشته باشد.")

        vector_frame = ttk.LabelFrame(parent, text="✒️ تنظیمات وکتورسازی", padding="10")
        vector_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(vector_frame, text="حذف نویز (Speckle):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.vector_speckle_var = tk.IntVar(value=4)
        speckle_widget = ttk.Spinbox(vector_frame, from_=0, to=50, textvariable=self.vector_speckle_var, width=10)
        speckle_widget.grid(row=0, column=1, padx=5, sticky=tk.W)
        Tooltip(speckle_widget, "حذف لکه‌ها و نویزهای کوچک‌تر از این اندازه (پیکسل).\nمقادیر بالاتر نویز بیشتری را حذف می‌کند.")
        ttk.Label(vector_frame, text="دقت رنگ:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.vector_color_precision_var = tk.IntVar(value=6)
        color_prec_widget = ttk.Spinbox(vector_frame, from_=1, to=8, textvariable=self.vector_color_precision_var, width=10)
        color_prec_widget.grid(row=1, column=1, padx=5, sticky=tk.W)
        Tooltip(color_prec_widget, "تعداد بیت برای هر کانال رنگی (1-8).\nمقادیر بالاتر دقت رنگ را افزایش ولی حجم فایل را نیز بیشتر می‌کند.")
        ttk.Label(vector_frame, text="آستانه گوشه‌ها:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.vector_corner_threshold_var = tk.IntVar(value=60)
        corner_widget = ttk.Spinbox(vector_frame, from_=0, to=180, textvariable=self.vector_corner_threshold_var, width=10)
        corner_widget.grid(row=2, column=1, padx=5, sticky=tk.W)
        Tooltip(corner_widget, "میزان تیزی گوشه‌ها (درجه).\nمقادیر کمتر گوشه‌های تیزتر و مقادیر بیشتر گوشه‌های نرم‌تری ایجاد می‌کند.")

        output_frame = ttk.LabelFrame(parent, text="💾 خروجی", padding="10")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.save_intermediate_var = tk.BooleanVar(value=self.config.get('output', {}).get('save_intermediate', True))
        ttk.Checkbutton(output_frame, text="ذخیره مراحل میانی", variable=self.save_intermediate_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
    def _on_model_profile_selected(self, event):
        selected_profile_name = self.model_profile_var.get()
        for profile in self.model_profiles:
            if profile['name'] == selected_profile_name:
                controlnet_names = [cn['name'] for cn in profile['controlnets']]
                self.controlnet_combo['values'] = controlnet_names
                if controlnet_names:
                    self.controlnet_combo.set(controlnet_names[0])
                    self._on_controlnet_selected(None)
                break

    def _on_controlnet_selected(self, event):
        selected_cn_name = self.controlnet_name_var.get()
        if "tile" in selected_cn_name.lower():
            self.tile_hint_label.config(text="(برای مدل Tile، تیک 'تشخیص لبه' را بردارید)")
        else:
            self.tile_hint_label.config(text="")

    # ... (بقیه توابع کلاس از پاسخ‌های قبلی کپی شوند) ...
    # ... (کد کامل این توابع باید در فایل نهایی شما وجود داشته باشد) ...
    def setup_color_tab(self, parent):
        method_frame = ttk.LabelFrame(parent, text="🎨 روش انتخاب", padding="10")
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.palette_method_var = tk.StringVar(value="auto")
        ttk.Radiobutton(method_frame, text="خودکار", variable=self.palette_method_var, value="auto").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(method_frame, text="دستی", variable=self.palette_method_var, value="custom").pack(anchor=tk.W, pady=2)
        
        count_frame = ttk.LabelFrame(parent, text="🔢 تعداد رنگ (در حالت خودکار)", padding="10")
        count_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(count_frame, text="تعداد:").grid(row=0, column=0, sticky=tk.W, pady=5)
        default_colors = self.config['processing']['color_quantization']['n_colors']
        self.n_colors_var = tk.IntVar(value=default_colors)
        ttk.Spinbox(count_frame, from_=2, to=20, textvariable=self.n_colors_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(count_frame, text="(حداقل: 2، حداکثر: 20)").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        custom_frame = ttk.LabelFrame(parent, text="✋ پالت دستی", padding="10")
        custom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.colors_display_frame = ttk.Frame(custom_frame)
        self.colors_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        buttons_frame = ttk.Frame(custom_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(buttons_frame, text="➕ افزودن", command=self.add_color).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="🗑️ پاک کردن", command=self.clear_colors).pack(side=tk.LEFT, padx=5)
        self.extract_button = ttk.Button(buttons_frame, text="📥 استخراج از تصویر ورودی", command=self.extract_palette_from_image, state=tk.DISABLED)
        self.extract_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(buttons_frame, text="📄 استخراج از فایل نمونه", command=self.extract_palette_from_sample_file).pack(side=tk.LEFT, padx=5)

        preset_frame = ttk.LabelFrame(parent, text="📚 پالت‌های آماده", padding="10")
        preset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preset_var = tk.StringVar(value="traditional_persian")
        
        presets = self.palette_manager.get_preset_options()
        
        for i, (text, value) in enumerate(presets):
            ttk.Radiobutton(preset_frame, text=text, variable=self.preset_var, value=value).grid(
                row=i//3, column=i%3, sticky=tk.W, pady=2, padx=10
            )
        
        preset_button_frame = ttk.Frame(preset_frame)
        preset_button_frame.grid(row=(len(presets)//3) + 1, column=0, columnspan=3, pady=10)
        
        ttk.Button(preset_button_frame, text="👁️ مشاهده پالت", command=self.preview_preset_palette).pack(side=tk.LEFT, padx=5)
        ttk.Button(preset_button_frame, text="✅ استفاده از این پالت", command=self.use_preset_palette).pack(side=tk.LEFT, padx=5)

        self.update_colors_display()

    def get_run_config(self):
        base_model_path = ""
        controlnet_path = ""
        selected_profile_name = self.model_profile_var.get()
        selected_cn_name = self.controlnet_name_var.get()

        for profile in self.model_profiles:
            if profile['name'] == selected_profile_name:
                base_model_path = profile['base_model']
                for cn in profile['controlnets']:
                    if cn['name'] == selected_cn_name:
                        controlnet_path = cn['path']
                        break
                break

        return {
            'base_model_path': base_model_path,
            'controlnet_path': controlnet_path,
            'remove_background': self.remove_bg_var.get(),
            'detect_edges': self.edge_detect_var.get(),
            'generate_design': self.ai_generate_var.get(),
            'apply_symmetry': self.symmetry_var.get(),
            'quantize_colors': self.quantize_var.get(),
            'vectorize': self.vectorize_var.get(),
            'save_intermediate': self.save_intermediate_var.get(),
            'is_full_design': self.is_full_design_var.get(),
            'sam_fast_mode': self.sam_fast_mode_var.get(),
            'vector_speckle': self.vector_speckle_var.get(),
            'vector_color_precision': self.vector_color_precision_var.get(),
            'vector_corner_threshold': self.vector_corner_threshold_var.get(),
        }

    def get_all_settings(self):
        converted_palette = [tuple(map(int, color)) for color in self.custom_palette]

        return {
            'carpet_width': self.carpet_width_var.get(),
            'carpet_height': self.carpet_height_var.get(),
            'shaneh': self.shaneh_var.get(),
            'tar': self.tar_var.get(),
            'remove_bg': self.remove_bg_var.get(),
            'edge_detect': self.edge_detect_var.get(),
            'ai_generate': self.ai_generate_var.get(),
            'symmetry': self.symmetry_var.get(),
            'quantize': self.quantize_var.get(),
            'vectorize': self.vectorize_var.get(),
            'is_full_design': self.is_full_design_var.get(),
            'model_profile_name': self.model_profile_var.get(),
            'controlnet_name': self.controlnet_name_var.get(),
            'controlnet_scale': self.controlnet_scale_var.get(),
            'steps': self.steps_var.get(),
            'guidance': self.guidance_var.get(),
            'edge_method': self.edge_method_var.get(),
            'sam_fast_mode': self.sam_fast_mode_var.get(),
            'save_intermediate': self.save_intermediate_var.get(),
            'palette_method': self.palette_method_var.get(),
            'n_colors': self.n_colors_var.get(),
            'custom_palette': converted_palette,
            'vector_speckle': self.vector_speckle_var.get(),
            'vector_color_precision': self.vector_color_precision_var.get(),
            'vector_corner_threshold': self.vector_corner_threshold_var.get(),
        }

    def set_all_settings(self, settings):
        try:
            self.carpet_width_var.set(settings.get('carpet_width', 200))
            self.carpet_height_var.set(settings.get('carpet_height', 300))
            self.shaneh_var.set(settings.get('shaneh', 50))
            self.tar_var.set(settings.get('tar', 12))
            self.remove_bg_var.set(settings.get('remove_bg', True))
            self.edge_detect_var.set(settings.get('edge_detect', True))
            self.ai_generate_var.set(settings.get('ai_generate', True))
            self.symmetry_var.set(settings.get('symmetry', True))
            self.quantize_var.set(settings.get('quantize', True))
            self.vectorize_var.set(settings.get('vectorize', False))
            self.is_full_design_var.set(settings.get('is_full_design', True))

            profile_name = settings.get('model_profile_name')
            if profile_name in self.model_profile_combo['values']:
                self.model_profile_var.set(profile_name)
                self._on_model_profile_selected(None)
                
                cn_name = settings.get('controlnet_name')
                if cn_name in self.controlnet_combo['values']:
                    self.controlnet_name_var.set(cn_name)
                    self._on_controlnet_selected(None)
            
            self.controlnet_scale_var.set(settings.get('controlnet_scale', 1.0))
            self.steps_var.set(settings.get('steps', 30))
            self.guidance_var.set(settings.get('guidance', 7.5))
            self.edge_method_var.set(settings.get('edge_method', 'HED'))
            self.sam_fast_mode_var.set(settings.get('sam_fast_mode', False))
            self.save_intermediate_var.set(settings.get('save_intermediate', True))
            self.palette_method_var.set(settings.get('palette_method', 'auto'))
            self.n_colors_var.set(settings.get('n_colors', 8))
            self.custom_palette = settings.get('custom_palette', [])
            self.vector_speckle_var.set(settings.get('vector_speckle', 4))
            self.vector_color_precision_var.set(settings.get('vector_color_precision', 6))
            self.vector_corner_threshold_var.set(settings.get('vector_corner_threshold', 60))
            
            self.update_colors_display()
            self.log("✅ تنظیمات با موفقیت از فایل بارگذاری شد.")
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در بارگذاری تنظیمات:\n{e}")
            self.log(f"❌ خطا در اعمال تنظیمات از فایل: {e}")
            
    def populate_profiles_combobox(self):
        profile_names = self.profile_manager.get_profile_names()
        self.profiles_combobox['values'] = profile_names
        if profile_names:
            self.profile_var.set(profile_names[0])
        else:
            self.profile_var.set("")

    def apply_selected_profile(self):
        profile_name = self.profile_var.get()
        if not profile_name:
            messagebox.showwarning("هشدار", "هیچ پروفایلی برای اعمال انتخاب نشده است.")
            return

        profile_data = self.profile_manager.get_profile(profile_name)
        if profile_data:
            self.shaneh_var.set(profile_data['shaneh'])
            self.tar_var.set(profile_data['tar'])
            self.custom_palette = [tuple(c) for c in profile_data['palette']]
            self.palette_method_var.set("custom")
            self.update_colors_display()
            self.log(f"✅ پروفایل '{profile_name}' با موفقیت اعمال شد.")
            self.update_status(f"پروفایل '{profile_name}' بارگذاری شد.")
        else:
            messagebox.showerror("خطا", f"پروفایل '{profile_name}' یافت نشد.")

    def save_new_profile(self):
        if self.palette_method_var.get() == 'auto' or not self.custom_palette:
            messagebox.showerror("خطا", "برای ذخیره پروفایل، باید یک پالت رنگی دستی (custom) داشته باشید.\n"
                                        "لطفاً ابتدا رنگ‌های مورد نظر را به پالت دستی اضافه یا استخراج کنید.")
            return

        profile_name = simpledialog.askstring("ذخیره پروفایل", "یک نام برای پروفایل جدید وارد کنید:", parent=self.root)
        
        if profile_name:
            if profile_name in self.profile_manager.get_profile_names():
                if not messagebox.askyesno("تایید", f"پروفایلی با نام '{profile_name}' از قبل وجود دارد. آیا می‌خواهید آن را بازنویسی کنید؟"):
                    return

            try:
                shaneh = self.shaneh_var.get()
                tar = self.tar_var.get()
                palette = self.custom_palette
                
                self.profile_manager.save_profile(profile_name, shaneh, tar, palette)
                self.log(f"💾 پروفایل جدید با نام '{profile_name}' ذخیره شد.")
                self.populate_profiles_combobox()
                self.profile_var.set(profile_name)
            except Exception as e:
                messagebox.showerror("خطا", f"خطا در ذخیره پروفایل: {e}")

    def delete_selected_profile(self):
        profile_name = self.profile_var.get()
        if not profile_name:
            messagebox.showwarning("هشدار", "هیچ پروفایلی برای حذف انتخاب نشده است.")
            return

        if messagebox.askyesno("تایید حذف", f"آیا از حذف پروفایل '{profile_name}' مطمئن هستید؟ این عمل غیرقابل بازگشت است."):
            if self.profile_manager.delete_profile(profile_name):
                self.log(f"🗑️ پروفایل '{profile_name}' با موفقیت حذف شد.")
                self.populate_profiles_combobox()
            else:
                messagebox.showerror("خطا", "خطا در حذف پروفایل.")

    def extract_palette_from_sample_file(self):
        path = filedialog.askopenfilename(
            title="انتخاب تصویر نمونه برای استخراج پالت",
            filetypes=[("تصاویر", "*.jpg *.jpeg *.png *.bmp"), ("همه فایل‌ها", "*.*")]
        )
        if not path:
            return

        try:
            sample_image = Image.open(path).convert('RGB')
            n_colors = self.n_colors_var.get()
            self.log(f"🔄 در حال استخراج {n_colors} رنگ از فایل نمونه: {os.path.basename(path)}...")
            self.update_status("در حال استخراج پالت از فایل نمونه...")

            quantizer = ColorQuantizer(n_colors=n_colors)
            palette = quantizer.extract_palette(sample_image)
            
            self.custom_palette = [tuple(color) for color in palette]
            self.palette_method_var.set("custom")
            self.update_colors_display()
            self.log(f"✅ {len(self.custom_palette)} رنگ با موفقیت از فایل نمونه استخراج شد.")
            self.update_status("پالت رنگی از فایل نمونه استخراج شد.")

        except Exception as e:
            messagebox.showerror("خطا", f"خطا در هنگام استخراج پالت از فایل نمونه:\n{e}")
            self.log(f"❌ خطا در استخراج پالت از نمونه: {e}")

    def update_density(self, *args):
        try:
            shaneh = self.shaneh_var.get()
            tar = self.tar_var.get()
            density = (shaneh / 10) * tar
            self.density_label.config(text=f"تراکم: {int(density)} گره/dm² | {int(density*100)} گره/m²")
        except (tk.TclError, ValueError):
            self.density_label.config(text="تراکم: ...")

    def add_color(self):
        color = colorchooser.askcolor(title="انتخاب رنگ")
        if color[0]:
            rgb = tuple(int(c) for c in color[0])
            if rgb not in self.custom_palette:
                self.custom_palette.append(rgb)
                self.palette_method_var.set("custom")
                self.update_colors_display()
    
    def clear_colors(self):
        if messagebox.askyesno("تایید", "آیا از پاک کردن تمام رنگ‌های پالت دستی مطمئن هستید؟"):
            self.custom_palette.clear()
            self.update_colors_display()

    def update_colors_display(self):
        for widget in self.colors_display_frame.winfo_children():
            widget.destroy()
        
        if not self.custom_palette:
            ttk.Label(self.colors_display_frame, text="رنگی به پالت دستی اضافه نشده است.\nاز دکمه‌های بالا برای افزودن یا استخراج رنگ استفاده کنید.").pack(pady=20)
            return
        
        canvas = tk.Canvas(self.colors_display_frame, borderwidth=0, background="#ffffff")
        scrollbar = ttk.Scrollbar(self.colors_display_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for i, color in enumerate(self.custom_palette):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2, padx=5)
            
            color_hex = '#%02x%02x%02x' % tuple(map(int, color))
            tk.Label(frame, bg=color_hex, width=5, relief=tk.SOLID, borderwidth=1).pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, text=f"RGB({color[0]}, {color[1]}, {color[2]})").pack(side=tk.LEFT, padx=5)
            ttk.Button(frame, text="✖", width=3, command=lambda idx=i: self.remove_color(idx)).pack(side=tk.RIGHT, padx=5)

    def remove_color(self, index):
        if 0 <= index < len(self.custom_palette):
            self.custom_palette.pop(index)
            self.update_colors_display()
    
    def preview_preset_palette(self):
        preset_key = self.preset_var.get()
        colors = self.palette_manager.get_palette(preset_key)
        preset_name = self.palette_manager.get_palette_name(preset_key)
        
        win = tk.Toplevel(self.root)
        win.title(f"پیش‌نمایش پالت: {preset_name}")
        win.geometry("600x200")
        win.transient(self.root)
        win.grab_set()

        canvas = tk.Canvas(win, bg='white')
        canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        n = len(colors)
        if n > 0:
            w = 500 // n
            for i, color in enumerate(colors):
                hex_color = '#%02x%02x%02x' % color
                x1 = 50 + i * w
                canvas.create_rectangle(x1, 50, x1+w, 120, fill=hex_color, outline='black')
                canvas.create_text(x1+w//2, 140, text=f"RGB\n{color[0]},{color[1]},{color[2]}", font=('Arial', 8))
        
        ttk.Button(win, text="بستن", command=win.destroy).pack(pady=10)

    def use_preset_palette(self):
        preset_key = self.preset_var.get()
        colors = self.palette_manager.get_palette(preset_key)
        preset_name = self.palette_manager.get_palette_name(preset_key)
        
        self.custom_palette = list(colors)
        self.palette_method_var.set("custom")
        self.update_colors_display()
        self.log(f"✅ پالت آماده '{preset_name}' با {len(colors)} رنگ در پالت دستی بارگذاری شد.")
        self.update_status(f"پالت '{preset_name}' انتخاب شد.")

    def extract_palette_from_image(self):
        if not self.input_image:
            messagebox.showwarning("هشدار", "ابتدا یک تصویر ورودی انتخاب کنید.")
            return
        
        try:
            n_colors = self.n_colors_var.get()
            self.log(f"🔄 در حال استخراج {n_colors} رنگ از تصویر ورودی...")
            self.update_status("در حال استخراج پالت رنگی...")
            quantizer = ColorQuantizer(n_colors=n_colors)
            palette = quantizer.extract_palette(self.input_image)
            
            self.custom_palette = [tuple(color) for color in palette]
            self.palette_method_var.set("custom")
            self.update_colors_display()
            self.log(f"✅ {len(self.custom_palette)} رنگ با موفقیت استخراج و در پالت دستی قرار گرفت.")
            self.update_status("پالت رنگی با موفقیت استخراج شد.")
        except Exception as e:
            messagebox.showerror("خطا", f"خطا در هنگام استخراج پالت رنگی:\n{e}")
            self.log(f"❌ خطا در استخراج پالت: {e}")

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def select_input_file(self):
        path = filedialog.askopenfilename(
            title="انتخاب تصویر ورودی",
            filetypes=[("تصاویر", "*.jpg *.jpeg *.png *.bmp"), ("همه فایل‌ها", "*.*")]
        )
        if path:
            self.preview_label.config(image='', text="در حال بارگذاری تصویر...")
            self.update_status(f"در حال باز کردن فایل: {os.path.basename(path)}...")
            self.root.update_idletasks()

            threading.Thread(target=self._load_image_thread, args=(path,), daemon=True).start()
    
    def _load_image_thread(self, path):
        try:
            image = Image.open(path).convert('RGB')
            self.root.after(0, self._finalize_image_loading, path, image)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("خطای باز کردن تصویر", f"امکان باز کردن فایل تصویر وجود ندارد.\nخطا: {e}"))
            self.root.after(0, self._reset_input_path)

    def _finalize_image_loading(self, path, image):
        self.input_image_path = path
        self.input_image = image
        self.input_path_var.set(path)

        self.log(f"✅ تصویر ورودی انتخاب شد: {os.path.basename(path)}")
        self.preview_button.config(state=tk.NORMAL)
        self.extract_button.config(state=tk.NORMAL)
        self.preview_input()
        self.update_status(f"تصویر {os.path.basename(path)} با موفقیت بارگذاری شد.")

    def _reset_input_path(self):
        self.input_image_path = None
        self.input_image = None
        self.input_path_var.set("")
        self.preview_button.config(state=tk.DISABLED)
        self.extract_button.config(state=tk.DISABLED)
        self.preview_label.config(image='', text="تصویری انتخاب نشده")
        self.update_status("خطا در بارگذاری تصویر. لطفاً فایل دیگری را امتحان کنید.")

    def preview_input(self):
        if not self.input_image:
            return
        try:
            image_copy = self.input_image.copy()
            thumbnail_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            image_copy.thumbnail((self.preview_label.winfo_width(), self.preview_label.winfo_height()), thumbnail_method)
            photo = ImageTk.PhotoImage(image_copy)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
        except Exception as e:
            messagebox.showerror("خطا در پیش‌نمایش", f"خطا در نمایش تصویر: {e}")
            
    def select_output_folder(self):
        path = filedialog.askdirectory(title="انتخاب پوشه خروجی")
        if path:
            self.output_path_var.set(path)
            self.log(f"📂 پوشه خروجی تنظیم شد: {path}")

    def open_output_folder(self):
        output_dir = self.output_path_var.get()
        if os.path.exists(output_dir):
            try:
                if sys.platform == 'win32':
                    os.startfile(output_dir)
                elif sys.platform == 'darwin':
                    os.system(f'open "{output_dir}"')
                else:
                    os.system(f'xdg-open "{output_dir}"')
            except Exception as e:
                messagebox.showerror("خطا", f"امکان باز کردن پوشه وجود ندارد.\n{e}")
        else:
            messagebox.showinfo("اطلاعات", "پوشه خروجی هنوز ایجاد نشده است. پس از اولین پردازش، این پوشه ساخته خواهد شد.")
    
    def start_processing(self):
        if not self.input_image:
            messagebox.showerror("خطا", "لطفاً ابتدا یک تصویر ورودی انتخاب کنید.")
            return
        
        if self.palette_method_var.get() == "custom" and not self.custom_palette:
            if not messagebox.askyesno("هشدار", "پالت رنگی دستی خالی است. آیا مایلید رنگ‌ها به صورت خودکار استخراج شوند?"):
                return
            self.palette_method_var.set("auto")
        
        if self.vectorize_var.get() and not shutil.which('vtracer'):
            messagebox.showerror("خطا: وابستگی خارجی",
                                 "قابلیت وکتورسازی انتخاب شده است، اما ابزار 'vtracer' یافت نشد.\n"
                                 "لطفاً آن را نصب کرده و در PATH سیستم قرار دهید، یا تیک وکتورسازی را بردارید.")
            return

        self.cancel_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.progress_bar.config(value=0)
        self.update_status("پردازش در حال آماده‌سازی...")
        
        self.processing_thread = threading.Thread(target=self.process_thread, daemon=True)
        self.processing_thread.start()

    def cancel_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.log("🛑 درخواست لغو پردازش ارسال شد... لطفاً منتظر بمانید.")
            self.update_status("در حال لغو پردازش...")
            self.cancel_event.set()
            self.cancel_button.config(state=tk.DISABLED)

    def process_thread(self):
        try:
            self.log("\n" + "="*80)
            self.log("🚀 پردازش آغاز شد...")
            self.log("="*80)
            
            if self.pipeline is None:
                self.log("⏳ در حال ساخت پایپلاین پردازش...")
                self.pipeline = CarpetDesignPipeline()

            self.apply_settings_to_pipeline()
            
            self.log("\n🎨 پردازش تصویر اصلی...")
            
            self.results = self.pipeline.process_image(
                input_image=self.input_image,
                output_dir=self.output_path_var.get(),
                run_config=self.get_run_config(),
                cancel_event=self.cancel_event,
                log_callback=self.log,
                progress_callback=self.update_progress_bar
            )
            
            if 'final_png' in self.results:
                self.root.after(0, lambda: self.show_result(self.results['final_png']))
            
            self.log("\n" + "="*80)
            self.log("✨ پردازش با موفقیت کامل شد!")
            self.log(f"📁 نتایج در پوشه زیر ذخیره شدند:\n{self.results.get('output_path', 'N/A')}")
            self.log("="*80)

            density = (self.shaneh_var.get() / 10) * self.tar_var.get()
            
            self.root.after(0, lambda: messagebox.showinfo(
                "موفقیت",
                f"پردازش با موفقیت انجام شد!\n\n"
                f"ابعاد فرش: {self.carpet_width_var.get()} × {self.carpet_height_var.get()} cm\n"
                f"تراکم: {self.shaneh_var.get()} شانه × {self.tar_var.get()} تار\n"
                f"تراکم کل: {int(density)} گره/dm²\n"
                f"تعداد رنگ‌ها: {len(self.custom_palette) if self.palette_method_var.get() == 'custom' else self.n_colors_var.get()}\n\n"
                f"نتایج در پوشه '{os.path.basename(self.results.get('output_path', '...'))}' ذخیره شد."
            ))

        except ProcessingCancelledError:
            self.log("\n" + "="*80)
            self.log("🛑 پردازش توسط کاربر لغو شد.")
            self.log("="*80)
            self.root.after(0, lambda: messagebox.showwarning("لغو شد", "عملیات پردازش توسط شما لغو شد."))
        
        except Exception as e:
            import traceback
            error_msg = f"یک خطای پیش‌بینی نشده رخ داد:\n{e}"
            self.log(f"\n❌ خطای بحرانی: {e}")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("خطای بحرانی", error_msg))
        finally:
            self.root.after(0, self.processing_finished)

    def apply_settings_to_pipeline(self):
        if not self.pipeline:
            return
        
        self.pipeline.config['processing']['edge_detection']['method'] = self.edge_method_var.get()
        self.pipeline.config['generation']['steps'] = self.steps_var.get()
        self.pipeline.config['generation']['guidance_scale'] = self.guidance_var.get()
        self.pipeline.config['generation']['controlnet_scale'] = self.controlnet_scale_var.get()
        
        if self.palette_method_var.get() == "custom" and self.custom_palette:
            self.pipeline.custom_palette = np.array(self.custom_palette)
        else:
            self.pipeline.custom_palette = None
        
        self.pipeline.carpet_specs = {
            'width_cm': self.carpet_width_var.get(),
            'height_cm': self.carpet_height_var.get(),
            'shaneh': self.shaneh_var.get(),
            'tar': self.tar_var.get()
        }
        
        self.log("⚙️ تنظیمات جدید اعمال شد:")
        self.log(f"   - ابعاد فرش: {self.carpet_width_var.get()}×{self.carpet_height_var.get()} cm")
        self.log(f"   - تراکم: {self.shaneh_var.get()} شانه × {self.tar_var.get()} تار")
        self.log(f"   - پروفایل مدل: {self.model_profile_var.get()}")

    def show_result(self, image_path):
        try:
            image = Image.open(image_path)
            thumbnail_method = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            image.thumbnail((self.preview_label.winfo_width(), self.preview_label.winfo_height()), thumbnail_method)
            photo = ImageTk.PhotoImage(image)
            self.preview_label.config(image=photo)
            self.preview_label.image = photo
        except Exception as e:
            self.log(f"❌ خطا در نمایش تصویر نهایی: {e}")
            messagebox.showwarning("خطای نمایش", f"امکان نمایش تصویر نهایی وجود نداشت.\n{e}")

    def update_progress_bar(self, current_step, total_steps):
        progress_percent = (current_step / total_steps) * 100
        self.progress_bar['value'] = progress_percent
        self.update_status(f"در حال انجام مرحله {current_step} از {total_steps}...")
        self.root.update_idletasks()

    def processing_finished(self):
        self.progress_bar['value'] = 0
        self.start_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        self.processing_thread = None
        self.update_status("آماده به کار...")

    def on_closing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if messagebox.askyesno("خروج", "پردازش در حال انجام است. آیا می‌خواهید آن را لغو کرده و خارج شوید؟"):
                self.cancel_processing()
                self.root.after(100, self.wait_for_thread_and_destroy)
        else:
            self.root.destroy()
            
    def wait_for_thread_and_destroy(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.root.after(100, self.wait_for_thread_and_destroy)
        else:
            self.root.destroy()

    def save_settings_to_file(self):
        filepath = filedialog.asksaveasfilename(
            title="ذخیره تنظیمات",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        settings = self.get_all_settings()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            self.log(f"💾 تنظیمات با موفقیت در فایل '{os.path.basename(filepath)}' ذخیره شد.")
        except Exception as e:
            messagebox.showerror("خطا", f"امکان ذخیره فایل تنظیمات وجود نداشت.\n{e}")

    def load_settings_from_file(self):
        filepath = filedialog.askopenfilename(
            title="بارگذاری تنظیمات",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            self.set_all_settings(settings)
        except Exception as e:
            messagebox.showerror("خطا", f"امکان خواندن فایل تنظیمات وجود نداشت.\n{e}")


def main():
    try:
        root = tk.Tk()
        app = CarpetDesignGUI(root)
        root.mainloop()
    except Exception as e:
        import traceback
        messagebox.showerror("خطای راه‌اندازی", f"برنامه با یک خطای غیرمنتظره مواجه شد و بسته خواهد شد.\n\n{e}\n\n{traceback.format_exc()}")

if __name__ == '__main__':
    main()