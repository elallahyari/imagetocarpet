import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_comparison_grid(images, titles=None, rows=2, cols=3, figsize=(15, 10)):
    """
    ایجاد گرید مقایسه تصاویر
    
    Args:
        images: لیست تصاویر
        titles: لیست عناوین
        rows: تعداد سطرها
        cols: تعداد ستون‌ها
        figsize: اندازه شکل
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        ax.imshow(img)
        ax.axis('off')
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
    
    # پنهان کردن محورهای اضافی
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def save_comparison_grid(images, titles, output_path, **kwargs):
    """
    ذخیره گرید مقایسه
    
    Args:
        images: لیست تصاویر
        titles: لیست عناوین
        output_path: مسیر خروجی
        **kwargs: پارامترهای اضافی
    """
    fig = create_comparison_grid(images, titles, **kwargs)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ گرید مقایسه ذخیره شد: {output_path}")

def calculate_image_stats(image):
    """
    محاسبه آمار تصویر
    
    Args:
        image: تصویر ورودی
        
    Returns:
        dict: آمار تصویر
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
    }
    
    if len(image.shape) == 3:
        stats['channels'] = image.shape[2]
        for i, channel in enumerate(['R', 'G', 'B'][:image.shape[2]]):
            stats[f'{channel}_mean'] = float(np.mean(image[:, :, i]))
    
    return stats

def save_processing_report(results, output_path):
    """
    ذخیره گزارش پردازش
    
    Args:
        results: نتایج پردازش
        output_path: مسیر فایل خروجی
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for key, value in results.items():
        if isinstance(value, Image.Image):
            report['results'][key] = {
                'type': 'image',
                'size': value.size,
                'mode': value.mode
            }
        elif isinstance(value, str):
            report['results'][key] = {
                'type': 'file',
                'path': value
            }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ گزارش ذخیره شد: {output_path}")

def resize_to_standard(image, max_size=1024):
    """
    تغییر اندازه به سایز استاندارد
    
    Args:
        image: تصویر ورودی
        max_size: حداکثر اندازه
        
    Returns:
        PIL Image: تصویر تغییر اندازه داده شده
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # محاسبه نسبت
    ratio = min(max_size / image.width, max_size / image.height)
    
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    return image