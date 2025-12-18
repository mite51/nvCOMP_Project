#!/usr/bin/env python3
"""
Generate algorithm-specific icons for nvCOMP archive types
Creates icons showing the compression algorithm visually with different colors
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_archive_icon(algorithm, color_bg, color_accent, icon_index, output_dir="."):
    """
    Create an archive icon for a specific compression algorithm
    
    Args:
        algorithm: Algorithm name (e.g., "LZ4", "Zstd", "Snappy")
        color_bg: Background color tuple (R, G, B)
        color_accent: Accent color tuple (R, G, B)
        icon_index: Icon index number for embedding
        output_dir: Directory to save icons
    """
    
    sizes = [16, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        # Create image with transparency
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        white = (255, 255, 255)
        dark = (50, 50, 50)
        
        # Draw document/folder shape (archive representation)
        margin = size // 10
        
        # Main rectangle (document body)
        doc_left = margin
        doc_top = margin * 2
        doc_right = size - margin
        doc_bottom = size - margin
        
        # Draw document with folded corner
        fold_size = size // 5
        
        # Main document body
        points = [
            (doc_left, doc_top),
            (doc_right - fold_size, doc_top),
            (doc_right, doc_top + fold_size),
            (doc_right, doc_bottom),
            (doc_left, doc_bottom)
        ]
        draw.polygon(points, fill=color_bg, outline=color_accent, width=max(1, size // 32))
        
        # Folded corner
        fold_points = [
            (doc_right - fold_size, doc_top),
            (doc_right - fold_size, doc_top + fold_size),
            (doc_right, doc_top + fold_size)
        ]
        draw.polygon(fold_points, fill=color_accent, outline=color_accent)
        
        # Draw compression bars (visual indicator)
        bar_width = (doc_right - doc_left - margin * 2)
        bar_height = max(2, size // 24)
        bar_spacing = max(3, size // 16)
        
        num_bars = 3
        start_y = (doc_bottom - doc_top - num_bars * bar_height - (num_bars - 1) * bar_spacing) // 2 + doc_top
        
        for i in range(num_bars):
            bar_y = start_y + i * (bar_height + bar_spacing)
            bar_x = doc_left + margin
            bar_length = bar_width * (0.8 - i * 0.15)  # Decreasing lengths
            
            draw.rectangle(
                [(bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height)],
                fill=color_accent
            )
        
        # Draw algorithm text
        text = algorithm.upper()
        
        # Calculate font size
        if size >= 64:
            font_size = max(8, size // 8)
        elif size >= 32:
            font_size = max(6, size // 10)
        else:
            font_size = max(5, size // 12)
        
        try:
            # Try to use a nice bold font
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                # Fallback to default
                font = ImageFont.load_default()
        
        # Get text bounding box
        if size >= 32:  # Only draw text for larger sizes
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position text at bottom
            text_x = (size - text_width) // 2
            text_y = doc_bottom - text_height - margin
            
            # Draw text with shadow
            shadow_offset = max(1, size // 64)
            if size >= 48:
                draw.text((text_x + shadow_offset, text_y + shadow_offset), text, fill=(0, 0, 0, 180), font=font)
            draw.text((text_x, text_y), text, fill=white, font=font)
        
        # Save PNG
        filename = f"{output_dir}/nvcomp_{algorithm.lower()}_{size}.png"
        img.save(filename)
        print(f"Created {filename}")
        
        images.append(img)
    
    # Create ICO file with multiple sizes embedded
    # Pillow's ICO format supports multiple sizes via append_images
    ico_filename = f"{output_dir}/nvcomp_{algorithm.lower()}.ico"
    
    # Convert all images to RGBA mode to ensure proper color depth
    rgba_images = [img.convert('RGBA') if img.mode != 'RGBA' else img for img in images]
    
    # Save all sizes to the ICO file (no compression for best quality)
    # Start with the largest image for better quality
    rgba_images[-1].save(
        ico_filename,
        format='ICO',
        sizes=[img.size for img in rgba_images],  # Explicit size list
        append_images=rgba_images[:-1],  # All other sizes
        bitmap_format='bmp'  # Use BMP format (uncompressed) for better quality
    )
    
    file_size = os.path.getsize(ico_filename)
    print(f"Created {ico_filename} with {len(images)} sizes - {file_size/1024:.1f} KB (icon index: {icon_index})")
    
    return images

def create_all_icons(output_dir="."):
    """Create icons for all supported compression algorithms"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Algorithm definitions: (name, bg_color, accent_color, icon_index)
    algorithms = [
        # Main nvCOMP icon (green)
        ("nvcomp", (34, 139, 34), (50, 205, 50), 0),
        
        # LZ4 (blue)
        ("lz4", (30, 100, 180), (70, 150, 255), 1),
        
        # Zstd (purple)
        ("zstd", (120, 50, 180), (180, 100, 255), 2),
        
        # Snappy (orange)
        ("snappy", (200, 100, 30), (255, 150, 50), 3),
        
        # GDeflate (teal)
        ("gdeflate", (30, 150, 150), (50, 200, 200), 4),
        
        # ANS (red)
        ("ans", (180, 50, 50), (255, 100, 100), 5),
        
        # Bitcomp (yellow/gold)
        ("bitcomp", (180, 140, 30), (255, 200, 50), 6),
    ]
    
    print("Generating icons for all compression algorithms...")
    print("=" * 60)
    
    main_icon_created = False
    for name, bg_color, accent_color, icon_index in algorithms:
        print(f"\nGenerating {name.upper()} icons...")
        create_archive_icon(name, bg_color, accent_color, icon_index, output_dir)
        
        # Copy nvcomp_nvcomp.ico to nvcomp.ico for main application icon
        if name == "nvcomp" and not main_icon_created:
            import shutil
            src = f"{output_dir}/nvcomp_nvcomp.ico"
            dst = f"{output_dir}/nvcomp.ico"
            shutil.copy2(src, dst)
            print(f"Copied {src} -> {dst} (main application icon)")
            main_icon_created = True
    
    print("\n" + "=" * 60)
    print("All icons generated successfully!")
    print(f"\nIcons saved to: {os.path.abspath(output_dir)}")
    print("\nIcon indices for registry:")
    for name, _, _, icon_index in algorithms:
        print(f"  {name.upper():12} -> Icon index {icon_index}")
    
    print("\nNote: nvcomp.ico is a copy of nvcomp_nvcomp.ico (main app icon)")

def create_main_icon_only(output_dir="."):
    """Create only the main nvCOMP icon (for quick regeneration)"""
    os.makedirs(output_dir, exist_ok=True)
    create_archive_icon("nvcomp", (34, 139, 34), (50, 205, 50), 0, output_dir)

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "."
    
    print("nvCOMP Archive Icon Generator")
    print("=" * 60)
    
    create_all_icons(output_dir)
    
    print("\nUsage notes:")
    print("- Icons are embedded in nvcomp_gui.exe during build")
    print("- Each algorithm has its own unique color scheme")
    print("- Icon indices 0-6 are used in file associations")
    print("- To regenerate, run: python generate_icons.py [output_dir]")

