#!/usr/bin/env python3
"""
Create a placeholder green icon for nvCOMP application
Creates both PNG and ICO formats
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_green_icon():
    """Create a green compression icon with 'nv' text"""
    
    sizes = [16, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        # Create image with green background
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Green color scheme
        green_bg = (34, 139, 34)  # Forest green
        green_light = (50, 205, 50)  # Lime green
        white = (255, 255, 255)
        
        # Draw rounded rectangle background
        margin = size // 10
        draw.rounded_rectangle(
            [(margin, margin), (size - margin, size - margin)],
            radius=size // 8,
            fill=green_bg,
            outline=green_light,
            width=max(1, size // 32)
        )
        
        # Draw "nv" text
        font_size = int(size * 0.35)
        try:
            # Try to use a nice font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback to default
            font = ImageFont.load_default()
        
        text = "nv"
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]
        
        # Draw text with shadow
        shadow_offset = max(1, size // 64)
        draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0, 128), font=font)
        draw.text((x, y), text, fill=white, font=font)
        
        # Save PNG
        img.save(f'nvcomp_{size}.png')
        print(f"Created nvcomp_{size}.png")
        
        images.append(img)
    
    # Create ICO file with multiple sizes
    images[0].save(
        'nvcomp.ico',
        format='ICO',
        sizes=[(s, s) for s in sizes]
    )
    print("Created nvcomp.ico")
    
    # Create a larger PNG for the main app icon
    main_size = 256
    main_img = Image.new('RGBA', (main_size, main_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(main_img)
    
    margin = main_size // 10
    draw.rounded_rectangle(
        [(margin, margin), (main_size - margin, main_size - margin)],
        radius=main_size // 8,
        fill=green_bg,
        outline=green_light,
        width=main_size // 32
    )
    
    # Larger text
    font_size = int(main_size * 0.35)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = "nv"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (main_size - text_width) // 2
    y = (main_size - text_height) // 2 - bbox[1]
    
    shadow_offset = main_size // 64
    draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0, 128), font=font)
    draw.text((x, y), text, fill=white, font=font)
    
    main_img.save('nvcomp_main.png')
    print("Created nvcomp_main.png")

if __name__ == '__main__':
    print("Creating nvCOMP icons...")
    create_green_icon()
    print("Done! Icons created in current directory.")
    print("\nYou can replace these with custom icons later.")

