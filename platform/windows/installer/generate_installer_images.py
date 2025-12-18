#!/usr/bin/env python3
"""
generate_installer_images.py - Generate placeholder installer images

This script creates the banner and dialog images required by the WiX installer.
You should replace these with professional branded images before release.

Requirements:
    pip install pillow

Usage:
    python generate_installer_images.py
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_banner(output_path="banner.bmp"):
    """Create installer banner image (493x58 pixels)"""
    width, height = 493, 58
    
    # Create gradient background
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Gradient from dark blue to lighter blue
    for y in range(height):
        ratio = y / height
        r = int(25 + (76 - 25) * ratio)
        g = int(25 + (175 - 25) * ratio)
        b = int(51 + (80 - 51) * ratio)
        draw.rectangle([(0, y), (width, y+1)], fill=(r, g, b))
    
    # Add text
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw title
    title = "nvCOMP"
    draw.text((20, 15), title, fill=(255, 255, 255), font=font)
    
    # Draw subtitle
    subtitle = "NVIDIA CUDA Compression Tools"
    draw.text((145, 22), subtitle, fill=(200, 220, 255), font=font_small)
    
    # Save as BMP
    img.save(output_path, 'BMP')
    print(f"Created: {output_path}")

def create_dialog(output_path="dialog.bmp"):
    """Create installer dialog image (493x312 pixels)"""
    width, height = 493, 312
    
    # Create gradient background
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    # Vertical gradient
    for x in range(width):
        ratio = x / width
        r = int(25 + (50 - 25) * ratio)
        g = int(25 + (100 - 25) * ratio)
        b = int(51 + (150 - 51) * ratio)
        draw.rectangle([(x, 0), (x+1, height)], fill=(r, g, b))
    
    # Add some decorative elements
    # Draw circuit-like pattern (representing GPU/compression)
    for i in range(0, height, 40):
        draw.line([(10, i), (width-10, i)], fill=(50, 120, 180), width=1)
    
    for i in range(0, width, 40):
        draw.line([(i, 10), (i, height-10)], fill=(50, 120, 180), width=1)
    
    # Save as BMP
    img.save(output_path, 'BMP')
    print(f"Created: {output_path}")

def create_icon_placeholder():
    """Create a simple icon if needed"""
    size = 256
    img = Image.new('RGB', (size, size), color=(25, 25, 51))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple compression icon (two arrows)
    # Arrow down (compress)
    draw.polygon([(128, 80), (128, 140), (90, 140), (128, 180), (166, 140), (128, 140)], 
                 fill=(76, 175, 80))
    
    # Text
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()
    
    draw.text((80, 200), "nvCOMP", fill=(255, 255, 255), font=font)
    
    img.save("nvcomp_installer_icon.png", 'PNG')
    print(f"Created: nvcomp_installer_icon.png")

if __name__ == "__main__":
    print("Generating WiX installer images...")
    print()
    
    # Change to installer directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    create_banner()
    create_dialog()
    create_icon_placeholder()
    
    print()
    print("Done! Images created successfully.")
    print()
    print("NOTE: These are placeholder images.")
    print("Replace them with professional branded images before release!")
    print()
    print("Banner requirements:")
    print("  - Size: 493 x 58 pixels")
    print("  - Format: 24-bit BMP")
    print("  - Content: Product name, logo, tagline")
    print()
    print("Dialog requirements:")
    print("  - Size: 493 x 312 pixels")
    print("  - Format: 24-bit BMP")
    print("  - Content: Branding, product imagery")

