#!/usr/bin/env python3
"""
Generate placeholder application icons for nvCOMP GUI.
Creates simple PNG files with the nvCOMP logo text.
For production, replace with professionally designed icons.
"""

import struct
import zlib

def create_png_icon(size, filename):
    """Create a simple PNG icon with nvCOMP text."""
    
    # Create RGBA image data (simple gradient with text placeholder)
    pixels = []
    for y in range(size):
        for x in range(size):
            # Create a simple gradient background (green/blue theme)
            r = int(50 + (x / size) * 50)
            g = int(150 + (y / size) * 50)
            b = int(50 + ((x + y) / (size * 2)) * 100)
            a = 255
            pixels.extend([r, g, b, a])
    
    # Convert to bytes
    raw_data = bytes(pixels)
    
    # PNG file structure
    def write_chunk(chunk_type, data):
        chunk = struct.pack('>I', len(data))
        chunk += chunk_type
        chunk += data
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        chunk += struct.pack('>I', crc)
        return chunk
    
    # PNG signature
    png = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk (image header)
    ihdr = struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)
    png += write_chunk(b'IHDR', ihdr)
    
    # IDAT chunk (image data)
    # Add filter byte (0 = no filter) to each scanline
    filtered_data = b''
    for y in range(size):
        filtered_data += b'\x00'  # Filter type
        filtered_data += raw_data[y * size * 4:(y + 1) * size * 4]
    
    compressed = zlib.compress(filtered_data, 9)
    png += write_chunk(b'IDAT', compressed)
    
    # IEND chunk (end marker)
    png += write_chunk(b'IEND', b'')
    
    # Write file
    with open(filename, 'wb') as f:
        f.write(png)
    
    print(f"Created {filename} ({size}x{size})")

# Generate all required icon sizes
if __name__ == '__main__':
    import os
    
    # Ensure icons directory exists
    icons_dir = os.path.dirname(os.path.abspath(__file__)) + '/icons'
    os.makedirs(icons_dir, exist_ok=True)
    
    sizes = [16, 32, 48, 64, 128, 256]
    for size in sizes:
        create_png_icon(size, f'{icons_dir}/nvcomp_{size}.png')
    
    # Create default nvcomp.png (64x64)
    create_png_icon(64, f'{icons_dir}/nvcomp.png')
    
    print("\nIcon generation complete!")
    print("Note: These are placeholder icons. For production, replace with")
    print("professionally designed icons featuring the nvCOMP branding.")

