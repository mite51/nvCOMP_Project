#!/usr/bin/env python3
"""
nvCOMP Nautilus Extension

Adds right-click context menu integration for Nautilus (GNOME Files) file manager.
Provides quick access to nvCOMP compression and decompression operations.

Installation:
  User:   cp nautilus_extension.py ~/.local/share/nautilus-python/extensions/nvcomp_extension.py
  System: cp nautilus_extension.py /usr/share/nautilus-python/extensions/nvcomp_extension.py

Requirements:
  - python3-nautilus package
  - nvcomp-gui in PATH or at known location

After installation, restart Nautilus:
  nautilus -q
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse

try:
    from gi import require_version
    require_version('Nautilus', '4.0')
    from gi.repository import Nautilus, GObject, Gio
    NAUTILUS_VERSION = 4
except (ValueError, ImportError):
    try:
        require_version('Nautilus', '3.0')
        from gi.repository import Nautilus, GObject, Gio
        NAUTILUS_VERSION = 3
    except (ValueError, ImportError):
        print("ERROR: python3-nautilus (Nautilus Python bindings) not installed", file=sys.stderr)
        print("Install with: sudo apt-get install python3-nautilus", file=sys.stderr)
        sys.exit(1)


class NvcompMenuProvider(GObject.GObject, Nautilus.MenuProvider):
    """
    Nautilus menu provider for nvCOMP compression operations.
    
    Adds context menu items for:
    - Compressing files/folders with different algorithms
    - Decompressing nvCOMP archives
    - Opening archives in nvCOMP GUI
    """
    
    __gtype_name__ = 'NvcompMenuProvider'
    
    # Supported compression algorithms
    ALGORITHMS = [
        ('lz4', 'LZ4 (Fast, Good Ratio)', 'Fast compression with good compression ratio'),
        ('zstd', 'Zstd (Best Ratio)', 'Best compression ratio, slower'),
        ('snappy', 'Snappy (Fastest)', 'Very fast compression, lower ratio'),
    ]
    
    # GPU algorithms (shown only if CUDA available)
    GPU_ALGORITHMS = [
        ('gdeflate', 'GDeflate (GPU)', 'GPU-optimized DEFLATE'),
        ('ans', 'ANS (GPU)', 'Asymmetric Numeral Systems'),
        ('bitcomp', 'Bitcomp (GPU)', 'Numerical data compression'),
    ]
    
    # nvCOMP archive extensions
    NVCOMP_EXTENSIONS = [
        '.lz4', '.zstd', '.zst', '.snappy', 
        '.nvcomp', '.gdeflate', '.ans', '.bitcomp'
    ]
    
    def __init__(self):
        super().__init__()
        self.nvcomp_gui_path = self._find_nvcomp_gui()
        self.cuda_available = self._check_cuda_available()
    
    def _find_nvcomp_gui(self) -> Optional[str]:
        """Find nvcomp-gui executable."""
        # Check common locations
        search_paths = [
            'nvcomp_gui',  # In PATH (installed name)
            'nvcomp-gui',  # Alternative name
            os.path.expanduser('~/Dev/nvCOMP_Project/build/gui/nvcomp_gui'),
            os.path.expanduser('~/.local/bin/nvcomp-gui'),
            os.path.expanduser('~/.local/bin/nvcomp_gui'),
            '/usr/local/bin/nvcomp-gui',
            '/usr/local/bin/nvcomp_gui',
            '/usr/bin/nvcomp-gui',
            '/usr/bin/nvcomp_gui',  # Debian package install location
        ]
        
        for path in search_paths:
            # Check if in PATH
            if '/' not in path:
                try:
                    result = subprocess.run(['which', path], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=1)
                    if result.returncode == 0 and result.stdout.strip():
                        return result.stdout.strip()
                except Exception:
                    pass
            # Check if file exists
            elif os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available on the system."""
        # Check for nvidia-smi
        try:
            result = subprocess.run(['which', 'nvidia-smi'], 
                                  capture_output=True, 
                                  timeout=1)
            if result.returncode == 0:
                # Verify GPU is accessible
                result = subprocess.run(['nvidia-smi', '-L'],
                                      capture_output=True,
                                      timeout=2)
                return result.returncode == 0
        except Exception:
            pass
        
        return False
    
    def _get_file_paths(self, files: List) -> List[str]:
        """Convert Nautilus file objects to filesystem paths."""
        paths = []
        for file_info in files:
            if hasattr(file_info, 'get_uri'):
                uri = file_info.get_uri()
            else:
                uri = file_info
            
            # Convert URI to path
            parsed = urlparse(uri)
            if parsed.scheme == 'file':
                path = unquote(parsed.path)
                paths.append(path)
        
        return paths
    
    def _is_compressible(self, path: str) -> bool:
        """Check if file/folder can be compressed."""
        # Don't compress already compressed nvCOMP files
        path_lower = path.lower()
        for ext in self.NVCOMP_EXTENSIONS:
            if path_lower.endswith(ext):
                return False
        return True
    
    def _is_nvcomp_archive(self, path: str) -> bool:
        """Check if file is an nvCOMP archive."""
        path_lower = path.lower()
        for ext in self.NVCOMP_EXTENSIONS:
            if path_lower.endswith(ext):
                return True
        return False
    
    def _run_nvcomp_gui(self, files: List[str], compress: bool = False, 
                       algorithm: Optional[str] = None) -> None:
        """Launch nvCOMP GUI with files."""
        if not self.nvcomp_gui_path:
            self._show_error("nvCOMP GUI not found", 
                           "Please install nvCOMP or add it to your PATH")
            return
        
        cmd = [self.nvcomp_gui_path]
        
        # Add command-line options
        if compress and algorithm:
            cmd.extend(['--compress', '--algorithm', algorithm])
        
        # Add files
        for file_path in files:
            cmd.extend(['--add-file', file_path])
        
        # Launch in background
        try:
            subprocess.Popen(cmd, 
                           start_new_session=True,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            
            # Send notification
            self._send_notification(
                "nvCOMP" if not compress else f"Compressing with {algorithm.upper()}",
                f"Processing {len(files)} item(s)..."
            )
        except Exception as e:
            self._show_error("Failed to launch nvCOMP", str(e))
    
    def _extract_here(self, files: List[str]) -> None:
        """Extract archives to current directory."""
        # For now, open in GUI for extraction
        # In future, could add direct extraction support
        self._run_nvcomp_gui(files, compress=False)
    
    def _extract_to_folder(self, files: List[str]) -> None:
        """Extract archives to new folder."""
        # Open in GUI, user can choose extraction location
        self._run_nvcomp_gui(files, compress=False)
    
    def _show_error(self, title: str, message: str) -> None:
        """Show error notification."""
        try:
            subprocess.run([
                'notify-send',
                '-i', 'dialog-error',
                '-u', 'normal',
                title,
                message
            ], timeout=1)
        except Exception:
            pass
    
    def _send_notification(self, title: str, message: str) -> None:
        """Send desktop notification."""
        try:
            subprocess.run([
                'notify-send',
                '-i', 'nvcomp',  # Use nvcomp icon if available
                '-u', 'low',
                title,
                message
            ], timeout=1)
        except Exception:
            pass
    
    def get_file_items(self, *args) -> List:
        """
        Return menu items for selected files.
        Called by Nautilus when user right-clicks on files.
        """
        # Handle different Nautilus API versions
        if NAUTILUS_VERSION >= 4:
            files = args[0] if args else []
        else:
            window, files = args[0], args[1] if len(args) > 1 else []
        
        if not files:
            return []
        
        # Check if nvCOMP is available
        if not self.nvcomp_gui_path:
            return []
        
        # Get file paths
        paths = self._get_file_paths(files)
        if not paths:
            return []
        
        # Determine what menu items to show
        has_compressible = any(self._is_compressible(p) for p in paths)
        has_archives = any(self._is_nvcomp_archive(p) for p in paths)
        
        menu_items = []
        
        # Compression menu
        if has_compressible:
            menu_items.extend(self._create_compression_menu(files))
        
        # Decompression menu
        if has_archives:
            menu_items.extend(self._create_decompression_menu(files))
        
        return menu_items
    
    def get_background_items(self, *args) -> List:
        """
        Return menu items for folder background.
        Called when user right-clicks on empty space in folder.
        """
        # Not currently used, but could add "Compress all files in folder" option
        return []
    
    def _create_compression_menu(self, files: List) -> List:
        """Create compression submenu."""
        paths = self._get_file_paths(files)
        
        # Create top-level menu item
        top_item = Nautilus.MenuItem(
            name='NvcompMenuProvider::CompressMenu',
            label='Compress with nvCOMP',
            tip='Compress files using GPU-accelerated compression',
            icon='nvcomp'
        )
        
        # Create submenu
        submenu = Nautilus.Menu()
        top_item.set_submenu(submenu)
        
        # Add algorithm options
        for algo_id, algo_name, algo_desc in self.ALGORITHMS:
            item = Nautilus.MenuItem(
                name=f'NvcompMenuProvider::Compress_{algo_id}',
                label=algo_name,
                tip=algo_desc
            )
            item.connect('activate', self._on_compress, files, algo_id)
            submenu.append_item(item)
        
        # Add GPU algorithms if CUDA available
        if self.cuda_available:
            # Separator
            separator = Nautilus.MenuItem(
                name='NvcompMenuProvider::Separator1',
                label='─' * 30,
                sensitive=False
            )
            submenu.append_item(separator)
            
            for algo_id, algo_name, algo_desc in self.GPU_ALGORITHMS:
                item = Nautilus.MenuItem(
                    name=f'NvcompMenuProvider::Compress_{algo_id}',
                    label=algo_name,
                    tip=algo_desc
                )
                item.connect('activate', self._on_compress, files, algo_id)
                submenu.append_item(item)
        
        # Separator
        separator = Nautilus.MenuItem(
            name='NvcompMenuProvider::Separator2',
            label='─' * 30,
            sensitive=False
        )
        submenu.append_item(separator)
        
        # Settings option
        settings_item = Nautilus.MenuItem(
            name='NvcompMenuProvider::Settings',
            label='Open nvCOMP GUI...',
            tip='Open nvCOMP GUI for more options'
        )
        settings_item.connect('activate', self._on_open_gui, files)
        submenu.append_item(settings_item)
        
        return [top_item]
    
    def _create_decompression_menu(self, files: List) -> List:
        """Create decompression menu items."""
        paths = self._get_file_paths(files)
        
        menu_items = []
        
        # Extract here
        item = Nautilus.MenuItem(
            name='NvcompMenuProvider::ExtractHere',
            label='Extract Here',
            tip='Extract archive to current directory'
        )
        item.connect('activate', self._on_extract_here, files)
        menu_items.append(item)
        
        # Extract to folder
        item = Nautilus.MenuItem(
            name='NvcompMenuProvider::ExtractToFolder',
            label='Extract to Folder...',
            tip='Extract archive to a new folder'
        )
        item.connect('activate', self._on_extract_to_folder, files)
        menu_items.append(item)
        
        # View archive
        item = Nautilus.MenuItem(
            name='NvcompMenuProvider::ViewArchive',
            label='View Archive',
            tip='View archive contents in nvCOMP'
        )
        item.connect('activate', self._on_open_gui, files)
        menu_items.append(item)
        
        return menu_items
    
    # Menu item callbacks
    
    def _on_compress(self, menu, files: List, algorithm: str) -> None:
        """Callback for compression menu items."""
        paths = self._get_file_paths(files)
        self._run_nvcomp_gui(paths, compress=True, algorithm=algorithm)
    
    def _on_extract_here(self, menu, files: List) -> None:
        """Callback for 'Extract Here' menu item."""
        paths = self._get_file_paths(files)
        self._extract_here(paths)
    
    def _on_extract_to_folder(self, menu, files: List) -> None:
        """Callback for 'Extract to Folder' menu item."""
        paths = self._get_file_paths(files)
        self._extract_to_folder(paths)
    
    def _on_open_gui(self, menu, files: List) -> None:
        """Callback for 'Open GUI' menu item."""
        paths = self._get_file_paths(files)
        self._run_nvcomp_gui(paths, compress=False)


# Nautilus will import this module and look for classes implementing MenuProvider
# This is the entry point for the extension

