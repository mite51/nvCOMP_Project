#!/usr/bin/env python3
"""
Unit tests for nvCOMP Nautilus Extension

Tests the Nautilus extension functionality without requiring Nautilus to be running.
Uses mock objects to simulate Nautilus file objects and menu system.

Run with:
    python3 -m pytest test_nautilus_extension.py -v
    or
    python3 test_nautilus_extension.py
"""

import unittest
import sys
import os
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path

# Add current directory to path for importing the extension
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock the Nautilus bindings before importing the extension
sys.modules['gi'] = MagicMock()
sys.modules['gi.repository'] = MagicMock()
sys.modules['gi.repository.Nautilus'] = MagicMock()
sys.modules['gi.repository.GObject'] = MagicMock()
sys.modules['gi.repository.Gio'] = MagicMock()

# Now import the extension
import nautilus_extension


class TestNvcompMenuProvider(unittest.TestCase):
    """Test cases for NvcompMenuProvider class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock for GObject.GObject
        self.mock_gobject = MagicMock()
        nautilus_extension.GObject.GObject = self.mock_gobject
        
        # Create provider instance
        self.provider = nautilus_extension.NvcompMenuProvider()
    
    def test_initialization(self):
        """Test that provider initializes correctly"""
        self.assertIsNotNone(self.provider)
        self.assertIsInstance(self.provider.ALGORITHMS, list)
        self.assertIsInstance(self.provider.GPU_ALGORITHMS, list)
        self.assertIsInstance(self.provider.NVCOMP_EXTENSIONS, list)
    
    def test_find_nvcomp_gui(self):
        """Test finding nvcomp-gui executable"""
        with patch('subprocess.run') as mock_run:
            # Simulate 'which' finding the executable
            mock_run.return_value = Mock(returncode=0, stdout='/usr/bin/nvcomp-gui\n')
            
            provider = nautilus_extension.NvcompMenuProvider()
            result = provider._find_nvcomp_gui()
            
            self.assertIsNotNone(result)
            self.assertIn('nvcomp', result.lower())
    
    def test_check_cuda_available_success(self):
        """Test CUDA availability check when GPU is present"""
        with patch('subprocess.run') as mock_run:
            # Simulate successful nvidia-smi calls
            mock_run.side_effect = [
                Mock(returncode=0),  # which nvidia-smi
                Mock(returncode=0),  # nvidia-smi -L
            ]
            
            provider = nautilus_extension.NvcompMenuProvider()
            result = provider._check_cuda_available()
            
            self.assertTrue(result)
    
    def test_check_cuda_available_failure(self):
        """Test CUDA availability check when GPU is not present"""
        with patch('subprocess.run') as mock_run:
            # Simulate nvidia-smi not found
            mock_run.return_value = Mock(returncode=1)
            
            provider = nautilus_extension.NvcompMenuProvider()
            result = provider._check_cuda_available()
            
            self.assertFalse(result)
    
    def test_get_file_paths_with_file_uris(self):
        """Test converting file URIs to paths"""
        mock_file1 = Mock()
        mock_file1.get_uri.return_value = 'file:///home/user/test.txt'
        
        mock_file2 = Mock()
        mock_file2.get_uri.return_value = 'file:///home/user/folder/data.bin'
        
        paths = self.provider._get_file_paths([mock_file1, mock_file2])
        
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], '/home/user/test.txt')
        self.assertEqual(paths[1], '/home/user/folder/data.bin')
    
    def test_get_file_paths_with_special_characters(self):
        """Test converting URIs with special characters (spaces, etc.)"""
        mock_file = Mock()
        mock_file.get_uri.return_value = 'file:///home/user/My%20Documents/test%20file.txt'
        
        paths = self.provider._get_file_paths([mock_file])
        
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], '/home/user/My Documents/test file.txt')
    
    def test_is_compressible_regular_file(self):
        """Test that regular files are compressible"""
        self.assertTrue(self.provider._is_compressible('/home/user/document.txt'))
        self.assertTrue(self.provider._is_compressible('/home/user/data.bin'))
        self.assertTrue(self.provider._is_compressible('/home/user/image.png'))
    
    def test_is_compressible_nvcomp_archive(self):
        """Test that nvCOMP archives are not compressible"""
        self.assertFalse(self.provider._is_compressible('/home/user/archive.lz4'))
        self.assertFalse(self.provider._is_compressible('/home/user/data.zstd'))
        self.assertFalse(self.provider._is_compressible('/home/user/file.snappy'))
        self.assertFalse(self.provider._is_compressible('/home/user/test.nvcomp'))
    
    def test_is_nvcomp_archive(self):
        """Test detecting nvCOMP archive files"""
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.lz4'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.zstd'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.zst'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.snappy'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.nvcomp'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.gdeflate'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.ans'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.bitcomp'))
        
        self.assertFalse(self.provider._is_nvcomp_archive('/home/user/test.txt'))
        self.assertFalse(self.provider._is_nvcomp_archive('/home/user/test.zip'))
    
    def test_is_nvcomp_archive_case_insensitive(self):
        """Test that archive detection is case-insensitive"""
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.LZ4'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.ZSTD'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/test.Snappy'))
    
    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_run_nvcomp_gui_compress(self, mock_run, mock_popen):
        """Test launching nvCOMP GUI for compression"""
        # Set up provider with known path
        self.provider.nvcomp_gui_path = '/usr/bin/nvcomp-gui'
        
        files = ['/home/user/test.txt', '/home/user/data.bin']
        self.provider._run_nvcomp_gui(files, compress=True, algorithm='lz4')
        
        # Verify Popen was called with correct arguments
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        
        self.assertIn('/usr/bin/nvcomp-gui', call_args)
        self.assertIn('--compress', call_args)
        self.assertIn('--algorithm', call_args)
        self.assertIn('lz4', call_args)
        self.assertIn('--add-file', call_args)
        self.assertIn('/home/user/test.txt', call_args)
        self.assertIn('/home/user/data.bin', call_args)
    
    @patch('subprocess.Popen')
    def test_run_nvcomp_gui_no_compress(self, mock_popen):
        """Test launching nvCOMP GUI without compression"""
        self.provider.nvcomp_gui_path = '/usr/bin/nvcomp-gui'
        
        files = ['/home/user/archive.lz4']
        self.provider._run_nvcomp_gui(files, compress=False)
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        
        self.assertIn('/usr/bin/nvcomp-gui', call_args)
        self.assertNotIn('--compress', call_args)
        self.assertIn('--add-file', call_args)
        self.assertIn('/home/user/archive.lz4', call_args)
    
    @patch('subprocess.run')
    def test_run_nvcomp_gui_not_found(self, mock_run):
        """Test error handling when nvcomp-gui not found"""
        self.provider.nvcomp_gui_path = None
        
        files = ['/home/user/test.txt']
        
        # Should not raise exception, but should try to show error
        self.provider._run_nvcomp_gui(files, compress=True, algorithm='lz4')
        
        # Verify error notification was attempted
        mock_run.assert_called()
    
    def test_get_file_items_no_files(self):
        """Test that no menu items returned when no files selected"""
        result = self.provider.get_file_items([])
        self.assertEqual(result, [])
    
    def test_get_file_items_no_nvcomp(self):
        """Test that no menu items returned when nvcomp-gui not available"""
        self.provider.nvcomp_gui_path = None
        
        mock_file = Mock()
        mock_file.get_uri.return_value = 'file:///home/user/test.txt'
        
        result = self.provider.get_file_items([mock_file])
        self.assertEqual(result, [])
    
    def test_algorithms_defined(self):
        """Test that all required algorithms are defined"""
        algo_ids = [algo[0] for algo in self.provider.ALGORITHMS]
        
        self.assertIn('lz4', algo_ids)
        self.assertIn('zstd', algo_ids)
        self.assertIn('snappy', algo_ids)
    
    def test_gpu_algorithms_defined(self):
        """Test that GPU algorithms are defined"""
        gpu_algo_ids = [algo[0] for algo in self.provider.GPU_ALGORITHMS]
        
        self.assertIn('gdeflate', gpu_algo_ids)
        self.assertIn('ans', gpu_algo_ids)
        self.assertIn('bitcomp', gpu_algo_ids)
    
    def test_extensions_defined(self):
        """Test that all supported extensions are defined"""
        extensions = self.provider.NVCOMP_EXTENSIONS
        
        self.assertIn('.lz4', extensions)
        self.assertIn('.zstd', extensions)
        self.assertIn('.zst', extensions)
        self.assertIn('.snappy', extensions)
        self.assertIn('.nvcomp', extensions)
        self.assertIn('.gdeflate', extensions)
        self.assertIn('.ans', extensions)
        self.assertIn('.bitcomp', extensions)
    
    @patch('subprocess.run')
    def test_send_notification(self, mock_run):
        """Test sending desktop notification"""
        self.provider._send_notification("Test Title", "Test Message")
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        
        self.assertIn('notify-send', call_args)
        self.assertIn('Test Title', call_args)
        self.assertIn('Test Message', call_args)
    
    @patch('subprocess.run')
    def test_show_error(self, mock_run):
        """Test showing error notification"""
        self.provider._show_error("Error Title", "Error Message")
        
        # Should call notify-send
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        
        self.assertIn('notify-send', call_args)
        self.assertIn('Error Title', call_args)


class TestNautilusIntegration(unittest.TestCase):
    """Integration tests for Nautilus extension"""
    
    def test_module_imports(self):
        """Test that extension module imports without errors"""
        self.assertIsNotNone(nautilus_extension)
        self.assertTrue(hasattr(nautilus_extension, 'NvcompMenuProvider'))
    
    def test_provider_has_required_methods(self):
        """Test that provider implements required Nautilus methods"""
        provider = nautilus_extension.NvcompMenuProvider()
        
        self.assertTrue(hasattr(provider, 'get_file_items'))
        self.assertTrue(callable(provider.get_file_items))
        
        self.assertTrue(hasattr(provider, 'get_background_items'))
        self.assertTrue(callable(provider.get_background_items))
    
    def test_nautilus_version_detection(self):
        """Test that Nautilus version is detected"""
        self.assertIn('NAUTILUS_VERSION', dir(nautilus_extension))
        version = nautilus_extension.NAUTILUS_VERSION
        self.assertIn(version, [3, 4])


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions and edge cases"""
    
    def setUp(self):
        self.provider = nautilus_extension.NvcompMenuProvider()
    
    def test_empty_file_paths(self):
        """Test handling of empty file path list"""
        result = self.provider._get_file_paths([])
        self.assertEqual(result, [])
    
    def test_file_with_no_extension(self):
        """Test handling of files without extensions"""
        # File without extension should be compressible
        self.assertTrue(self.provider._is_compressible('/home/user/README'))
        
        # Should not be detected as nvCOMP archive
        self.assertFalse(self.provider._is_nvcomp_archive('/home/user/README'))
    
    def test_hidden_files(self):
        """Test handling of hidden files (starting with dot)"""
        self.assertTrue(self.provider._is_compressible('/home/user/.config'))
        self.assertFalse(self.provider._is_compressible('/home/user/.cache.lz4'))
    
    def test_multi_extension_files(self):
        """Test handling of files with multiple extensions"""
        # .tar.lz4 should be detected as nvCOMP archive
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/backup.tar.lz4'))
        self.assertFalse(self.provider._is_compressible('/home/user/backup.tar.lz4'))
    
    def test_volume_files(self):
        """Test handling of multi-volume archives"""
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/data.vol001.lz4'))
        self.assertTrue(self.provider._is_nvcomp_archive('/home/user/data.vol002.lz4'))
        self.assertFalse(self.provider._is_compressible('/home/user/data.vol001.lz4'))


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNvcompMenuProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestNautilusIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())

