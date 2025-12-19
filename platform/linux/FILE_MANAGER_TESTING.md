# File Manager Integration - Manual Testing Checklist

This document provides comprehensive manual testing procedures for nvCOMP file manager context menu integration on Linux.

## Test Environment Setup

### Prerequisites Checklist

- [ ] Ubuntu/Debian-based Linux distribution (20.04, 22.04, or 24.04)
- [ ] GNOME desktop (Nautilus) OR Cinnamon desktop (Nemo)
- [ ] nvcomp-gui built and accessible
- [ ] python3-nautilus package installed (for Nautilus only)
- [ ] notify-send available (libnotify-bin package)
- [ ] NVIDIA GPU with CUDA (optional, for GPU algorithm testing)

### Installation Verification

- [ ] Run installation script: `./install_file_manager_integration.sh`
- [ ] Verify installation completed without errors
- [ ] Check files installed to `~/.local/share/nautilus-python/extensions/`
- [ ] Check files installed to `~/.local/share/nemo/scripts/nvCOMP/`
- [ ] Restart file manager (Nautilus: `nautilus -q`, then reopen)

---

## Test Suite 1: Nautilus Integration

### Test 1.1: Basic Menu Display

**Test Case:** Right-click context menu appears for regular files

- [ ] Open Nautilus
- [ ] Navigate to a folder with test files
- [ ] Right-click on a single text file (e.g., `.txt`)
- [ ] **Expected:** "Compress with nvCOMP" menu item appears
- [ ] Menu item has nvcomp icon (if available)
- [ ] Menu item is enabled and clickable

**Test Case:** Submenu displays correctly

- [ ] Hover over "Compress with nvCOMP" menu item
- [ ] **Expected:** Submenu expands showing:
  - [ ] "LZ4 (Fast, Good Ratio)"
  - [ ] "Zstd (Best Ratio)"
  - [ ] "Snappy (Fastest)"
  - [ ] Separator line
  - [ ] GPU algorithms (if CUDA available):
    - [ ] "GDeflate (GPU)"
    - [ ] "ANS (GPU)"
    - [ ] "Bitcomp (GPU)"
  - [ ] Separator line
  - [ ] "Open nvCOMP GUI..."

### Test 1.2: Compression - Single File

**Test Case:** Compress single file with LZ4

- [ ] Right-click on a test file (e.g., `test_data.bin`)
- [ ] Select "Compress with nvCOMP" → "LZ4 (Fast, Good Ratio)"
- [ ] **Expected:** 
  - [ ] Desktop notification appears: "Compressing with LZ4..."
  - [ ] nvCOMP GUI opens
  - [ ] File is pre-loaded in the GUI
  - [ ] Algorithm is pre-selected to LZ4
  - [ ] Compression starts automatically

**Test Case:** Compress single file with Zstd

- [ ] Right-click on a test file
- [ ] Select "Compress with nvCOMP" → "Zstd (Best Ratio)"
- [ ] **Expected:** GUI opens with Zstd selected

**Test Case:** Compress single file with Snappy

- [ ] Right-click on a test file
- [ ] Select "Compress with nvCOMP" → "Snappy (Fastest)"
- [ ] **Expected:** GUI opens with Snappy selected

### Test 1.3: Compression - Multiple Files

**Test Case:** Compress multiple files

- [ ] Select multiple files (Ctrl+Click or Shift+Click)
- [ ] Right-click on selection
- [ ] Select "Compress with nvCOMP" → "LZ4 (Fast, Good Ratio)"
- [ ] **Expected:**
  - [ ] Notification shows "Processing X item(s)..."
  - [ ] GUI opens with all selected files loaded
  - [ ] All files appear in the file list

### Test 1.4: Compression - Folder

**Test Case:** Compress entire folder

- [ ] Right-click on a folder
- [ ] Select "Compress with nvCOMP" → "LZ4 (Fast, Good Ratio)"
- [ ] **Expected:**
  - [ ] GUI opens with folder path loaded
  - [ ] Folder compression is configured

### Test 1.5: Decompression Menu

**Test Case:** Decompress menu for archives

- [ ] Create a test archive (compress a file first)
- [ ] Right-click on `.lz4` file
- [ ] **Expected:** Context menu shows:
  - [ ] "Extract Here"
  - [ ] "Extract to Folder..."
  - [ ] "View Archive"
- [ ] No "Compress with nvCOMP" option (already compressed)

**Test Case:** Extract Here functionality

- [ ] Right-click on an archive file
- [ ] Select "Extract Here"
- [ ] **Expected:**
  - [ ] GUI opens for extraction
  - [ ] Archive is loaded in decompress mode

**Test Case:** View Archive functionality

- [ ] Right-click on an archive file
- [ ] Select "View Archive"
- [ ] **Expected:**
  - [ ] GUI opens
  - [ ] Archive viewer shows contents

### Test 1.6: GPU Algorithm Tests (Requires CUDA)

**Prerequisites:**
- [ ] CUDA toolkit installed
- [ ] NVIDIA GPU available
- [ ] nvidia-smi command works

**Test Case:** GPU algorithms visible

- [ ] Right-click on a file
- [ ] Open "Compress with nvCOMP" submenu
- [ ] **Expected:** GPU algorithms section visible after separator

**Test Case:** Compress with GDeflate

- [ ] Select "GDeflate (GPU)" from menu
- [ ] **Expected:** GUI opens with GDeflate selected

**Test Case:** GPU algorithms hidden without CUDA

- [ ] Temporarily rename/move nvidia-smi: `sudo mv /usr/bin/nvidia-smi /usr/bin/nvidia-smi.bak`
- [ ] Restart Nautilus: `nautilus -q`
- [ ] Right-click on file and check menu
- [ ] **Expected:** GPU algorithms not shown
- [ ] Restore nvidia-smi: `sudo mv /usr/bin/nvidia-smi.bak /usr/bin/nvidia-smi`

### Test 1.7: File Type Detection

**Test Case:** Already compressed files

- [ ] Right-click on `.lz4` file
- [ ] **Expected:** Only extract options shown, no compress menu

**Test Case:** Multiple file types

- [ ] Select mix of regular files and archives
- [ ] Right-click
- [ ] **Expected:** Both compress and extract options shown

**Test Case:** Supported archive extensions

Test each extension is recognized:
- [ ] `.lz4`
- [ ] `.zstd`
- [ ] `.zst`
- [ ] `.snappy`
- [ ] `.nvcomp`
- [ ] `.gdeflate`
- [ ] `.ans`
- [ ] `.bitcomp`

### Test 1.8: Error Handling

**Test Case:** nvcomp-gui not found

- [ ] Temporarily move nvcomp-gui out of PATH
- [ ] Right-click on file
- [ ] **Expected:** No menu items appear OR error notification

**Test Case:** Invalid file selection

- [ ] Try right-clicking on non-accessible file (permissions issue)
- [ ] **Expected:** Graceful error handling

---

## Test Suite 2: Nemo Integration

### Test 2.1: Basic Menu Display

**Test Case:** Scripts appear in Nemo context menu

- [ ] Open Nemo file manager
- [ ] Right-click on a file
- [ ] Navigate to "Scripts" submenu
- [ ] **Expected:** nvCOMP folder with scripts:
  - [ ] "Compress with nvCOMP"
  - [ ] "Compress with nvCOMP (LZ4)"
  - [ ] "Compress with nvCOMP (Zstd)"
  - [ ] "Compress with nvCOMP (Snappy)"

### Test 2.2: Compression - Single File

**Test Case:** Generic compress script

- [ ] Right-click on a test file
- [ ] Scripts → nvCOMP → "Compress with nvCOMP"
- [ ] **Expected:** GUI opens with file loaded

**Test Case:** LZ4-specific script

- [ ] Right-click on a test file
- [ ] Scripts → nvCOMP → "Compress with nvCOMP (LZ4)"
- [ ] **Expected:**
  - [ ] Notification shows "Compressing with LZ4..."
  - [ ] GUI opens with LZ4 pre-selected
  - [ ] Compression starts automatically

**Test Case:** Zstd-specific script

- [ ] Test "Compress with nvCOMP (Zstd)"
- [ ] **Expected:** GUI opens with Zstd selected

**Test Case:** Snappy-specific script

- [ ] Test "Compress with nvCOMP (Snappy)"
- [ ] **Expected:** GUI opens with Snappy selected

### Test 2.3: Compression - Multiple Files

**Test Case:** Multiple file selection

- [ ] Select multiple files in Nemo
- [ ] Right-click → Scripts → nvCOMP → "Compress with nvCOMP (LZ4)"
- [ ] **Expected:**
  - [ ] All selected files passed to GUI
  - [ ] Notification shows correct file count

### Test 2.4: Error Handling

**Test Case:** No files selected

- [ ] Right-click in empty space (background)
- [ ] Check if scripts appear
- [ ] If they appear and you click one:
- [ ] **Expected:** Error dialog or notification

**Test Case:** Script permissions

- [ ] Verify scripts are executable: `ls -la ~/.local/share/nemo/scripts/nvCOMP/`
- [ ] **Expected:** All scripts have execute permission (+x)

---

## Test Suite 3: Cross-Functional Tests

### Test 3.1: Notifications

**Test Case:** Desktop notifications work

- [ ] Compress a file using context menu
- [ ] **Expected:** Notification appears in notification area
- [ ] Notification has appropriate icon
- [ ] Notification message is clear and informative

**Test Case:** Multiple operations

- [ ] Start compression on file 1
- [ ] Immediately start compression on file 2
- [ ] **Expected:** Separate GUI windows or queued operations

### Test 3.2: File Paths with Special Characters

**Test Case:** Spaces in filename

- [ ] Create file: `test file with spaces.txt`
- [ ] Compress via context menu
- [ ] **Expected:** File loads correctly, no path errors

**Test Case:** Unicode characters

- [ ] Create file: `测试文件.txt` or `tëst.txt`
- [ ] Compress via context menu
- [ ] **Expected:** Handles unicode correctly

**Test Case:** Long paths

- [ ] Create deeply nested directory structure
- [ ] Compress file from deep path
- [ ] **Expected:** No path length errors

### Test 3.3: Performance

**Test Case:** Menu response time

- [ ] Right-click on file
- [ ] Time how long menu takes to appear
- [ ] **Expected:** Menu appears within 1 second

**Test Case:** Large file selection

- [ ] Select 100+ files
- [ ] Right-click and select compress
- [ ] **Expected:** System remains responsive

---

## Test Suite 4: Integration with Desktop Environment

### Test 4.1: MIME Type Integration

**Test Case:** Archive icons

- [ ] Compress a file
- [ ] Check if `.lz4` file has custom icon
- [ ] **Expected:** nvcomp icon displayed (from Task 5.1)

**Test Case:** Double-click to open

- [ ] Double-click on `.lz4` file
- [ ] **Expected:** Opens in nvCOMP GUI (from Task 5.1)

### Test 4.2: Settings Persistence

**Test Case:** Algorithm choice persistence

- [ ] Compress file with LZ4 via context menu
- [ ] Open GUI manually
- [ ] **Expected:** Last used settings remembered (from GUI settings)

---

## Test Suite 5: Installation & Uninstallation

### Test 5.1: User Installation

**Test Case:** Clean user install

- [ ] Run: `./install_file_manager_integration.sh`
- [ ] **Expected:**
  - [ ] No errors during installation
  - [ ] Success message displayed
  - [ ] Files created in `~/.local/share/`
  - [ ] Dependencies checked and reported

**Test Case:** Installation without dependencies

- [ ] Remove python3-nautilus: `sudo apt remove python3-nautilus`
- [ ] Run installation script
- [ ] **Expected:** Warning about missing dependency
- [ ] Script continues with Nemo installation

### Test 5.2: System-Wide Installation

**Test Case:** System-wide install

- [ ] Run: `sudo ./install_file_manager_integration.sh --system`
- [ ] **Expected:**
  - [ ] Files installed to `/usr/share/`
  - [ ] All users can access integration

### Test 5.3: Uninstallation

**Test Case:** User uninstall

- [ ] Run: `./install_file_manager_integration.sh --uninstall`
- [ ] **Expected:**
  - [ ] Files removed from `~/.local/share/`
  - [ ] Nautilus restarted
  - [ ] Context menus no longer appear

**Test Case:** System-wide uninstall

- [ ] Run: `sudo ./install_file_manager_integration.sh --system --uninstall`
- [ ] **Expected:**
  - [ ] Files removed from `/usr/share/`
  - [ ] Integration removed for all users

### Test 5.4: Reinstallation

**Test Case:** Install over existing installation

- [ ] Install once
- [ ] Run installation again
- [ ] **Expected:** Updates files, no errors

---

## Test Suite 6: Compatibility Testing

### Test 6.1: Distribution Compatibility

Test on multiple distributions:

- [ ] **Ubuntu 20.04 (Focal)**
  - [ ] Nautilus version: _____
  - [ ] Integration works: Yes/No
  
- [ ] **Ubuntu 22.04 (Jammy)**
  - [ ] Nautilus version: _____
  - [ ] Integration works: Yes/No
  
- [ ] **Ubuntu 24.04 (Noble)**
  - [ ] Nautilus version: _____
  - [ ] Integration works: Yes/No
  
- [ ] **Fedora (latest)**
  - [ ] Nautilus version: _____
  - [ ] Integration works: Yes/No
  
- [ ] **Linux Mint (latest)**
  - [ ] Nemo version: _____
  - [ ] Integration works: Yes/No

### Test 6.2: Desktop Environment Compatibility

- [ ] **GNOME** (with Nautilus)
- [ ] **Cinnamon** (with Nemo)
- [ ] **Ubuntu (GNOME variant)**
- [ ] **Pop!_OS**

---

## Regression Testing

After any code changes, verify:

- [ ] Existing unit tests pass: `python3 test_nautilus_extension.py`
- [ ] No Python syntax errors: `python3 -m py_compile nautilus_extension.py`
- [ ] Shell script syntax: `bash -n nemo_script.sh`
- [ ] Installation script: `bash -n install_file_manager_integration.sh`

---

## Bug Report Template

If issues are found, document them using this template:

```
**Bug Title:** [Short description]

**Component:** [Nautilus Extension / Nemo Script / Installer]

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Behavior:**


**Actual Behavior:**


**Environment:**
- OS: 
- Desktop: 
- File Manager: 
- Python Version: 
- CUDA Available: Yes/No

**Error Messages/Logs:**


**Screenshots:** [If applicable]
```

---

## Testing Sign-Off

**Tester Name:** _____________________  
**Date:** _____________________  
**Test Environment:** _____________________  

**Overall Results:**
- Total Tests: _____
- Passed: _____
- Failed: _____
- Skipped: _____

**Critical Issues Found:** _____

**Recommendation:** ☐ Approve for Release  ☐ Needs Fixes  ☐ Major Issues

**Notes:**

