#!/bin/bash
# Quick reinstall and test script for nvCOMP context menu fix

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Reinstalling nvCOMP with fixed Nautilus extension...${NC}"
echo ""

# Reinstall
sudo dpkg -i packaged/nvcomp-gui_1.0.0-1_amd64.deb

echo ""
echo -e "${YELLOW}Testing if extension can find nvcomp_gui...${NC}"

# Test the extension
/usr/bin/python3 << 'EOF'
import sys
sys.path.insert(0, '/usr/share/nautilus-python/extensions')

# Load the extension
exec(open('/usr/share/nautilus-python/extensions/nvcomp_extension.py').read())

# Create an instance
provider = NvcompMenuProvider()

print(f"✓ nvcomp_gui found at: {provider.nvcomp_gui_path}")

# Test with a dummy file
class FakeFile:
    def get_uri(self):
        return "file:///tmp/test.txt"

files = [FakeFile()]
items = provider.get_file_items(files)
print(f"✓ Extension returns {len(items)} menu items for test file")

if len(items) > 0:
    print("✓✓ SUCCESS! Context menu should now work!")
else:
    print("✗ ERROR: Still no menu items")
EOF

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Now test it:${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "1. Restart Nautilus:"
echo -e "   ${YELLOW}nautilus -q${NC}"
echo ""
echo "2. Open Nautilus and right-click any file"
echo "   You should see 'Compress with nvCOMP' menu"
echo ""

