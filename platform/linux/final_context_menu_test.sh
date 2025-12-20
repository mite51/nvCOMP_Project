#!/bin/bash
# Final test - verify context menu appears

echo "=========================================="
echo "  Final Context Menu Test"
echo "=========================================="
echo ""

# Remove old user extension
if [ -f ~/.local/share/nautilus-python/extensions/nvcomp_extension.py ]; then
    echo "✓ Removing old user extension..."
    rm -v ~/.local/share/nautilus-python/extensions/nvcomp_extension.py
fi

# Check system extension
if [ -f /usr/share/nautilus-python/extensions/nvcomp_extension.py ]; then
    echo "✓ System extension exists"
    if grep -q "__gtype_name__" /usr/share/nautilus-python/extensions/nvcomp_extension.py; then
        echo "  ✓ Has __gtype_name__ fix"
    else
        echo "  ✗ Missing __gtype_name__ fix!"
        exit 1
    fi
else
    echo "✗ System extension NOT found!"
    exit 1
fi

# Restart Nautilus
echo ""
echo "Restarting Nautilus..."
nautilus -q
sleep 1

echo ""
echo "=========================================="
echo "  Context Menu Should Now Work!"
echo "=========================================="
echo ""
echo "Open Nautilus and right-click any file"
echo "You should see: 'Compress with nvCOMP'"
echo ""

