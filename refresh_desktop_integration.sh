#!/bin/bash
# Script to refresh Linux desktop integration for nvCOMP

echo "=== Refreshing nvCOMP Desktop Integration ==="
echo ""

# Update icon cache
echo "1. Updating icon cache..."
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor 2>/dev/null && echo "   ✓ Icon cache updated" || echo "   ⚠ Icon cache update failed (non-critical)"
else
    echo "   ⚠ gtk-update-icon-cache not found (optional)"
fi

# Update MIME database
echo "2. Updating MIME database..."
if command -v update-mime-database &> /dev/null; then
    update-mime-database ~/.local/share/mime 2>/dev/null && echo "   ✓ MIME database updated" || echo "   ✗ MIME update failed"
else
    echo "   ✗ update-mime-database not found (required)"
fi

# Update desktop database
echo "3. Updating desktop database..."
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database ~/.local/share/applications 2>/dev/null && echo "   ✓ Desktop database updated" || echo "   ✗ Desktop database update failed"
else
    echo "   ✗ update-desktop-database not found (required)"
fi

# Refresh file manager (if possible)
echo "4. Refreshing file manager..."
if pgrep -x "nautilus" > /dev/null; then
    nautilus -q 2>/dev/null && echo "   ✓ Nautilus restarted" || echo "   ⚠ Could not restart Nautilus"
elif pgrep -x "nemo" > /dev/null; then
    nemo -q 2>/dev/null && echo "   ✓ Nemo restarted" || echo "   ⚠ Could not restart Nemo"
elif pgrep -x "dolphin" > /dev/null; then
    killall dolphin 2>/dev/null && echo "   ✓ Dolphin killed (will restart on next use)" || echo "   ⚠ Could not restart Dolphin"
else
    echo "   ⓘ No known file manager detected"
fi

echo ""
echo "=== Testing Icon Visibility ==="
echo ""

# Test if icon is visible
echo "Icon locations:"
ls -1 ~/.local/share/icons/hicolor/*/apps/nvcomp.png 2>/dev/null | while read icon; do
    echo "   ✓ $icon"
done

echo ""
echo "Desktop file:"
if [ -f ~/.local/share/applications/nvcomp.desktop ]; then
    echo "   ✓ ~/.local/share/applications/nvcomp.desktop"
else
    echo "   ✗ Desktop file not found!"
fi

echo ""
echo "MIME types:"
if [ -f ~/.local/share/mime/packages/nvcomp.xml ]; then
    echo "   ✓ ~/.local/share/mime/packages/nvcomp.xml"
else
    echo "   ✗ MIME file not found!"
fi

echo ""
echo "=== Recommendations ==="
echo ""
echo "If icons still don't show:"
echo "1. Log out and log back in (recommended)"
echo "2. Run: xdg-icon-resource forceupdate"
echo "3. Restart your desktop environment"
echo ""
echo "To test file association:"
echo "   xdg-mime query default application/x-lz4"
echo "   (should show: nvcomp.desktop)"
echo ""
echo "To view the icon:"
echo "   xdg-icon-resource lookup --size 48 nvcomp"
echo ""

