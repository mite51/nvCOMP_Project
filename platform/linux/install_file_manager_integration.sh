#!/bin/bash
#
# install_file_manager_integration.sh
#
# Installs file manager context menu integration for nvCOMP.
# Supports Nautilus (GNOME) and Nemo (Cinnamon) file managers.
#
# Usage:
#   ./install_file_manager_integration.sh [--system] [--uninstall]
#
# Options:
#   --system     Install system-wide (requires sudo)
#   --uninstall  Remove integration
#   --help       Show this help message
#
# By default, installs to user directories (~/.local/share)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_WIDE=false
UNINSTALL=false

# Paths
NAUTILUS_USER_DIR="$HOME/.local/share/nautilus-python/extensions"
NAUTILUS_SYSTEM_DIR="/usr/share/nautilus-python/extensions"
NEMO_USER_DIR="$HOME/.local/share/nemo/scripts/nvCOMP"
NEMO_SYSTEM_DIR="/usr/share/nemo/scripts/nvCOMP"

# Functions

print_color() {
    local color="$1"
    shift
    echo -e "${color}${*}${NC}"
}

print_header() {
    echo
    print_color "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_color "$BLUE" "$*"
    print_color "$BLUE" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
}

print_success() {
    print_color "$GREEN" "✓ $*"
}

print_error() {
    print_color "$RED" "✗ $*"
}

print_warning() {
    print_color "$YELLOW" "⚠ $*"
}

print_info() {
    print_color "$BLUE" "ℹ $*"
}

show_help() {
    cat << EOF
nvCOMP File Manager Integration Installer

Usage: $0 [OPTIONS]

Options:
    --system        Install system-wide (requires sudo, affects all users)
    --uninstall     Remove file manager integration
    --help          Show this help message

Examples:
    # Install for current user only (recommended)
    $0

    # Install system-wide for all users
    sudo $0 --system

    # Uninstall user integration
    $0 --uninstall

    # Uninstall system-wide integration
    sudo $0 --system --uninstall

Supported File Managers:
    - Nautilus (GNOME Files) - Requires python3-nautilus package
    - Nemo (Cinnamon)        - Uses bash scripts

Integration Features:
    - Right-click context menu on files/folders
    - Compress with different algorithms (LZ4, Zstd, Snappy)
    - GPU algorithm support (if CUDA available)
    - Extract archives
    - View archive contents

EOF
    exit 0
}

check_dependencies() {
    print_header "Checking Dependencies"
    
    local missing_deps=0
    local need_python_nautilus=false
    local need_notify_send=false
    
    # Check for nvcomp-gui
    print_info "Checking for nvcomp-gui..."
    if command -v nvcomp-gui &> /dev/null; then
        local nvcomp_path="$(command -v nvcomp-gui)"
        print_success "Found nvcomp-gui: $nvcomp_path"
    elif [[ -x "$HOME/Dev/nvCOMP_Project/build/gui/nvcomp_gui" ]]; then
        print_success "Found nvcomp-gui: $HOME/Dev/nvCOMP_Project/build/gui/nvcomp_gui"
    else
        print_warning "nvcomp-gui not found in PATH"
        print_info "The integration will still be installed, but you may need to:"
        print_info "  1. Add nvcomp-gui to your PATH, or"
        print_info "  2. Edit the extension files to specify the full path"
    fi
    
    # Check for python3-nautilus (for Nautilus integration)
    print_info "Checking for python3-nautilus..."
    if python3 -c "import gi; gi.require_version('Nautilus', '4.0'); from gi.repository import Nautilus" 2>/dev/null; then
        print_success "Found python3-nautilus (Nautilus 4.0)"
    elif python3 -c "import gi; gi.require_version('Nautilus', '3.0'); from gi.repository import Nautilus" 2>/dev/null; then
        print_success "Found python3-nautilus (Nautilus 3.0)"
    else
        print_warning "python3-nautilus not found"
        print_info "Nautilus integration will not work without it."
        need_python_nautilus=true
        missing_deps=$((missing_deps + 1))
    fi
    
    # Check for notify-send (optional)
    print_info "Checking for notify-send..."
    if command -v notify-send &> /dev/null; then
        print_success "Found notify-send (desktop notifications available)"
    else
        print_warning "notify-send not found (notifications disabled)"
        need_notify_send=true
        missing_deps=$((missing_deps + 1))
    fi
    
    echo
    
    # Offer to install missing dependencies
    if [[ $missing_deps -gt 0 ]]; then
        print_header "Missing Dependencies Detected"
        
        if [[ "$need_python_nautilus" == "true" ]]; then
            print_warning "python3-nautilus is required for Nautilus (GNOME Files) integration"
        fi
        
        if [[ "$need_notify_send" == "true" ]]; then
            print_info "notify-send is recommended for desktop notifications"
        fi
        
        echo
        print_info "Would you like to install missing dependencies now? [Y/n]"
        read -r response
        
        if [[ -z "$response" || "$response" =~ ^[Yy]$ ]]; then
            install_missing_dependencies "$need_python_nautilus" "$need_notify_send"
        else
            print_info "Skipping dependency installation."
            print_info "You can install manually later with:"
            if [[ "$need_python_nautilus" == "true" ]]; then
                print_info "  sudo apt-get install python3-nautilus"
            fi
            if [[ "$need_notify_send" == "true" ]]; then
                print_info "  sudo apt-get install libnotify-bin"
            fi
            echo
        fi
    fi
}

install_missing_dependencies() {
    local need_python_nautilus="$1"
    local need_notify_send="$2"
    
    print_header "Installing Missing Dependencies"
    
    local packages=()
    
    if [[ "$need_python_nautilus" == "true" ]]; then
        packages+=("python3-nautilus")
    fi
    
    if [[ "$need_notify_send" == "true" ]]; then
        packages+=("libnotify-bin")
    fi
    
    if [[ ${#packages[@]} -eq 0 ]]; then
        print_info "No packages to install"
        return 0
    fi
    
    print_info "Installing: ${packages[*]}"
    echo
    
    # Update package list
    print_info "Updating package list..."
    if sudo apt-get update; then
        print_success "Package list updated"
    else
        print_error "Failed to update package list"
        return 1
    fi
    
    echo
    
    # Install packages
    print_info "Installing packages..."
    if sudo apt-get install -y "${packages[@]}"; then
        print_success "Successfully installed: ${packages[*]}"
        echo
        
        # Verify installation
        if [[ "$need_python_nautilus" == "true" ]]; then
            if python3 -c "import gi; gi.require_version('Nautilus', '4.0'); from gi.repository import Nautilus" 2>/dev/null || \
               python3 -c "import gi; gi.require_version('Nautilus', '3.0'); from gi.repository import Nautilus" 2>/dev/null; then
                print_success "python3-nautilus verification: OK"
            else
                print_warning "python3-nautilus installed but verification failed"
                print_info "You may need to log out and back in"
            fi
        fi
        
        if [[ "$need_notify_send" == "true" ]]; then
            if command -v notify-send &> /dev/null; then
                print_success "notify-send verification: OK"
            fi
        fi
        
        return 0
    else
        print_error "Failed to install packages"
        print_info "Please install manually:"
        for pkg in "${packages[@]}"; do
            print_info "  sudo apt-get install $pkg"
        done
        return 1
    fi
}

install_nautilus_extension() {
    print_header "Installing Nautilus Extension"
    
    local target_dir
    if [[ "$SYSTEM_WIDE" == "true" ]]; then
        target_dir="$NAUTILUS_SYSTEM_DIR"
    else
        target_dir="$NAUTILUS_USER_DIR"
    fi
    
    print_info "Target directory: $target_dir"
    
    # Create directory
    mkdir -p "$target_dir"
    print_success "Created directory: $target_dir"
    
    # Copy extension
    local source_file="$SCRIPT_DIR/nautilus_extension.py"
    local target_file="$target_dir/nvcomp_extension.py"
    
    if [[ ! -f "$source_file" ]]; then
        print_error "Source file not found: $source_file"
        return 1
    fi
    
    cp "$source_file" "$target_file"
    chmod 644 "$target_file"
    print_success "Installed: $target_file"
    
    # Restart Nautilus
    print_info "Restarting Nautilus..."
    if pgrep -x nautilus > /dev/null; then
        nautilus -q 2>/dev/null || true
        sleep 1
        print_success "Nautilus restarted"
    else
        print_info "Nautilus not running, no restart needed"
    fi
    
    echo
}

install_nemo_scripts() {
    print_header "Installing Nemo Scripts"
    
    local target_dir
    if [[ "$SYSTEM_WIDE" == "true" ]]; then
        target_dir="$NEMO_SYSTEM_DIR"
    else
        target_dir="$NEMO_USER_DIR"
    fi
    
    print_info "Target directory: $target_dir"
    
    # Create directory
    mkdir -p "$target_dir"
    print_success "Created directory: $target_dir"
    
    # Copy main script
    local source_file="$SCRIPT_DIR/nemo_script.sh"
    
    if [[ ! -f "$source_file" ]]; then
        print_error "Source file not found: $source_file"
        return 1
    fi
    
    # Create multiple scripts for different algorithms
    local scripts=(
        "Compress with nvCOMP"
        "Compress with nvCOMP (LZ4)"
        "Compress with nvCOMP (Zstd)"
        "Compress with nvCOMP (Snappy)"
    )
    
    for script_name in "${scripts[@]}"; do
        local target_file="$target_dir/$script_name"
        cp "$source_file" "$target_file"
        chmod 755 "$target_file"
        print_success "Installed: $script_name"
    done
    
    echo
}

uninstall_nautilus_extension() {
    print_header "Uninstalling Nautilus Extension"
    
    local target_dir
    if [[ "$SYSTEM_WIDE" == "true" ]]; then
        target_dir="$NAUTILUS_SYSTEM_DIR"
    else
        target_dir="$NAUTILUS_USER_DIR"
    fi
    
    local target_file="$target_dir/nvcomp_extension.py"
    
    if [[ -f "$target_file" ]]; then
        rm -f "$target_file"
        print_success "Removed: $target_file"
        
        # Also remove .pyc files
        rm -f "$target_dir/__pycache__/nvcomp_extension."*.pyc 2>/dev/null || true
        
        # Restart Nautilus
        print_info "Restarting Nautilus..."
        if pgrep -x nautilus > /dev/null; then
            nautilus -q 2>/dev/null || true
            sleep 1
            print_success "Nautilus restarted"
        fi
    else
        print_warning "Nautilus extension not found: $target_file"
    fi
    
    echo
}

uninstall_nemo_scripts() {
    print_header "Uninstalling Nemo Scripts"
    
    local target_dir
    if [[ "$SYSTEM_WIDE" == "true" ]]; then
        target_dir="$NEMO_SYSTEM_DIR"
    else
        target_dir="$NEMO_USER_DIR"
    fi
    
    if [[ -d "$target_dir" ]]; then
        rm -rf "$target_dir"
        print_success "Removed: $target_dir"
    else
        print_warning "Nemo scripts directory not found: $target_dir"
    fi
    
    echo
}

show_post_install_info() {
    print_header "Installation Complete!"
    
    print_info "File manager integration has been installed."
    echo
    print_info "Features:"
    echo "  • Right-click on files/folders to compress"
    echo "  • Right-click on archives to extract"
    echo "  • Multiple algorithm support (LZ4, Zstd, Snappy)"
    echo "  • GPU acceleration (if CUDA available)"
    echo
    print_info "To use:"
    echo "  1. Open Nautilus or Nemo file manager"
    echo "  2. Right-click on a file or folder"
    echo "  3. Select 'Compress with nvCOMP'"
    echo "  4. Choose your compression algorithm"
    echo
    print_info "Troubleshooting:"
    echo "  • If menu doesn't appear, try logging out and back in"
    echo "  • For Nautilus, ensure python3-nautilus is installed"
    echo "  • Check that nvcomp-gui is in your PATH or at a known location"
    echo
    
    if [[ "$SYSTEM_WIDE" == "false" ]]; then
        print_info "Installed for current user only."
        print_info "To install system-wide, run: sudo $0 --system"
    else
        print_info "Installed system-wide for all users."
    fi
    
    echo
}

show_post_uninstall_info() {
    print_header "Uninstallation Complete!"
    
    print_info "File manager integration has been removed."
    echo
    
    if [[ "$SYSTEM_WIDE" == "false" ]]; then
        print_info "Removed user integration only."
        print_info "To remove system-wide integration, run: sudo $0 --system --uninstall"
    else
        print_info "Removed system-wide integration."
    fi
    
    echo
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --system)
            SYSTEM_WIDE=true
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for root privileges if system-wide
if [[ "$SYSTEM_WIDE" == "true" && $EUID -ne 0 ]]; then
    print_error "System-wide installation requires root privileges"
    print_info "Please run: sudo $0 --system"
    exit 1
fi

# Main installation/uninstallation logic
if [[ "$UNINSTALL" == "true" ]]; then
    # Uninstall
    print_header "nvCOMP File Manager Integration - Uninstaller"
    uninstall_nautilus_extension
    uninstall_nemo_scripts
    show_post_uninstall_info
else
    # Install
    print_header "nvCOMP File Manager Integration - Installer"
    check_dependencies
    install_nautilus_extension
    install_nemo_scripts
    show_post_install_info
fi

print_success "Done!"

