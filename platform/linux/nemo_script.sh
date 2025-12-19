#!/bin/bash
#
# nvCOMP Nemo Context Menu Script
#
# Provides right-click context menu integration for Nemo file manager (Cinnamon desktop).
# This script is called by Nemo when user selects files and chooses the nvCOMP menu option.
#
# Installation:
#   User:   mkdir -p ~/.local/share/nemo/scripts
#           cp nemo_script.sh ~/.local/share/nemo/scripts/"Compress with nvCOMP"
#           chmod +x ~/.local/share/nemo/scripts/"Compress with nvCOMP"
#
#   System: cp nemo_script.sh /usr/share/nemo/scripts/"Compress with nvCOMP"
#           chmod +x /usr/share/nemo/scripts/"Compress with nvCOMP"
#
# Nemo Environment Variables:
#   NEMO_SCRIPT_SELECTED_FILE_PATHS - Newline-separated list of selected file paths
#   NEMO_SCRIPT_SELECTED_URIS       - Newline-separated list of selected URIs
#   NEMO_SCRIPT_CURRENT_URI         - Current directory URI
#   NEMO_SCRIPT_WINDOW_GEOMETRY     - Window geometry
#
# Note: For algorithm-specific scripts, create multiple copies with different names:
#   - "Compress with nvCOMP (LZ4)"
#   - "Compress with nvCOMP (Zstd)"
#   - etc.

set -e

# Configuration
NVCOMP_GUI=""
ALGORITHM=""
AUTO_COMPRESS=false

# Function to find nvcomp-gui executable
find_nvcomp_gui() {
    local search_paths=(
        "nvcomp-gui"
        "$HOME/Dev/nvCOMP_Project/build/gui/nvcomp_gui"
        "$HOME/.local/bin/nvcomp-gui"
        "/usr/local/bin/nvcomp-gui"
        "/usr/bin/nvcomp-gui"
    )
    
    for path in "${search_paths[@]}"; do
        if command -v "$path" &> /dev/null; then
            echo "$path"
            return 0
        elif [[ -f "$path" && -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done
    
    return 1
}

# Function to send desktop notification
send_notification() {
    local title="$1"
    local message="$2"
    local urgency="${3:-normal}"
    
    if command -v notify-send &> /dev/null; then
        notify-send -i nvcomp -u "$urgency" "$title" "$message" 2>/dev/null || true
    fi
}

# Function to show error dialog
show_error() {
    local title="$1"
    local message="$2"
    
    # Try zenity first
    if command -v zenity &> /dev/null; then
        zenity --error --title="$title" --text="$message" --width=400 2>/dev/null || true
    fi
    
    # Also send notification
    send_notification "$title" "$message" "critical"
}

# Function to check if file is nvCOMP archive
is_nvcomp_archive() {
    local file="$1"
    local ext="${file##*.}"
    local ext_lower="${ext,,}"
    
    case "$ext_lower" in
        lz4|zstd|zst|snappy|nvcomp|gdeflate|ans|bitcomp)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to check CUDA availability
check_cuda_available() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi -L &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Parse script name to determine algorithm
# If script is named "Compress with nvCOMP (LZ4)", extract "LZ4"
SCRIPT_NAME="$(basename "$0")"
REGEX='[(]([^)]+)[)]'
if [[ "$SCRIPT_NAME" =~ $REGEX ]]; then
    ALGO_NAME="${BASH_REMATCH[1]}"
    case "${ALGO_NAME^^}" in
        LZ4)
            ALGORITHM="lz4"
            AUTO_COMPRESS=true
            ;;
        ZSTD)
            ALGORITHM="zstd"
            AUTO_COMPRESS=true
            ;;
        SNAPPY)
            ALGORITHM="snappy"
            AUTO_COMPRESS=true
            ;;
        GDEFLATE)
            ALGORITHM="gdeflate"
            AUTO_COMPRESS=true
            ;;
        ANS)
            ALGORITHM="ans"
            AUTO_COMPRESS=true
            ;;
        BITCOMP)
            ALGORITHM="bitcomp"
            AUTO_COMPRESS=true
            ;;
    esac
fi

# Find nvCOMP GUI
NVCOMP_GUI="$(find_nvcomp_gui)"
if [[ -z "$NVCOMP_GUI" ]]; then
    show_error "nvCOMP Not Found" \
        "Could not find nvcomp-gui executable.\n\nPlease install nvCOMP or add it to your PATH."
    exit 1
fi

# Get selected files from Nemo environment variable
if [[ -z "$NEMO_SCRIPT_SELECTED_FILE_PATHS" ]]; then
    show_error "No Files Selected" \
        "Please select one or more files or folders to compress."
    exit 1
fi

# Read selected files into array
mapfile -t SELECTED_FILES <<< "$NEMO_SCRIPT_SELECTED_FILE_PATHS"

# Remove empty entries
SELECTED_FILES=("${SELECTED_FILES[@]}" )

if [[ ${#SELECTED_FILES[@]} -eq 0 ]]; then
    show_error "No Files Selected" \
        "Please select one or more files or folders to compress."
    exit 1
fi

# Check if we have archives (for decompression) or regular files (for compression)
HAS_ARCHIVES=false
HAS_COMPRESSIBLE=false

for file in "${SELECTED_FILES[@]}"; do
    if is_nvcomp_archive "$file"; then
        HAS_ARCHIVES=true
    else
        HAS_COMPRESSIBLE=true
    fi
done

# Build command
CMD=("$NVCOMP_GUI")

# If we have archives and no algorithm specified, open for decompression
if [[ "$HAS_ARCHIVES" == "true" && "$AUTO_COMPRESS" == "false" ]]; then
    # Open archives in GUI for viewing/extraction
    for file in "${SELECTED_FILES[@]}"; do
        CMD+=("--add-file" "$file")
    done
    
    send_notification "nvCOMP" "Opening ${#SELECTED_FILES[@]} archive(s)..."

# If we have compressible files and algorithm specified, compress
elif [[ "$HAS_COMPRESSIBLE" == "true" && "$AUTO_COMPRESS" == "true" ]]; then
    CMD+=("--compress" "--algorithm" "$ALGORITHM")
    for file in "${SELECTED_FILES[@]}"; do
        if ! is_nvcomp_archive "$file"; then
            CMD+=("--add-file" "$file")
        fi
    done
    
    send_notification "nvCOMP" "Compressing ${#SELECTED_FILES[@]} item(s) with ${ALGORITHM^^}..."

# Otherwise, just open GUI with files
else
    for file in "${SELECTED_FILES[@]}"; do
        CMD+=("--add-file" "$file")
    done
    
    send_notification "nvCOMP" "Opening ${#SELECTED_FILES[@]} item(s)..."
fi

# Launch nvCOMP GUI in background
"${CMD[@]}" &

# Wait a moment for the GUI to start
sleep 0.5

exit 0

