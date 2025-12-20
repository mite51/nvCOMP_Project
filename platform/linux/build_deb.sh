#!/bin/bash
# build_deb.sh - Build Debian/Ubuntu .deb packages for nvCOMP
#
# Usage:
#   ./build_deb.sh [options]
#
# Options:
#   --clean        Clean build directory before building
#   --no-sign      Skip package signing (for testing)
#   --install-deps Install build dependencies
#   --help         Show this help message

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Options
CLEAN=0
NO_SIGN=0
INSTALL_DEPS=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=1
            shift
            ;;
        --no-sign)
            NO_SIGN=1
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean        Clean build directory before building"
            echo "  --no-sign      Skip package signing (for testing)"
            echo "  --install-deps Install build dependencies"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  nvCOMP Debian Package Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the project root
cd "$PROJECT_ROOT"
echo -e "${GREEN}Project root:${NC} $PROJECT_ROOT"

# Install dependencies if requested
if [ $INSTALL_DEPS -eq 1 ]; then
    echo -e "\n${YELLOW}Installing build dependencies...${NC}"
    sudo apt-get update
    sudo apt-get install -y \
        debhelper \
        devscripts \
        build-essential \
        cmake \
        g++-12 \
        qt6-base-dev \
        qt6-base-dev-tools \
        libqt6core6 \
        libqt6widgets6 \
        libqt6gui6 \
        lintian \
        fakeroot
    
    # Optional CUDA (recommended)
    if ! dpkg -l | grep -q nvidia-cuda-toolkit; then
        echo -e "${YELLOW}CUDA Toolkit not installed. GPU support will be disabled.${NC}"
        echo -e "${YELLOW}To enable GPU support, install: sudo apt-get install nvidia-cuda-toolkit${NC}"
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Check for required tools
echo -e "\n${YELLOW}Checking required tools...${NC}"
MISSING_TOOLS=0

for tool in dpkg-buildpackage debuild lintian cmake g++; do
    if ! command -v $tool &> /dev/null; then
        echo -e "${RED}✗ Missing: $tool${NC}"
        MISSING_TOOLS=1
    else
        echo -e "${GREEN}✓ Found: $tool${NC}"
    fi
done

if [ $MISSING_TOOLS -eq 1 ]; then
    echo -e "\n${RED}Error: Missing required tools${NC}"
    echo -e "Run with ${YELLOW}--install-deps${NC} to install them"
    exit 1
fi

# Check for GCC-12 (needed for CUDA on Ubuntu 24.04)
if command -v g++-12 &> /dev/null; then
    echo -e "${GREEN}✓ Found: g++-12 (required for CUDA compatibility)${NC}"
else
    echo -e "${YELLOW}⚠ g++-12 not found. Install with: sudo apt-get install g++-12${NC}"
fi

# Check CUDA availability
if command -v nvcc &> /dev/null || [ -d /usr/local/cuda ]; then
    echo -e "${GREEN}✓ CUDA detected (GPU support enabled)${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not detected (CPU-only build)${NC}"
fi

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo -e "\n${YELLOW}Cleaning build directory...${NC}"
    rm -rf build/
    rm -rf debian/.debhelper/
    rm -f debian/files
    rm -f debian/*.log
    rm -f debian/*.substvars
    rm -rf debian/nvcomp-*/
    echo -e "${GREEN}✓ Clean complete${NC}"
fi

# Verify debian/ directory exists
if [ ! -d "platform/linux/debian" ]; then
    echo -e "\n${RED}Error: debian/ directory not found!${NC}"
    echo "Expected location: platform/linux/debian/"
    exit 1
fi

# Copy debian directory to project root (required by debuild)
echo -e "\n${YELLOW}Preparing debian directory...${NC}"
rm -rf debian/
cp -r platform/linux/debian/ debian/
echo -e "${GREEN}✓ Debian directory ready${NC}"

# Verify all required files
echo -e "\n${YELLOW}Verifying package files...${NC}"
REQUIRED_FILES=(
    "debian/control"
    "debian/rules"
    "debian/changelog"
    "debian/copyright"
    "debian/nvcomp-gui.install"
    "debian/nvcomp-cli.install"
)

# Note: debian/compat is NOT required with modern debhelper (13+)
# Compat level is specified in debian/control via debhelper-compat

MISSING_FILES=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ Missing: $file${NC}"
        MISSING_FILES=1
    else
        echo -e "${GREEN}✓ Found: $file${NC}"
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo -e "\n${RED}Error: Missing required package files${NC}"
    exit 1
fi

# Build the package
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Building Debian Packages${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

BUILD_OPTS=""
if [ $NO_SIGN -eq 1 ]; then
    BUILD_OPTS="-us -uc"
    echo -e "${YELLOW}Building unsigned packages (for testing)${NC}"
else
    echo -e "${YELLOW}Building signed packages${NC}"
fi

# Run debuild
echo -e "\n${YELLOW}Running debuild...${NC}"
if debuild $BUILD_OPTS -b 2>&1 | tee build.log; then
    echo -e "\n${GREEN}✓ Package build successful!${NC}"
else
    echo -e "\n${RED}✗ Package build failed!${NC}"
    echo -e "Check ${YELLOW}build.log${NC} for details"
    exit 1
fi

# Create output directory and move packages
echo -e "\n${YELLOW}Organizing packages...${NC}"
PACKAGE_DIR="$PROJECT_ROOT/platform/linux/packaged"
mkdir -p "$PACKAGE_DIR"

# Move all package files
if ls ../*.deb ../*.ddeb ../*.buildinfo ../*.changes 1> /dev/null 2>&1; then
    mv ../*.deb ../*.ddeb ../*.buildinfo ../*.changes "$PACKAGE_DIR/" 2>/dev/null || true
    echo -e "${GREEN}✓ Packages moved to platform/linux/packaged/${NC}"
fi

# List generated packages
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Generated Packages${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if ls "$PACKAGE_DIR"/*.deb 1> /dev/null 2>&1; then
    for deb in "$PACKAGE_DIR"/*.deb; do
        SIZE=$(du -h "$deb" | cut -f1)
        echo -e "${GREEN}✓ $(basename "$deb")${NC} ($SIZE)"
    done
else
    echo -e "${RED}No .deb files found!${NC}"
    exit 1
fi

# Run lintian checks
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Lintian Quality Checks${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for deb in "$PACKAGE_DIR"/*.deb; do
    if [ -f "$deb" ]; then
        echo -e "${YELLOW}Checking $(basename "$deb")...${NC}"
        if lintian "$deb" 2>&1 | tee -a lintian.log; then
            echo -e "${GREEN}✓ Lintian check passed${NC}"
        else
            echo -e "${YELLOW}⚠ Lintian found some issues (see lintian.log)${NC}"
        fi
        echo ""
    fi
done

# Package information
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Package Information${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for deb in "$PACKAGE_DIR"/*.deb; do
    if [ -f "$deb" ]; then
        echo -e "${GREEN}Package:${NC} $(basename "$deb")"
        dpkg-deb --info "$deb" | grep -E "Package:|Version:|Architecture:|Depends:"
        echo ""
    fi
done

# Installation instructions
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}  Installation Instructions${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To install the package:"
echo -e "  ${YELLOW}sudo dpkg -i platform/linux/packaged/nvcomp-gui_*.deb${NC}"
echo -e "  ${YELLOW}sudo apt-get install -f${NC}  # Install dependencies"
echo ""
echo "To install CLI only:"
echo -e "  ${YELLOW}sudo dpkg -i platform/linux/packaged/nvcomp-cli_*.deb${NC}"
echo ""
echo "To uninstall:"
echo -e "  ${YELLOW}sudo apt-get remove nvcomp-gui${NC}"
echo ""
echo "Package files location:"
echo -e "  ${YELLOW}$PACKAGE_DIR/${NC}"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

