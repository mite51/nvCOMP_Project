#!/bin/bash
# Compatibility entry point; keep the build logic in one place.
set -euo pipefail
exec "$(dirname "$(realpath "$0")")/../build_deb.sh" "$@"
