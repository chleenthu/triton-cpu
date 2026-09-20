#!/usr/bin/env bash
# Apply the patches in this directory to the current triton-cpu checkout.
#
# Usage:
#   ./patches/apply_patches.sh
#
# These patches remove hard-coded NVIDIA/AMD LLVM codegen library
# dependencies from the build so triton-cpu can be linked against an LLVM
# built with a reduced target list, e.g. -DLLVM_TARGETS_TO_BUILD="host;RISCV".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

for patch in "${SCRIPT_DIR}"/*.patch; do
    echo ""
    echo "  Applying $(basename "${patch}") ..."
    if git -C "${REPO_ROOT}" apply --check "${patch}" 2>/dev/null; then
        git -C "${REPO_ROOT}" apply "${patch}"
        echo "  OK"
    elif git -C "${REPO_ROOT}" apply --check --reverse "${patch}" 2>/dev/null; then
        echo "  SKIPPED (patch already applied)"
    else
        echo "  ERROR: patch does not apply cleanly. Check for conflicts."
        exit 1
    fi
done

echo ""
echo "All patches applied successfully."
