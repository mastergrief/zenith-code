#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/libhrm_alloc_hook.so"
gcc -shared -fPIC -O2 -Wall -Wextra -o "${OUT}" "${ROOT}/libhrm_alloc_hook.c" -ldl -lpthread
echo "built ${OUT}"
