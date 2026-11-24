#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTS_DIR="$SCRIPT_DIR/venv_parts"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== SentiMind: Restoring backend/venv ==="
if [ ! -d "$PARTS_DIR" ]; then
  echo "Error: Directory $PARTS_DIR not found!"
  exit 1
fi

echo "Combining and extracting 90MB chunks to backend/venv..."
cat "$PARTS_DIR"/backend_venv.tar.gz.part_* | tar -xzf - -C "$PROJECT_ROOT"

echo "✓ backend/venv restored successfully at $SCRIPT_DIR/venv"
