#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export QT_QPA_PLATFORM=wayland
exec "$DIR/.venv/bin/python3" "$DIR/image_viewer.py" "$@"
