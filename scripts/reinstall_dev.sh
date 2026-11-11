#!/usr/bin/env bash
# Reinstall the renamed package into the local .venv
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m pip install -U 'setuptools>=68' 'wheel'
python -m pip uninstall -y emg-denoise emg-eeg-denoise || true
python -m pip install -e '.[dev]'
python -m pip show emg-denoise
command -v emg-denoise
command -v eed
emg-denoise --help | head -8
