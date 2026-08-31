#!/usr/bin/env bash
set -euo pipefail

python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
git submodule update --init --recursive
python -m pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple torch==2.1.2
python -m pip install -r third_party/BasicTS/requirements.txt 'blosc2==2.7.1'
python -m pip install --no-deps -e third_party/BasicTS
python -m pip install -e '.[test]'
python -c 'import basicts, torch; assert basicts.__version__ == "1.1.0"; assert torch.__version__ == "2.1.2+cu118"'
python -m pip check
