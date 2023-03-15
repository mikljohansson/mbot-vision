#!/bin/bash
set -e

export PYTHONPATH=$(dirname $0)/../src:$(dirname $0):${PYTHONPATH}
export MODEL_PATH=${MODEL_PATH:-src/model}

if [[ "$@" == "" ]]; then
  python -m unittest discover -s $(dirname $0) -p '*_test.py'
else
  python -m unittest "$@"
fi
