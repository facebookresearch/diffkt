#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKING_DIR="$(pwd)"

pushd "$SCRIPT_DIR" > /dev/null

if [ -e /opt/intel/oneapi/setvars.sh ]; then
  source /opt/intel/oneapi/setvars.sh
fi

mkdir -p ../build

cd ../build

# Build.
if [ ! -f "Makefile"  ]; then
    cmake ..
fi
make -j8 > /dev/null

popd > /dev/null
