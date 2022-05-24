#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "$SCRIPT_DIR" > /dev/null

mkdir -p build

cd build

# Build.
if [ ! -f "Makefile"  ]; then
    cmake ..
fi
make -j8 > /dev/null

popd > /dev/null
