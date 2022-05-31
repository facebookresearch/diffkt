#!/usr/bin/env bash

#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
