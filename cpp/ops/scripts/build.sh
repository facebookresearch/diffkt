#!/usr/bin/env bash

#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

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
