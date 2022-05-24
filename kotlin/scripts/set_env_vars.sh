#!/usr/bin/env bash

#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Get path to where this file is. https://stackoverflow.com/a/246128
DIR="$( cd "$( dirname "${BASH_SOURCE[0]:-${(%):-%x}}"   )" >/dev/null 2>&1 && pwd   )"
KOTLIN_DIR=$(dirname $DIR)
REPO_ROOT=$(dirname $KOTLIN_DIR)

# Set Github env vars for authenticating to github packages
GITHUB_DOTENV=$KOTLIN_DIR/github.env
if [ -f "$GITHUB_DOTENV"  ]; then
    source $GITHUB_DOTENV
    export $(cut -d= -f1 $GITHUB_DOTENV)
else
    echo "Warning: github.env not found so Github env vars were not automatically set"
fi

# Set machine-dependent env vars. Extension is used for locating shared libraries
if [ "$(uname)" = "Darwin"  ]; then
    export ENV_NAME="DARWIN"
    export DYLIB_EXTENSION=".dylib"
    if [ "$(echo $(sw_vers -productVersion) | cut -c1-5)" = "10.15"  ]; then
        export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
        export CPLUS_INCLUDE_PATH="/usr/local/opt/gflags/include/"
    fi
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux"  ]; then
    export ENV_NAME="LINUX"
    export DYLIB_EXTENSION=".so"
fi
