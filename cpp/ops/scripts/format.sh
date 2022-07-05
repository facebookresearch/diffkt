#!/usr/bin/env bash

#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

# We add the -print to the components of the "or" that we want to print.
# https://stackoverflow.com/questions/4210042/how-to-exclude-a-directory-in-find-command#comment23853029_4210072
# Inline comments come from https://stackoverflow.com/a/12797512.
find . \
  -path ./build -prune `# exclude the build directory`\
  -or -name "*.cpp" -print \
  -or -name "*.h" -print \
  | xargs clang-format -i
