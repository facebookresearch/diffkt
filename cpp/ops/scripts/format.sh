#!/usr/bin/env bash

set -e

# We add the -print to the components of the "or" that we want to print.
# https://stackoverflow.com/questions/4210042/how-to-exclude-a-directory-in-find-command#comment23853029_4210072
# Inline comments come from https://stackoverflow.com/a/12797512.
find . \
  -path ./build -prune `# exclude the build directory`\
  -or -name "*.cpp" -print \
  -or -name "*.h" -print \
  | xargs clang-format -i
