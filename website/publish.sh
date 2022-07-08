#!/bin/bash
#
current_branch=`git rev-parse --abbrev-ref HEAD`
# save your current work
cd ..
git add .
git commit -m "updated website"
# cleanup gh-pages
git checkout gh-pages
[ -d "docs" ] && git rm -r docs
git commit -m "cleaned out docs"
# cleanup current_branch
git checkout $current_branch
rm -rf docs/*
cd website
rm -rf static/api
rm -rf build/*
cd ..
# generate the kotlin doc off of main to have most current version of source
#git checkout main
cd kotlin
./gradlew :api:dokkaHtml
cd ..
#git checkout $current_branch
cd website
npm run build
cp -a build/. ../docs/
cp CNAME ../docs
cd ..
git checkout gh-pages
git add .
git commit -m "new docs"
git push origin gh-pages --force
git checkout $current_branch
cd website
