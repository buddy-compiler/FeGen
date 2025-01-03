#!/bin/bash
ROOT=$(pwd)
BUILD_DIR=build
WHEEL_BUILD_DIR=build

mkdir -p $BUILD_DIR

# copy from src to build
cp ./setup.py $BUILD_DIR/setup.py
cp ./MANIFEST.in $BUILD_DIR/MANIFEST.in
cp ./README.md $BUILD_DIR/README.md
cp ./requirements.txt $BUILD_DIR/requirements.txt

cp -r ./python $BUILD_DIR/python
cp -r ./build/lib/python/* $BUILD_DIR/python/FeGen

cd $BUILD_DIR

# build wheel
mkdir dist
cd dist
pip3 wheel --no-deps ..
cd ..

# delete build files
rm ./setup.py
rm ./MANIFEST.in
rm ./README.md
rm ./requirements.txt
rm -rf ./python
rm -rf $WHEEL_BUILD_DIR

cd $ROOT