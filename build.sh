#!/bin/sh
export C_INCLUDE_PATH=$PWD
export LIBRARY_PATH=$PWD 

cd rwkv.cpp
cmake . -DRWKV_BUILD_SHARED_LIBRARY=OFF
cmake --build .
cp librwkv.a ..
cp ggml/src/libggml.a ..
cd ..

cd examples
go build -o ai .
cd ..
cp examples/ai .
cd aimodels
