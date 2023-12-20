#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/rwkv.cpp/build

cd rwkv.cpp
mkdir -p build
cd build
# cmake .. -DRWKV_CUBLAS=ON
cmake ..
cmake --build . --config Release
cd ../..

cd examples
go build .


