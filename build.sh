#!/bin/sh

cd rwkv.cpp
cmake .
cmake --build . --config Release
cp librwkv.dylib ..
cd ..

cd examples
go build -o ai .
cd ..
cp examples/ai .

export DYLD_LIBRARY_PATH=./ 
./ai
