#!/bin/sh

cd rwkv.cpp
cmake .
cmake --build . --config Release
cp bin/Release/rwkv.dll ..
cd ..

cd examples
go build -o ai .
cd ..
cp examples/ai .

export DYLD_LIBRARY_PATH=./ 
./ai
