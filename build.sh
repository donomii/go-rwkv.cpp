#!/bin/sh

cd rwkv.cpp
cmake .
cmake --build . --config Release
cp bin/Release/rwkv.dll ..
cd ..

which go
go build -o ai .

export DYLD_LIBRARY_PATH=./ 
./ai
