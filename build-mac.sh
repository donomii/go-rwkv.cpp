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

cd aimodels
sh downloadconvert.sh  
cd ..

export DYLD_LIBRARY_PATH=./ 
./ai
