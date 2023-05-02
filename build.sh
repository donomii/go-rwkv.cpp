#!/bin/sh

cd aimodels
cat $(ls x*) > RWKV-4-Raven-14B-v9-Eng99%-Other1%-20230412-ctx8192_quant4.bin 
cd ..

cd rwkv.cpp
cmake .
cmake --build . --config Release
cp bin/Release/rwkv.dll ..
cd ..

which go
go build -o ai .

export DYLD_LIBRARY_PATH=./ 
./ai
