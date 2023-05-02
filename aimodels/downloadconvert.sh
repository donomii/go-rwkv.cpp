#!/bin/sh

pip3 install torch numpy

LARGE=https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-14B-v11x-Eng99%25-Other1%25-20230501-ctx8192
SMALL=https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-1B5-v11-Eng99%25-Other1%25-20230425-ctx4096

wget -nc ${SMALL}.pth
python3 ../rwkv.cpp/rwkv//convert_pytorch_to_ggml.py ${SMALL}.pth         ${SMALL}_float16.bin  float16
python3 ../rwkv.cpp/rwkv/quantize.py                 ${SMALL}_float16.bin ${SMALL}_quant4.bin Q4_2
cp ${SMALL}_quant4.bin small.bin

wget -nc ${LARGE}.pth
python3 ../rwkv.cpp/rwkv//convert_pytorch_to_ggml.py ${LARGE}.pth         ${LARGE}_float16.bin  float16
python3 ../rwkv.cpp/rwkv/quantize.py                 ${LARGE}_float16.bin ${LARGE}_quant4.bin Q4_2
cp ${LARGE}_quant4.bin large.bin

