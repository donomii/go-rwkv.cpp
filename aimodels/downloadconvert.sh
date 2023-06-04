#!/bin/sh

pip3 install torch numpy

LARGE=RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192
SMALL=RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096

wget -c https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/${SMALL}.pth
python3 ../rwkv.cpp/rwkv//convert_pytorch_to_ggml.py ${SMALL}.pth         ${SMALL}_float16.bin  float16
python3 ../rwkv.cpp/rwkv/quantize.py                 ${SMALL}_float16.bin ${SMALL}_quant4.bin Q4_2
cp ${SMALL}_quant4.bin small.bin

wget -c https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/${LARGE}.pth
python3 ../rwkv.cpp/rwkv//convert_pytorch_to_ggml.py ${LARGE}.pth         ${LARGE}_float16.bin  float16
python3 ../rwkv.cpp/rwkv/quantize.py                 ${LARGE}_float16.bin ${LARGE}_quant4.bin Q4_2
cp ${LARGE}_quant4.bin large.bin

