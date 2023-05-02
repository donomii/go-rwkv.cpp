#!/bin/sh

wget https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-1B5-v9-Eng99%25-Other1%25-20230411-ctx4096.pth
python3 ../rwkv.cpp/rwkv//convert_pytorch_to_ggml.py RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096.pth RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096_float16.bin float16
python3 ../rwkv.cpp/rwkv/quantize.py                 RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096_float16.bin RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096_quant4.bin 4

wget https://huggingface.co/BlinkDL/rwkv-4-raven/blob/main/RWKV-4-Raven-14B-v9-Eng99%25-Other1%25-20230412-ctx8192.pth
python3 ../rwkv.cpp/rwkv//convert_pytorch_to_ggml.py RWKV-4-Raven-14B-v9-Eng99%25-Other1%25-20230412-ctx8192.pth         RWKV-4-Raven-14B-v9-Eng99%25-Other1%25-20230412-ctx8192_float16.bin  float16
python3 ../rwkv.cpp/rwkv/quantize.py                 RWKV-4-Raven-14B-v9-Eng99%25-Other1%25-20230412-ctx8192_float16.bin RWKV-4-Raven-14B-v9-Eng99%25-Other1%25-20230412-ctx8192_quant4.bin 4

