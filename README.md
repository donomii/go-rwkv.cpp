# AI with RWKV

# [![Go Reference](https://pkg.go.dev/badge/github.com/donomii/go-rwkv.cpp.svg)](https://pkg.go.dev/github.com/donomii/go-rwkv.cpp) go-rwkv.cpp

gowrkv.go is a wrapper around [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp), which is an adaption of ggml.cpp.

## Features

rkwv.cpp is generally faster, due to keeping the intermediate state of the model, so the entire prompt doesn't have to be reprocessed every time.  For more details, see [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp).

Also, the available models for rwkv.cpp are fully open-source, unlike llama.  You can use these models commercially, and you can modify them to your heart's content.

Training may also be faster, I haven't had a chance to try that yet.

## Installation

Installation is currently complex.  go-rkwv.cpp does not work with ```go get``` yet (patches very welcome).  You will need go, a c++ compiler(clang on Mac), and cmake.

### Download

You must clone this repo /recursively/, as it contains submodules.

```bash
    git clone --recursive https://github.com/donomii/go-rwkv.cpp
```

### Building

There is a build script, build.sh, which will build the c++ library and the go wrapper. Please file bug reports if it doesn't work for you.

```bash
    ./build-mac.sh
```

There is now an alternate build, which builds statically thanks to a makefile provided by @mudler.   

```bash
    make example/ai
```

### Download models

The download script will download some models, and convert them to the correct format.

```bash
    cd aimodels
    sh downloadconvert.sh
```

### Install

go-rwkv.cpp currently builds against the dynamic library librwkv.dylib.  This is not ideal, but it works for now.  You will need to copy this library to a location where the system linker can find it.  On Mac, this is /usr/local/lib.

```bash
    cp librwkv.dylib /usr/local/lib
    export DYLD_LIBRARY_PATH=/Users/donomii/git/go-rwkv.cpp/rwkv.cpp/
```

If you don't want to install it globally, you can set the DYLD_LIBRARY_PATH environment variable to the directory containing librwkv.dylib.

## Use

See the example/ directory for a full working chat program. The following is a minimal example.

```go
    package main

    import (
        "fmt"
        "github.com/donomii/go-rwkv.cpp"
    )

    func main() {
        model := LoadFiles("aimodels/small.bin", "rwkv.cpp/rwkv/20B_tokenizer.json", 8)
    model.ProcessInput("You are a chatbot that is very good at chatting.  blah blah blah")
    response := model.Generate(100, "\n")
    fmt.Println(response)

    }
```

You must use the tokenizer file from rwkv.cpp.  go-rwkv contains a re-implementation of the tokenizer, but it is a minimal implementation that contains just enough code to work with rwkv (and there are probably bugs in it).

## Packaging

To ship a working program that includes this AI, you will need to include the following files:

* librwkv.dylib
* the model file (e.g. RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096_quant4.bin)
* the tokenizer file (i.e. 20B_tokenizer.json)

If you don't install librwkv.dylib globally, you will need to set the DYLD_LIBRARY_PATH environment variable to the directory containing librwkv.dylib.

## License

This program is licensed under the MIT license.  See LICENSE for details.

As far as I am aware, the Raven models are also open source.
