package main

import (
	"errors"
	"unsafe"
)

/*
#include <stdlib.h>
*/
//#cgo CFLAGS: -I./rwkv.cpp/ggml/include/ggml/
//#cgo CPPFLAGS: -I./rwkv.cpp/ggml/include/ggml/
//#cgo LDFLAGS: -L${SRCDIR}  -lrwkv
// #include "includes.h"
import "C"


type Context struct {
	cCtx *C.struct_rwkv_context
}

func InitFromFile(modelFilePath string, nThreads uint32) (*Context, error) {
	cModelFilePath := C.CString(modelFilePath)
	defer C.free(unsafe.Pointer(cModelFilePath))

	cCtx := C.rwkv_init_from_file(cModelFilePath, C.uint32_t(nThreads))
	if cCtx == nil {
		return nil, errors.New("failed to initialize rwkv context from file")
	}

	return &Context{cCtx}, nil
}

func (ctx *Context) Eval(token int32, stateIn []float32) ([]float32, []float32, bool, error) {
	var cStateIn, cStateOut, cLogitsOut *C.float

	logitsOut := make([]float32, ctx.GetLogitsBufferElementCount())
	stateOut := make([]float32, ctx.GetStateBufferElementCount())

	if stateIn != nil {
		cStateIn = (*C.float)(unsafe.Pointer(&stateIn[0]))
	}

	if stateOut != nil {
		cStateOut = (*C.float)(unsafe.Pointer(&stateOut[0]))
	}

	if logitsOut != nil {
		cLogitsOut = (*C.float)(unsafe.Pointer(&logitsOut[0]))
	}

	success := C.rwkv_eval(ctx.cCtx, C.int32_t(token), cStateIn, cStateOut, cLogitsOut)
	if success == false {
		return nil, nil, false, errors.New("failed to evaluate rwkv")
	}

	return stateOut, logitsOut, true, nil
}

func (ctx *Context) GetStateBufferElementCount() uint32 {
	return uint32(C.rwkv_get_state_buffer_element_count(ctx.cCtx))
}

func (ctx *Context) GetLogitsBufferElementCount() uint32 {
	return uint32(C.rwkv_get_logits_buffer_element_count(ctx.cCtx))
}

func (ctx *Context) Free() {
	C.rwkv_free(ctx.cCtx)
	ctx.cCtx = nil
}

func QuantizeModelFile(modelFilePathIn, modelFilePathOut string, formatName string) (bool, error) {
	cModelFilePathIn := C.CString(modelFilePathIn)
	defer C.free(unsafe.Pointer(cModelFilePathIn))

	cModelFilePathOut := C.CString(modelFilePathOut)
	defer C.free(unsafe.Pointer(cModelFilePathOut))

	cFormatName := C.CString(formatName)
	defer C.free(unsafe.Pointer(cFormatName))

	success := C.rwkv_quantize_model_file(cModelFilePathIn, cModelFilePathOut, cFormatName)
	if success == false {
		return false, errors.New("failed to quantize the model file")
	}

	return true, nil
}

func GetSystemInfoString() string {
	return C.GoString(C.rwkv_get_system_info_string())
}
