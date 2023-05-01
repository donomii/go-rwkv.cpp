package main

/*
#include <stdlib.h>
*/
//#cgo CFLAGS: -I./rwkv.cpp/ggml/include/ggml/
//#cgo CPPFLAGS: -I./rwkv.cpp/ggml/include/ggml/
//#cgo LDFLAGS: -L${SRCDIR}  -lrwkv
// #include "includes.h"
import "C"

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"
	"text/template"
	"bytes"
)

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

func main() {

	ctx, err := InitFromFile("aimodels/RWKV-4-Raven-14B-v9-Eng99%-Other1%-20230412-ctx8192_quant4.bin", 8)

	elem_size := ctx.GetStateBufferElementCount()
	logit_size := ctx.GetLogitsBufferElementCount()

	elem_buff := make([]float32, elem_size)
	logit_buff := make([]float32, logit_size)
	if err != nil {
		panic(err)
	}
	preambleTemplate := `The following is a verbose detailed conversation between {{ .User }} and a woman {{ .Bot }}. {{ .Bot }} is intelligent, friendly and likeable. {{ .Bot }} is likely to agree with {{ .User }}.

{{ .User }}{{ .Separator }} Hello {{ .Bot }}, how are you doing?

{{ .Bot }}{{ .Separator }} Hi {{ .User }}! Thanks, I'm fine. What about you?

{{ .User }}{{ .Separator }} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{{ .Bot }}{{ .Separator }} Not at all! I'm listening.

`

//Uses text Template to fill out the template

	type Preamble struct {
		User      string
		Bot       string
		Separator string
	}

	preamble := Preamble{
		User:      "Bob",
		Bot: 	 "Alice",
		Separator: ":",
	}
	
	//collect the template results in a buffer
	var b bytes.Buffer
	t := template.Must(template.New("preamble").Parse(preambleTemplate))
	err = t.Execute(&b, preamble)
	if err != nil {
		panic(err)
	}
	pre := b.String()




	tk, err := LoadTokeniser("rwkv.cpp/rwkv/20B_tokenizer.json")
	if err != nil {
		panic(err)
	}
	tokens, err := tk.Encode(pre)
	if err != nil {
		panic(err)
	}
	fmt.Println("Loading preamble")

	for _, t := range tokens {
		fmt.Print(t.Value)
		
		elem_buff, logit_buff, _, err = ctx.Eval(int32(t.ID), elem_buff)
		if err != nil {
			panic(err)
		}

		
	}

	for i := 0; i < 100; i++ {

		newtoken, err := sampleLogits(logit_buff, 0.2, 1, map[int]float32{})
		if err != nil {
			panic(err)
		}

	
		elem_buff, logit_buff, _, err = ctx.Eval(int32(newtoken), elem_buff)
		if err != nil {
			panic(err)
		}

		

		chars := DeTokenise(tk, []int{newtoken})
		fmt.Print(chars)

		if strings.Contains(chars, "\n") {
			break
		}
	}

}
