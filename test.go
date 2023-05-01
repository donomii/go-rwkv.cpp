package main

/*
#include <stdlib.h>
*/
//#cgo CFLAGS: -I/Users/jeremyprice/git/rwkv.cpp/ggml/include/ggml/
//#cgo CPPFLAGS: -I/Users/jeremyprice/git/rwkv.cpp/ggml/include/ggml/
//#cgo LDFLAGS: -L${SRCDIR}  -lrwkv
// #include "rwkv.h"
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
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

func QuantizeModelFile(modelFilePathIn, modelFilePathOut string, qType uint32) (bool, error) {
	cModelFilePathIn := C.CString(modelFilePathIn)
	defer C.free(unsafe.Pointer(cModelFilePathIn))

	cModelFilePathOut := C.CString(modelFilePathOut)
	defer C.free(unsafe.Pointer(cModelFilePathOut))

	success := C.rwkv_quantize_model_file(cModelFilePathIn, cModelFilePathOut, C.uint32_t(qType))
	if success == false {
		return false, errors.New("failed to quantize the model file")
	}

	return true, nil
}

func GetSystemInfoString() string {
	return C.GoString(C.rwkv_get_system_info_string())
}

func main() {

	ctx, err := InitFromFile("RWKV_quant.bin", 8)

	elem_size := ctx.GetStateBufferElementCount()
	logit_size := ctx.GetLogitsBufferElementCount()

	elem_buff := make([]float32, elem_size)
	logit_buff := make([]float32, logit_size)
	if err != nil {
		panic(err)
	}
	preambleTemplate := `The following is a verbose detailed conversation between {{ .User }} and a young girl {{ .Bot }}. {{ .Bot }} is intelligent, friendly and cute. {{ .Bot }} is likely to agree with {{ .User }}.

{{ .User }}{{ .Separator }} Hello {{ .Bot }}, how are you doing?

{{ .Bot }}{{ .Separator }} Hi {{ .User }}! Thanks, I'm fine. What about you?

{{ .User }}{{ .Separator }} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{{ .Bot }}{{ .Separator }} Not at all! I'm listening.

{{ .User }}{{ .Separator }} Hi

{{ .Bot }}{{ .Separator }} Hi

{{ .User }}{{ .Separator }} How are you?

{{ .Bot }}{{ .Separator }} I'm fine. How are you?

{{ .User }}{{ .Separator }} Also good.  What do you want to do today?

{{ .Bot }}{{ .Separator }} I don't know. What do you want to do?

{{ .User }}{{ .Separator }} I want to sleep now

{{ .Bot }}{{ .Separator }} Me too.

{{ .User }}{{ .Separator }} Excellent, let's talk tomorrow

{{ .Bot }}{{ .Separator }} Ok, see you tomorrow

{{ .User }}{{ .Separator }} Good morning!

{{ .Bot }}{{ .Separator }}`

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




	tk, err := LoadTokeniser("rwkv/20B_tokenizer.json")
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

/*{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "special": true,
      "content": "<|endoftext|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false
    },
    {
      "id": 1,
      "special": true,
      "content": "<|padding|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false
    },
    {
      "id": 50276,
      "special": false,
      "content": "  ",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true
    }
  ],
  "normalizer": {
    "type": "NFC"
  },
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  },
  "post_processor": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  },
  "decoder": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "vocab": {
      "<|endoftext|>": 0,
      "<|padding|>": 1,
      "!": 2,
      "\"": 3,
      "#": 4,
      "$": 5,
*/

type Tokenizer struct {
	AddedTokens   []AddedToken  `json:"added_tokens"`
	Normalizer    Normalizer    `json:"normalizer"`
	PreTokenizer  PreTokenizer  `json:"pre_tokenizer"`
	PostProcessor PostProcessor `json:"post_processor"`
	Decoder       Decoder       `json:"decoder"`
	Model         Model         `json:"model"`
}

type AddedToken struct {
	Id         int    `json:"id"`
	Special    bool   `json:"special"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	Lstrip     bool   `json:"lstrip"`
	Rstrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
}

type Normalizer struct {
	Type string `json:"type"`
}

type PreTokenizer struct {
	Type           string `json:"type"`
	AddPrefixSpace bool   `json:"add_prefix_space"`
	TrimOffsets    bool   `json:"trim_offsets"`
}

type PostProcessor struct {
	Type           string `json:"type"`
	AddPrefixSpace bool   `json:"add_prefix_space"`
	TrimOffsets    bool   `json:"trim_offsets"`
}

type Decoder struct {
	Type           string `json:"type"`
	AddPrefixSpace bool   `json:"add_prefix_space"`
	TrimOffsets    bool   `json:"trim_offsets"`
}

type Model struct {
	Type                    string         `json:"type"`
	Dropout                 float32        `json:"dropout"`
	UnkToken                string         `json:"unk_token"`
	ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
	EndOfWordSuffix         string         `json:"end_of_word_suffix"`
	FuseUnk                 bool           `json:"fuse_unk"`
	Vocab                   map[string]int `json:"vocab"`
	Merges                  []string       `json:"merges"`
}

func LoadTokeniser(file string) (Tokenizer, error) {
	var tokeniser Tokenizer
	jsonFile, err := os.Open(file)
	if err != nil {
		return tokeniser, err
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)
	json.Unmarshal(byteValue, &tokeniser)
	return tokeniser, nil
}

func (t Tokenizer) Encode(text string) ([]Token, error) {
	return Tokenize(text, t)
}
