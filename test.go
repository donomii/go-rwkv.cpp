package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"text/template"

)

// A structure to hold the conversation state
type ConversationState struct {
	// The context
	ctx *Context
	UserText []string
	BotText  []string
}




func main() {

	ctx, err := InitFromFile("aimodels/RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096_quant4.bin", 8)

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

{{ .Bot }}{{ .Separator }} Not at all! I'm listening.`

	//Uses text Template to fill out the template

	type Preamble struct {
		User      string
		Bot       string
		Separator string
	}

	preamble := Preamble{
		User:      "Bob",
		Bot:       "Alice",
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
	var conv ConversationState
	//if the conversation file exists, load it, print it, and processit as input, adding the correct names
	//if it doesn't exist, create it an empty data struct
	if _, err := os.Stat("conversation.json"); err == nil {
		fmt.Println("Loading conversation")
		data, err := ioutil.ReadFile("conversation.json")
		if err != nil {
			panic(err)
		}
		
		err = json.Unmarshal(data, &conv)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(conv.UserText); i++ {
			fmt.Println(conv.UserText[i])
			fmt.Println(conv.BotText[i])
			input := "\n\nBob: " + conv.UserText[i] + "\n\nAlice: " + conv.BotText[i]
			elem_buff, logit_buff, err = process_input(input, elem_buff, &tk, ctx)
			if err != nil {
				panic(err)
			}
		}
	} else {
		fmt.Println("Creating conversation")
		conv = ConversationState{}
		conv.UserText = append(conv.UserText, pre)
		conv.BotText = append(conv.BotText, "")
		data, err := json.Marshal(conv)
		if err != nil {
			panic(err)
		}
		ioutil.WriteFile("conversation.json", data, 0644)
	}


	// Read lines from stdin, and submit them to the model, until the user types exit

	fmt.Println("Enter text to send to the model, or type exit to quit")
	conv = ConversationState{}
	reader := bufio.NewReader(os.Stdin)
	
	for {
		fmt.Print("Enter text: ")
		text, _ := reader.ReadString('\n')
		text = strings.Replace(text, "\n", "", -1)
		conv.UserText = append(conv.UserText, text)
		text = "\n\nBob: " + text + "\n\nAlice:"

		if text == "exit" {
			break
		}

		elem_buff, logit_buff, err = process_input(text, elem_buff, &tk, ctx)
		response_text := ""
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
			response_text += chars

			if strings.Contains(chars, "\n") {
				break
			}
		}

		conv.BotText = append(conv.BotText, response_text)
		//Save the conversation state to a file
		data ,err:= json.Marshal(conv)
		if err != nil {
			panic(err)
		}
		ioutil.WriteFile("conversation.json", data, 0644)
	}

}


func process_input(input string, elem_buff []float32, tk *Tokenizer, ctx *Context) ([]float32, []float32, error) {
	tokens, err := tk.Encode(input)
	if err != nil {
		panic(err)
	}

	logit_buff := make([]float32, ctx.GetLogitsBufferElementCount())

	for _, t := range tokens {
		fmt.Print(t.Value)

		elem_buff, logit_buff, _, err = ctx.Eval(int32(t.ID), elem_buff)
		if err != nil {
			panic(err)
		}

	}

	return elem_buff, logit_buff, nil
}
