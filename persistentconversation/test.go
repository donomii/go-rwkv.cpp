package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"text/template"

	rwkv "github.com/donomii/go-rwkv.cpp"
)

// A structure to hold the conversation state
type ConversationState struct {
	// The context
	ctx      *rwkv.Context
	UserText []string
	BotText  []string
}

func main() {
	Model := rwkv.LoadFiles("aimodels/large.bin", "rwkv.cpp/python/20B_tokenizer.json", 8)
	preambleTemplate := `The following is a verbose detailed conversation between {{ .User }} and a woman, {{ .Bot }}. {{ .Bot }} is intelligent, friendly and likeable. {{ .Bot }} is likely to agree with {{ .User }}.

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
	err := t.Execute(&b, preamble)
	if err != nil {
		panic(err)
	}
	pre := b.String()

	Model.ProcessInput(pre)
	var conv ConversationState
	//if the conversation file exists, load it, print it, and process it as input, adding the correct names
	//if it doesn't exist, create it as an empty data struct
	if _, err := os.Stat("conversation.json"); err == nil {
		fmt.Println("Loading conversation")
		data, err := os.ReadFile("conversation.json")
		if err != nil {
			panic(err)
		}

		err = json.Unmarshal(data, &conv)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(conv.UserText); i++ {
			//fmt.Println(conv.UserText[i])
			//fmt.Println(conv.BotText[i])
			input := "\n\nBob: " + conv.UserText[i] + "\n\nAlice: " + conv.BotText[i]
			fmt.Print(input)
			Model.ProcessInput(input)

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
		os.WriteFile("conversation.json", data, 0644)
	}

	// Read lines from stdin, and submit them to the model, until the user types exit

	//fmt.Println("Enter text to send to the model, or type exit to quit")
	conv = ConversationState{}
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\n\nBob: ")
		text, _ := reader.ReadString('\n')
		text = strings.Replace(text, "\n", "", -1)

		if text == "exit" {
			break
		}

		conv.UserText = append(conv.UserText, text)
		text = "\n\nBob: " + text + "\n\nAlice:"

		Model.ProcessInput(text)
		fmt.Print("\nAlice:")

		response_text := Model.GenerateResponse(100, "\n", 0.1, 0, func(s string) bool {
			fmt.Print(s)
			return true
		})

		conv.BotText = append(conv.BotText, response_text)

		//Save the conversation state to a file
		data, err := json.Marshal(conv)
		if err != nil {
			panic(err)
		}
		os.WriteFile("conversation.json", data, 0644)
	}

}
