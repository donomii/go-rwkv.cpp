package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

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
	Model := rwkv.LoadFiles("../aimodels/small.bin", "../rwkv.cpp/python/20B_tokenizer.json", 8)
	// Read lines from stdin, and submit them to the model, until the user types exit

	fmt.Println("Enter text to send to the model, or type exit to quit")
	reader := bufio.NewReader(os.Stdin)

	for {
		text, _ := reader.ReadString('\n')
		text = strings.Replace(text, "\n", "", -1)

		if text == "exit" {
			break
		}

		text = `Q & A

Question:
` + text + `

Detailed Expert Answer:
`

		Model.ProcessInput(text)

		response_text := Model.GenerateResponse(100, "\n", 1.2, 0.5, func(s string) bool {
			//fmt.Print(s)
			return true
		})
		fmt.Println(response_text)

	}

}
