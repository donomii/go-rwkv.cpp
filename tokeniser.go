package rwkv

import (
	"fmt"
	"strings"
	"io/ioutil"
	"encoding/json"
	"unicode"
	"os"
)

type Token struct {
	ID    int
	Value string
	Start int
	End   int
}


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


func Tokenize(input string, pipelineConfig Tokenizer) ([]Token, error) {

	for _, spesh := range pipelineConfig.AddedTokens {
		pipelineConfig.Model.Vocab[spesh.Content] = spesh.Id
	}

	// Normalization
	switch pipelineConfig.Normalizer.Type {
	case "NFC":
		//input = strings.ToLower(input)
	default:
		return nil, fmt.Errorf("Invalid normalizer type.")
	}

	//fmt.Println("Normalized input:", input)

	// Pre-tokenization
	switch pipelineConfig.PreTokenizer.Type {
	case "ByteLevel":
		input = ByteLevelPreTokenize(input, pipelineConfig.PreTokenizer.AddPrefixSpace)
	default:
		return nil, fmt.Errorf("Invalid pre-tokenizer type.")
	}

	//fmt.Println("Pre-tokenized input:", input)

	// Model
	var toks []string
	switch pipelineConfig.Model.Type {
	case "BPE":
		toks = BPETokenizeWithMerges(pipelineConfig, input)
	default:
		return nil, fmt.Errorf("Invalid model type.")
	}

	//fmt.Println("Tokenized input:", toks)

	// Post-processing
	var tokens []Token
	switch pipelineConfig.PostProcessor.Type {
	case "ByteLevel":
		tokens = ByteLevelDecode(toks, pipelineConfig)
	default:
		return nil, fmt.Errorf("Invalid post-processor type.")
	}

	//fmt.Println("Post-processed input:", tokens)

	return tokens, nil
}

func ByteLevelPreTokenize(input string, addPrefixSpace bool) string {
	if addPrefixSpace {
		input = " " + input
	}

	//Mark the beginning of words with "\u0120"
	//Beginnings of words occur between a word and a non-word character
	var lastChar rune
	var output string
	for _, char := range input {
		if !unicode.IsLetter(lastChar) && unicode.IsLetter(char) {
			output += "\u0120"
		}
		if char == '\n' {
			output += "Ċ"
		} else {
			output += string(char)
		}
		lastChar = char
	}

	return output
}

var detokenMap map[int]string

func DeTokenise(tk Tokenizer, tokens []int) string {
	if detokenMap == nil {
		detokenMap = make(map[int]string)
		for key, value := range tk.Model.Vocab {
			detokenMap[value] = key
		}
	}
	var output string
	for _, token := range tokens {
		if val, ok := detokenMap[token]; ok {
			output += strings.ReplaceAll(strings.Replace(val, "\u0120", " ", -1), "Ċ", "\n")
		}
	}
	return output
}

var pairMap map[string]string

func BPETokenizeWithMerges(tokenizer Tokenizer, text string) []string {
	merges := tokenizer.Model.Merges

	splits := strings.Split(text, "")

	if pairMap == nil {
		pairMap = make(map[string]string)
		for _, pairStr := range merges {
			pair := strings.Split(pairStr, " ")
			pairMap[pair[0]+pair[1]] = pair[0] + pair[1]
		}

	}

	for i := 0; i < len(splits); i++ {
		if i > 0 {
			prev := splits[i-1]
			current := splits[i]
			pair := prev + current
			if val, ok := pairMap[pair]; ok {
				splits[i-1] = val
				splits = append(splits[:i], splits[i+1:]...)
				i = i - 2
			}
		}
	}
	return splits

}

func ByteLevelDecode(in []string, tk Tokenizer) []Token {
	decodedTokens := []Token{}
	for _, tok := range in {
		val := strings.ReplaceAll(strings.ReplaceAll(tok, "\u0120", " "), "Ċ", "\n")
		num := tk.Model.Vocab[tok]
		if num != 0 {
			//fmt.Printf("Token: %s, ID: %d\n", val, num)
			decodedTokens = append(decodedTokens, Token{ID: num, Value: val, Start: 0, End: 0})
		}
	}
	//decodedTokens = append(decodedTokens, Token{ID: 0, Value: "<|End of document|>", Start: 0, End: 0})
	return decodedTokens
}
