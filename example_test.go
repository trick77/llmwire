package llmwire_test

import (
	"fmt"

	"github.com/trick77/llmwire"
)

// The refusal a caller sees when they ask a model to stop thinking and it
// cannot. It arrives before any socket opens, and it names what the model does
// take — an error that only says "no" leaves the reader to go and find the
// accepted set, which for these models is not reliably documented anywhere.
func ExampleClient_Validate_unsupportedReasoning() {
	c := llmwire.New(llmwire.Config{})

	_, err := c.Validate(llmwire.ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []llmwire.Message{llmwire.User("hello")},
		Reasoning: llmwire.ReasoningOff(),
	})
	fmt.Println(err)

	// Output:
	// llmwire: model "glm-5.3-flash": thinking cannot be disabled on this model; accepted: low, high, max; or pass BestEffort to send the nearest supported request
}

// An effort level outside the model's accepted set is refused the same way. The
// set is per-model and narrower than the vendor's global enum: this model takes
// three of the seven levels its provider documents.
func ExampleClient_Validate_unsupportedEffort() {
	c := llmwire.New(llmwire.Config{})

	_, err := c.Validate(llmwire.ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []llmwire.Message{llmwire.User("hello")},
		Reasoning: llmwire.ReasoningEffort("medium"),
	})
	fmt.Println(err)

	// Output:
	// llmwire: model "glm-5.3-flash": effort "medium" is not accepted by this model; accepted: low, high, max; or pass BestEffort to send the nearest supported request
}

// A request that can be honoured approximately succeeds and reports a warning,
// rather than failing. Here the model cannot enforce a schema but can still be
// held to JSON.
func ExampleClient_Validate_warnings() {
	c := llmwire.New(llmwire.Config{})

	warnings, err := c.Validate(llmwire.ChatRequest{
		Model:       "mimo-v2.5-pro",
		Messages:    []llmwire.Message{llmwire.User("hello")},
		Temperature: ptr(0.2),
	})
	fmt.Println("error:", err)
	for _, w := range warnings {
		fmt.Printf("%s: %s\n", w.Kind, w.Feature)
	}

	// Output:
	// error: <nil>
	// compatibility: temperature
}

func ptr(f float64) *float64 { return &f }
