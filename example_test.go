package llmwire_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"

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

// One completion, against a stub endpoint so the example is runnable. Warnings
// ride back alongside a successful response, which is the whole tier-3 policy: a
// coercion that worked is not an error, and a caller who ignores the middle value
// still gets a working call.
func ExampleClient_Chat() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Echo back which output-cap parameter arrived. On this model only
		// max_tokens is honoured — max_completion_tokens is ACCEPTED and then
		// ignored, which is why the caller never chooses the spelling.
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		_, capped := body["max_tokens"]
		fmt.Println("max_tokens sent:", capped)

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"model":"glm-5.3-flash","choices":[{"finish_reason":"stop",`+
			`"message":{"content":"four"}}],`+
			`"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
	}))
	defer srv.Close()

	c := llmwire.New(llmwire.Config{BaseURL: srv.URL})
	cap := 16
	resp, warnings, err := c.Chat(context.Background(), llmwire.ChatRequest{
		Model:     "glm-5.3-flash",
		Messages:  []llmwire.Message{llmwire.User("What is two plus two?")},
		MaxTokens: &cap,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("answer:", resp.Content)
	fmt.Println("finish:", resp.FinishReason)
	fmt.Println("cost:", resp.Usage.Cost.NanoUSD, "nano-USD via", resp.Usage.Cost.Provenance)
	fmt.Println("warnings:", len(warnings))

	// Output:
	// max_tokens sent: true
	// answer: four
	// finish: stop
	// cost: 2500 nano-USD via from-table
	// warnings: 0
}

// Streaming is an iterator. Close is always correct to defer, including after
// reading to the end, and the accumulated result plus its usage and cost are
// available once the loop finishes — usage arrives in the final chunk, so there is
// nowhere earlier for it to be.
func ExampleClient_ChatStream() {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, frame := range []string{
			`{"choices":[{"delta":{"content":"two "}}]}`,
			`{"choices":[{"delta":{"content":"plus two"}},{"delta":{}}]}`,
			`{"choices":[{"delta":{"content":" is four"},"finish_reason":"stop"}]}`,
			`[DONE]`,
		} {
			fmt.Fprintf(w, "data: %s\n\n", frame)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer srv.Close()

	c := llmwire.New(llmwire.Config{BaseURL: srv.URL})
	stream, _, err := c.ChatStream(context.Background(), llmwire.ChatRequest{
		Model:    "glm-5.3-flash",
		Messages: []llmwire.Message{llmwire.User("What is two plus two?")},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	defer stream.Close()

	for stream.Next() {
		if ev := stream.Event(); ev.Kind == llmwire.EventContent {
			fmt.Print(ev.Text)
		}
	}
	fmt.Println()
	if err := stream.Err(); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("finish:", stream.Result().FinishReason)

	// Output:
	// two plus two is four
	// finish: stop
}

func ptr(f float64) *float64 { return &f }
