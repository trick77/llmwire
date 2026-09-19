package llmwire

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func modelsServer(t *testing.T, status int, body string, seen *http.Request) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if seen != nil {
			*seen = *r.Clone(context.Background())
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestListModels_ReadsGatewayLimits(t *testing.T) {
	var seen http.Request
	srv := modelsServer(t, 200, `{"object":"list","data":[
		{"id":"ai-gateway-gpt-5.5","object":"model","max_input_tokens":400000,"max_output_tokens":128000,"litellm_params":{"model":"azure/gpt-5.5"}},
		{"id":"plain","object":"model"}
	]}`, &seen)

	c := New(Config{BaseURL: srv.URL, APIKey: "team-key-of-no-known-shape"})
	entries, warnings, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(warnings) != 0 {
		t.Errorf("warnings = %v, want none", warnings)
	}
	if seen.Method != http.MethodGet || seen.URL.Path != "/models" {
		t.Errorf("request = %s %s, want GET /models", seen.Method, seen.URL.Path)
	}
	if seen.Header.Get("Authorization") != "Bearer team-key-of-no-known-shape" {
		t.Error("the key must go out as a bearer token")
	}
	if seen.Header.Get("Content-Type") != "" {
		t.Error("a GET must not declare a Content-Type")
	}
	if len(entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(entries))
	}
	e := entries[0]
	if e.ID != "ai-gateway-gpt-5.5" || e.MaxInputTokens == nil || *e.MaxInputTokens != 400000 ||
		e.MaxOutputTokens == nil || *e.MaxOutputTokens != 128000 {
		t.Errorf("entry = %+v", e)
	}
	if !strings.Contains(string(e.Raw), `"litellm_params"`) {
		t.Error("Raw must keep the fields this package does not model")
	}
	if p := entries[1]; p.ID != "plain" || p.MaxInputTokens != nil || p.MaxOutputTokens != nil {
		t.Errorf("plain entry = %+v, want nil limits and no warning", p)
	}
}

func TestListModels_UnusableLimitIsNilWithAWarning(t *testing.T) {
	cases := []struct {
		name, value string
		want        *int64
		warn        string
	}{
		{"integral float", "16385.0", ptr(int64(16385)), ""},
		{"null", "null", nil, ""},
		{"bool", "true", nil, "not a number"},
		{"string", `"lots"`, nil, "not a number"},
		{"quoted number", `"400000"`, nil, "not a number"},
		{"fraction", "16385.5", nil, "not an integer"},
		{"zero", "0", nil, "not positive"},
		{"negative", "-1", nil, "not positive"},
		{"huge", "1e30", nil, "out of range"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := modelsServer(t, 200, `{"data":[{"id":"m","max_input_tokens":`+tc.value+`}]}`, nil)
			c := New(Config{BaseURL: srv.URL})
			entries, warnings, err := c.ListModels(context.Background())
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(entries) != 1 {
				t.Fatalf("got %d entries, want 1", len(entries))
			}
			got := entries[0].MaxInputTokens
			switch {
			case tc.want == nil && got != nil:
				t.Errorf("MaxInputTokens = %d, want nil", *got)
			case tc.want != nil && (got == nil || *got != *tc.want):
				t.Errorf("MaxInputTokens = %v, want %d", got, *tc.want)
			}
			if tc.warn == "" {
				if len(warnings) != 0 {
					t.Errorf("warnings = %v, want none", warnings)
				}
				return
			}
			if len(warnings) != 1 || warnings[0].Feature != "max_input_tokens" ||
				!strings.Contains(warnings[0].Details, tc.warn) || !strings.Contains(warnings[0].Details, `"m"`) {
				t.Errorf("warnings = %v, want one naming the model and %q", warnings, tc.warn)
			}
		})
	}
}

func TestListModels_RowsWithoutAnIdAreSkippedNotFatal(t *testing.T) {
	srv := modelsServer(t, 200, `{"data":[{"object":"model"},"junk",{"id":"ok"}]}`, nil)
	c := New(Config{BaseURL: srv.URL})
	entries, warnings, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(entries) != 1 || entries[0].ID != "ok" {
		t.Errorf("entries = %+v, want only ok", entries)
	}
	if len(warnings) != 2 {
		t.Errorf("warnings = %v, want one per skipped row", warnings)
	}
}

func TestListModels_NoDataListIsAShapeError(t *testing.T) {
	for _, body := range []string{`{"object":"list"}`, `{"data":null}`, `{"data":{"id":"m"}}`} {
		srv := modelsServer(t, 200, body, nil)
		c := New(Config{BaseURL: srv.URL})
		_, _, err := c.ListModels(context.Background())
		if !errors.Is(err, ErrResponseShape) {
			t.Errorf("body %s: err = %v, want ErrResponseShape", body, err)
		}
	}
}

func TestListModels_ErrorStatusIsTypedAndRedacted(t *testing.T) {
	const key = "listing-key-of-no-known-shape"
	srv := modelsServer(t, 401, `{"error":{"message":"bad key `+key+`","type":"auth_error","code":"401"}}`, nil)
	c := New(Config{BaseURL: srv.URL, APIKey: key})
	_, _, err := c.ListModels(context.Background())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 401 {
		t.Fatalf("err = %v, want an APIError with status 401", err)
	}
	if strings.Contains(err.Error(), key) {
		t.Error("the key leaked into the error text")
	}
}

func TestListModels_HonoursTheCallerContext(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))
	defer srv.Close()
	defer close(block)

	c := New(Config{BaseURL: srv.URL})
	ctx, cancel := context.WithCancel(context.Background())
	go cancel()
	_, _, err := c.ListModels(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}

func ptr[T any](v T) *T { return &v }
