package llmwire

import (
	"bytes"
	"errors"
	"log/slog"
	"strings"
	"testing"
)

// gatewayEnv is a Lookup over a map: the litellm pair plus whatever list the
// test declares, nothing from the process.
func gatewayEnv(models string) func(string) (string, bool) {
	m := map[string]string{
		"LLMWIRE_LITELLM_BASE_URL": "https://gw.example/v1/",
		"LLMWIRE_LITELLM_API_KEY":  "gw-key",
		GatewayModelsEnv:           models,
	}
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

func TestParseGatewayModels(t *testing.T) {
	got, err := parseGatewayModels(" gpt-5.4-mini = ai-gateway-gpt-5.4-mini ,text-embedding-3-small,, ")
	if err != nil {
		t.Fatal(err)
	}
	if got["gpt-5.4-mini"] != "ai-gateway-gpt-5.4-mini" || got["text-embedding-3-small"] != "text-embedding-3-small" || len(got) != 2 {
		t.Fatalf("got %v", got)
	}
	for _, bad := range []string{"=alias", "id=", "id=a=b", "id,id"} {
		if _, err := parseGatewayModels(bad); err == nil || !strings.Contains(err.Error(), GatewayModelsEnv) {
			t.Errorf("%q: err = %v, want one naming the variable", bad, err)
		}
	}
}

func TestFromEnv_gatewayModels(t *testing.T) {
	t.Run("a listed model goes to the gateway under its alias", func(t *testing.T) {
		c, err := FromEnv("mimo-v2.5-pro", Config{Lookup: gatewayEnv("mimo-v2.5-pro=proxy-mimo")})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://gw.example/v1" || c.apiKey != "gw-key" {
			t.Fatalf("got %q %q, want the gateway's host and key", c.baseURL, c.apiKey)
		}
		p, err := c.Registry().Lookup("mimo-v2.5-pro")
		if err != nil {
			t.Fatal(err)
		}
		if p.WireModelID != "proxy-mimo" || p.Provider != "litellm" || p.Gateway != "litellm" {
			t.Fatalf("route = %s", p)
		}
		// The route inherits the model, not its price: a proxy prices its own
		// calls. And not its provenance: nobody probed the route.
		if p.Cost != nil || p.Verified == VerifiedMeasured {
			t.Fatalf("cost %v verified %q: a route carries neither", p.Cost, p.Verified)
		}
		if !p.Reasoning.Supported {
			t.Fatal("capabilities must be inherited whole")
		}
		// MiMo's host wants the opencode identity; the gateway is not MiMo's host.
		if c.session != nil {
			t.Fatal("the gateway must not inherit the vendor's identity")
		}
	})

	t.Run("a bare id keeps the public name on the wire", func(t *testing.T) {
		c, err := FromEnv("text-embedding-3-small", Config{Lookup: gatewayEnv("text-embedding-3-small")})
		if err != nil {
			t.Fatal(err)
		}
		p, _ := c.Registry().LookupEmbedding("text-embedding-3-small")
		if p.WireModelID != "text-embedding-3-small" || p.Embedding.DefaultDimensions != 1536 {
			t.Fatalf("route = %s, dims %d", p, p.Embedding.DefaultDimensions)
		}
	})

	t.Run("an unlisted model is untouched", func(t *testing.T) {
		t.Setenv("LLMWIRE_MIMO_BASE_URL", "")
		m := gatewayEnv("glm-5.3-flash=proxy-glm")
		c, err := FromEnv("mimo-v2.5-pro", Config{Lookup: func(k string) (string, bool) {
			if k == "LLMWIRE_MIMO_API_KEY" {
				return "mimo-key", true
			}
			return m(k)
		}})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://token-plan-sgp.xiaomimimo.com/v1" || c.apiKey != "mimo-key" {
			t.Fatalf("got %q %q, want MiMo's own host and key", c.baseURL, c.apiKey)
		}
		if Default().byID["mimo-v2.5-pro"].Provider != "mimo" {
			t.Fatal("the shared registry was mutated")
		}
	})

	t.Run("the gateway's URL and key are required and named", func(t *testing.T) {
		for _, missing := range []string{"LLMWIRE_LITELLM_BASE_URL", "LLMWIRE_LITELLM_API_KEY"} {
			full := gatewayEnv("mimo-v2.5-pro")
			_, err := FromEnv("mimo-v2.5-pro", Config{Lookup: func(k string) (string, bool) {
				if k == missing {
					return "", false
				}
				return full(k)
			}})
			var me *MissingEnvError
			if !errors.As(err, &me) || me.Var != missing {
				t.Errorf("without %s: err = %v, want a MissingEnvError naming it", missing, err)
			}
		}
	})

	t.Run("an unknown id in the list is an error even for another model", func(t *testing.T) {
		// The list is validated as a whole on every build: a typo'd entry
		// would otherwise let the model it meant fall through to its vendor,
		// bypassing the gateway with nothing said.
		_, err := FromEnv("mimo-v2.5-pro", Config{Lookup: gatewayEnv("mimo-v2.5-pr=x")})
		var ue *UnknownModelError
		if !errors.As(err, &ue) || !strings.Contains(err.Error(), GatewayModelsEnv) {
			t.Fatalf("err = %v, want UnknownModelError naming the variable", err)
		}
		if _, err := FromEnv("mimo-v2.5-pro", Config{Lookup: gatewayEnv("mimo-v2.5-pro=a=b")}); err == nil {
			t.Fatal("a malformed entry must fail the build")
		}
	})

	t.Run("a route validates against its model", func(t *testing.T) {
		reg := registryFrom(t, `providers:
  litellm: {}
  vendor:
    base_url: https://vendor.example/v1
profiles:
  - id: forced
    provider: vendor
    wire_model_id: forced
    max_tokens_param: max_completion_tokens
    verified: measured
    temperature: {supported: true, forced_value: 1.0}
    streaming: {supported: true, accepts_stream_options: true}
`)
		c, err := FromEnv("forced", Config{Registry: reg, Lookup: gatewayEnv("forced=proxy-forced")})
		if err != nil {
			t.Fatal(err)
		}
		off := 0.3
		if _, err := c.Validate(ChatRequest{Model: "forced", Messages: []Message{User("hi")}, Temperature: &off}); err == nil {
			t.Fatal("the route must refuse what its model refuses, locally")
		}
	})

	t.Run("an explicit base url still wins over the list", func(t *testing.T) {
		c, err := FromEnv("mimo-v2.5-pro", Config{BaseURL: "https://cfg.example", Lookup: gatewayEnv("mimo-v2.5-pro=proxy")})
		if err != nil {
			t.Fatal(err)
		}
		p, _ := c.Registry().Lookup("mimo-v2.5-pro")
		if c.baseURL != "https://cfg.example" || p.WireModelID != "mimo-v2.5-pro" {
			t.Fatalf("got %q wire %q", c.baseURL, p.WireModelID)
		}
	})

	t.Run("the log line names the route", func(t *testing.T) {
		var buf bytes.Buffer
		log := slog.New(slog.NewTextHandler(&buf, nil))
		if _, err := FromEnv("mimo-v2.5-pro", Config{Logger: log, Lookup: gatewayEnv("mimo-v2.5-pro=proxy-mimo")}); err != nil {
			t.Fatal(err)
		}
		for _, want := range []string{"wire_model=proxy-mimo", "gateway=litellm", "provider=litellm", "base_url=https://gw.example/v1", "api_key=LLMWIRE_LITELLM_API_KEY"} {
			if !strings.Contains(buf.String(), want) {
				t.Errorf("missing %q in %s", want, buf.String())
			}
		}
	})
}

func TestRegistry_viaGatewayRefusesARoute(t *testing.T) {
	reg := registryFrom(t, `providers:
  litellm: {}
  vendor:
    base_url: https://vendor.example/v1
profiles:
  - id: m
    provider: vendor
    wire_model_id: m
    max_tokens_param: max_tokens
    verified: measured
  - id: r
    base: m
    gateway: litellm
    provider: litellm
    wire_model_id: alias
`)
	if _, err := reg.viaGateway("r", "x"); err == nil || !strings.Contains(err.Error(), "already a route") {
		t.Fatalf("err = %v", err)
	}
}
