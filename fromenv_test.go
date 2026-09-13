package llmwire

import (
	"errors"
	"strings"
	"testing"
)

func TestFromEnv(t *testing.T) {
	t.Run("a URL variable for a shipped host is the old contract and is refused", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "https://api.example/v4/")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "k")
		_, err := FromEnv("glm-5.3-flash", Config{})
		var se *StaleEnvError
		if !errors.As(err, &se) || se.Var != "LLMWIRE_ZAI_BASE_URL" || se.Provider != "zai" {
			t.Fatalf("got %v, want a StaleEnvError naming the variable", err)
		}
		if !strings.Contains(err.Error(), "delete the variable") {
			t.Fatalf("the error must say what to do: %v", err)
		}
	})

	t.Run("a Lookup replaces the environment and trims the slash", func(t *testing.T) {
		reg, err := NewRegistry([]byte("providers:\n  gw: {}\n" + chatHead + "    provider: gw\n"))
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("LLMWIRE_GW_BASE_URL", "")
		t.Setenv("LLMWIRE_GW_API_KEY", "")
		m := map[string]string{"LLMWIRE_GW_BASE_URL": "https://map.example/v1/", "LLMWIRE_GW_API_KEY": "m"}
		c, err := FromEnv("m", Config{Registry: reg, Lookup: func(k string) (string, bool) { v, ok := m[k]; return v, ok }})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://map.example/v1" || c.apiKey != "m" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("explicit base url leaves the environment alone", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "https://env.example")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "env")
		c, err := FromEnv("glm-5.3-flash", Config{BaseURL: "https://cfg.example"})
		if err != nil {
			t.Fatal(err)
		}
		// Not "env": the provider's key is for the provider's host.
		if c.baseURL != "https://cfg.example" || c.apiKey != "" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("explicit base url needs no variables at all", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "")
		c, err := FromEnv("glm-5.3-flash", Config{BaseURL: "https://cfg.example", APIKey: "cfg"})
		if err != nil {
			t.Fatal(err)
		}
		if c.apiKey != "cfg" {
			t.Fatalf("got %q", c.apiKey)
		}
	})

	t.Run("explicit api key wins over the environment", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "env")
		c, err := FromEnv("glm-5.3-flash", Config{APIKey: "cfg"})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL == "" || c.apiKey != "cfg" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("the shipped host is used when no variable overrides it", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "k")
		c, err := FromEnv("glm-5.3-flash", Config{})
		if err != nil {
			t.Fatal(err)
		}
		want, err := Default().Provider("zai")
		if err != nil {
			t.Fatal(err)
		}
		if want.BaseURL == "" || c.baseURL != want.BaseURL || c.apiKey != "k" {
			t.Fatalf("got %q %q, want the shipped host %q", c.baseURL, c.apiKey, want.BaseURL)
		}
	})

	t.Run("a provider that ships no host names the variable", func(t *testing.T) {
		reg, err := NewRegistry([]byte("providers:\n  gw: {}\n" + chatHead + "    provider: gw\n"))
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("LLMWIRE_GW_BASE_URL", "")
		t.Setenv("LLMWIRE_GW_API_KEY", "k")
		_, err = FromEnv("m", Config{Registry: reg})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "LLMWIRE_GW_BASE_URL" || me.Provider != "gw" || me.Model != "m" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("a provider absent from providers: is the same as one with no host", func(t *testing.T) {
		reg, err := NewRegistry([]byte(chatHead + "    provider: gw\n"))
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("LLMWIRE_GW_BASE_URL", "")
		t.Setenv("LLMWIRE_GW_API_KEY", "k")
		_, err = FromEnv("m", Config{Registry: reg})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "LLMWIRE_GW_BASE_URL" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("whitespace-only api key names the variable", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "")
		t.Setenv("LLMWIRE_ZAI_API_KEY", " ")
		_, err := FromEnv("glm-5.3-flash", Config{})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "LLMWIRE_ZAI_API_KEY" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("no_api_key sends no key and reads no key variable", func(t *testing.T) {
		reg, err := NewRegistry([]byte(chatHead + "    provider: local\n    no_api_key: true\n"))
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("LLMWIRE_LOCAL_BASE_URL", "http://localhost:8080/v1")
		t.Setenv("LLMWIRE_LOCAL_API_KEY", "")
		c, err := FromEnv("m", Config{Registry: reg})
		if err != nil {
			t.Fatal(err)
		}
		if c.apiKey != "" {
			t.Fatalf("got key %q", c.apiKey)
		}
	})

	t.Run("a profile without a provider is a profile bug, named", func(t *testing.T) {
		reg, err := NewRegistry([]byte(chatHead))
		if err != nil {
			t.Fatal(err)
		}
		_, err = FromEnv("m", Config{Registry: reg})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "" || me.Model != "m" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("unknown model", func(t *testing.T) {
		_, err := FromEnv("nope", Config{})
		var ue *UnknownModelError
		if !errors.As(err, &ue) {
			t.Fatalf("got %v", err)
		}
	})
}

// Every shipped profile names a provider, so FromEnv works for all of them;
// and the name is what the variable derives from.
func TestEveryEmbeddedProfileNamesAProvider(t *testing.T) {
	for _, id := range Default().Models() {
		p, err := Default().Lookup(id)
		if err != nil {
			t.Fatal(err)
		}
		if p.Provider == "" {
			t.Errorf("%s: no provider", id)
		}
		if p.BaseURLEnv() != "LLMWIRE_"+strings.ToUpper(p.Provider)+"_BASE_URL" {
			t.Errorf("%s: base url var %q", id, p.BaseURLEnv())
		}
	}
}

// no_api_key follows the provider: a derived profile that names a new host
// starts from "keyed", whatever its base said.
func TestFromEnv_newProviderResetsNoAPIKey(t *testing.T) {
	reg, err := NewRegistry([]byte(chatHead + "    provider: local\n    no_api_key: true\n" +
		"  - id: m-via\n    base: m\n    gateway: litellm\n    provider: litellm\n    wire_model_id: alias/m\n"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("LLMWIRE_LITELLM_BASE_URL", "https://gw.example")
	t.Setenv("LLMWIRE_LITELLM_API_KEY", "")
	_, err = FromEnv("m-via", Config{Registry: reg})
	var me *MissingEnvError
	if !errors.As(err, &me) || me.Var != "LLMWIRE_LITELLM_API_KEY" {
		t.Fatalf("got %v, want the gateway key demanded", err)
	}
}

func TestProfile_providerNameIsValidated(t *testing.T) {
	_, err := NewRegistry([]byte(chatHead + "    provider: Z-AI\n"))
	if err == nil {
		t.Fatal("a provider with a dash or capitals must be refused")
	}
}

// The shipped hosts: every vendor provider carries one, so an application
// needs only its key. litellm is the deliberate exception, a self-hosted
// gateway has no public host.
func TestEveryEmbeddedVendorProviderShipsAHost(t *testing.T) {
	reg := Default()
	for _, id := range reg.Models() {
		p, err := reg.Lookup(id)
		if err != nil {
			t.Fatal(err)
		}
		pv, err := reg.Provider(p.Provider)
		if err != nil {
			t.Errorf("%s: %v", id, err)
			continue
		}
		if p.Provider != "litellm" && pv.BaseURL == "" {
			t.Errorf("%s: provider %q ships no base_url; an application would have to know the host", id, p.Provider)
		}
		if p.BaseURL != pv.BaseURL {
			t.Errorf("%s: profile BaseURL %q does not match provider %q", id, p.BaseURL, pv.BaseURL)
		}
	}
	if _, err := reg.Provider("nope"); err == nil {
		t.Error("an unknown provider name must error, not read as no host")
	}
}

// The identity follows the provider: a host sold as opencode's backend gets
// the client string from FromEnv, and no application has to know to ask.
func TestFromEnv_identityFollowsTheProvider(t *testing.T) {
	t.Setenv("LLMWIRE_MIMO_BASE_URL", "")
	t.Setenv("LLMWIRE_MIMO_API_KEY", "k")
	t.Setenv("LLMWIRE_ZAI_API_KEY", "k")
	mimo, err := FromEnv("mimo-v2.5-pro", Config{})
	if err != nil {
		t.Fatal(err)
	}
	if mimo.session == nil {
		t.Error("mimo: provider says emulate_opencode, client presents as neutral")
	}
	zai, err := FromEnv("glm-5.3-flash", Config{})
	if err != nil {
		t.Fatal(err)
	}
	if zai.session != nil {
		t.Error("zai: provider does not emulate, client presents as opencode")
	}
}

// A derived profile's host follows ITS provider, not its base's: the route is
// what changes when a model is reached through a gateway.
func TestRegistry_derivedProfileTakesItsOwnProvidersHost(t *testing.T) {
	reg, err := NewRegistry([]byte("providers:\n  vendor:\n    base_url: https://vendor.example/v1\n  gw: {}\n" +
		chatHead + "    provider: vendor\n" +
		"  - id: m-via\n    base: m\n    gateway: litellm\n    provider: gw\n    wire_model_id: alias/m\n"))
	if err != nil {
		t.Fatal(err)
	}
	direct, _ := reg.Lookup("m")
	via, _ := reg.Lookup("m-via")
	if direct.BaseURL != "https://vendor.example/v1" || via.BaseURL != "" {
		t.Fatalf("direct %q via %q", direct.BaseURL, via.BaseURL)
	}
}

func TestRegistry_providerHostIsValidated(t *testing.T) {
	for name, doc := range map[string]string{
		"http":        "providers:\n  v:\n    base_url: http://vendor.example/v1\n",
		"query":       "providers:\n  v:\n    base_url: https://vendor.example/v1?api_key=x\n",
		"credentials": "providers:\n  v:\n    base_url: https://user:pw@vendor.example/v1\n",
		"no host":     "providers:\n  v:\n    base_url: https:///v1\n",
		"unknown key": "providers:\n  v:\n    url: https://vendor.example/v1\n",
		"bad name":    "providers:\n  Vendor-1:\n    base_url: https://vendor.example/v1\n",
		"chat route":  "providers:\n  v:\n    base_url: https://vendor.example/v1/chat/completions\n",
		"embed route": "providers:\n  v:\n    base_url: https://vendor.example/v1/embeddings/\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := NewRegistry([]byte(doc + chatHead)); err == nil {
				t.Fatal("must be refused at load")
			}
		})
	}
}

// A typo in provider: must fail the load, not surface later as a missing
// environment variable that sends the operator to the wrong file.
func TestRegistry_profileProviderMustExistWhenProvidersAreDeclared(t *testing.T) {
	_, err := NewRegistry([]byte("providers:\n  openai:\n    base_url: https://vendor.example/v1\n" +
		chatHead + "    provider: opeanai\n"))
	if err == nil || !strings.Contains(err.Error(), "opeanai") {
		t.Fatalf("got %v, want a load error naming the typo", err)
	}
}

// LLMWIRE_EMULATE_OPENCODE is the operator's switch: true adds the identity on
// a host the profiles do not mark, false takes nothing away from one they do,
// and a value that is not a boolean is refused with the variable named.
func TestFromEnv_emulateOpenCodeSwitch(t *testing.T) {
	t.Setenv("LLMWIRE_MIMO_API_KEY", "k")
	t.Setenv("LLMWIRE_ZAI_API_KEY", "k")

	t.Setenv(EnvEmulateOpenCode, "true")
	zai, err := FromEnv("glm-5.3-flash", Config{})
	if err != nil {
		t.Fatal(err)
	}
	if zai.session == nil {
		t.Error("zai with the switch on: client presents as neutral")
	}
	explicit, err := FromEnv("glm-5.3-flash", Config{BaseURL: "http://127.0.0.1:1"})
	if err != nil {
		t.Fatal(err)
	}
	if explicit.session == nil {
		t.Error("explicit BaseURL with the switch on: client presents as neutral")
	}

	t.Setenv(EnvEmulateOpenCode, "false")
	mimo, err := FromEnv("mimo-v2.5-pro", Config{})
	if err != nil {
		t.Fatal(err)
	}
	if mimo.session == nil {
		t.Error("mimo with the switch off: the provider flag must keep the identity")
	}
	zai, err = FromEnv("glm-5.3-flash", Config{})
	if err != nil {
		t.Fatal(err)
	}
	if zai.session != nil {
		t.Error("zai with the switch off: client presents as opencode")
	}

	t.Setenv(EnvEmulateOpenCode, "yes please")
	if _, err := FromEnv("glm-5.3-flash", Config{}); err == nil || !strings.Contains(err.Error(), EnvEmulateOpenCode) {
		t.Fatalf("garbage value: err = %v, want the variable named", err)
	}
}
