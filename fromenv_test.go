package llmwire

import (
	"errors"
	"strings"
	"testing"
)

func TestFromEnv(t *testing.T) {
	t.Run("reads both variables and trims the slash", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "https://api.example/v4/")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "k")
		c, err := FromEnv("glm-5.3-flash", Config{})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://api.example/v4" || c.apiKey != "k" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("a Lookup replaces the environment", func(t *testing.T) {
		t.Setenv("LLMWIRE_MIMO_BASE_URL", "")
		t.Setenv("LLMWIRE_MIMO_API_KEY", "")
		m := map[string]string{"LLMWIRE_MIMO_BASE_URL": "https://map.example", "LLMWIRE_MIMO_API_KEY": "m"}
		c, err := FromEnv("mimo-v2.5-pro", Config{Lookup: func(k string) (string, bool) { v, ok := m[k]; return v, ok }})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://map.example" || c.apiKey != "m" {
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
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "https://env.example")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "env")
		c, err := FromEnv("glm-5.3-flash", Config{APIKey: "cfg"})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://env.example" || c.apiKey != "cfg" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("unset base url names the variable and the provider", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "")
		t.Setenv("LLMWIRE_ZAI_API_KEY", "k")
		_, err := FromEnv("glm-5.3-flash", Config{})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "LLMWIRE_ZAI_BASE_URL" || me.Provider != "zai" || me.Model != "glm-5.3-flash" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("whitespace-only api key names the variable", func(t *testing.T) {
		t.Setenv("LLMWIRE_ZAI_BASE_URL", "https://api.example")
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

func TestProfile_providerNameIsValidated(t *testing.T) {
	_, err := NewRegistry([]byte(chatHead + "    provider: Z-AI\n"))
	if err == nil {
		t.Fatal("a provider with a dash or capitals must be refused")
	}
}
