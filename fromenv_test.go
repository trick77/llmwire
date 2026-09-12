package llmwire

import (
	"errors"
	"testing"
)

func TestFromEnv(t *testing.T) {
	t.Run("reads both variables and trims the slash", func(t *testing.T) {
		t.Setenv("BACKEND_CHAT_BASE_URL", "https://api.example/v4/")
		t.Setenv("BACKEND_CHAT_API_KEY", "k")
		c, err := FromEnv("glm-5.3-flash", Config{})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://api.example/v4" || c.apiKey != "k" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("explicit cfg wins over the environment", func(t *testing.T) {
		t.Setenv("BACKEND_CHAT_BASE_URL", "https://env.example")
		t.Setenv("BACKEND_CHAT_API_KEY", "env")
		c, err := FromEnv("glm-5.3-flash", Config{BaseURL: "https://cfg.example", APIKey: "cfg"})
		if err != nil {
			t.Fatal(err)
		}
		if c.baseURL != "https://cfg.example" || c.apiKey != "cfg" {
			t.Fatalf("got %q %q", c.baseURL, c.apiKey)
		}
	})

	t.Run("unset base url names the variable", func(t *testing.T) {
		t.Setenv("BACKEND_CHAT_BASE_URL", "")
		t.Setenv("BACKEND_CHAT_API_KEY", "k")
		_, err := FromEnv("glm-5.3-flash", Config{})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "BACKEND_CHAT_BASE_URL" || me.Model != "glm-5.3-flash" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("whitespace-only api key names the variable", func(t *testing.T) {
		t.Setenv("BACKEND_CHAT_BASE_URL", "https://api.example")
		t.Setenv("BACKEND_CHAT_API_KEY", " ")
		_, err := FromEnv("glm-5.3-flash", Config{})
		var me *MissingEnvError
		if !errors.As(err, &me) || me.Var != "BACKEND_CHAT_API_KEY" {
			t.Fatalf("got %v", err)
		}
	})

	t.Run("profile without api_key_env sends no key", func(t *testing.T) {
		reg, err := NewRegistry([]byte(chatHead + "    base_url_env: LOCAL_URL\n"))
		if err != nil {
			t.Fatal(err)
		}
		t.Setenv("LOCAL_URL", "http://localhost:8080/v1")
		c, err := FromEnv("m", Config{Registry: reg})
		if err != nil {
			t.Fatal(err)
		}
		if c.apiKey != "" {
			t.Fatalf("got key %q", c.apiKey)
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
