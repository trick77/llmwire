package llmwire

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
)

// One client for several models on several hosts.
//
// An application with a chat lane on one vendor and an embeddings or vision
// lane on another would otherwise build one client per host and pick among
// them by model, which is the provider table this library exists to hold.
// FromEnvModels builds that table from the profiles: each model's requests go
// to its provider's host with its provider's key, and swapping a lane's model is
// a config change even when the new model lives elsewhere.

// FromEnvModels builds one Client serving exactly the models named, each
// reached the way FromEnv would reach it: host from its provider (or
// LLMWIRE_<PROVIDER>_BASE_URL where the provider ships none, refused where it
// does), key from LLMWIRE_<PROVIDER>_API_KEY, LLMWIRE_LITELLM_MODELS routing
// and LLMWIRE_EMULATE_OPENCODE applied as there. Models sharing a provider
// share one connection pool and one opencode session identity, except that its
// no_api_key profiles, which send no key, get a client of their own.
//
// Every unset variable is named in one error (errors.Join of
// *MissingEnvError, each variable once), so a deployment is fixed in one pass.
// A request for a model not named here is a *ModelNotConfiguredError, before
// any socket opens: the list is the contract, not whichever host a key happens
// to open.
//
// cfg is FromEnv's: an explicit cfg.BaseURL sends every model to that one host
// with cfg.APIKey as given and reads no variable (a test fake, a stand-in). A
// cfg.APIKey with no BaseURL is refused when the models span more than one
// provider, since it would carry one vendor's key to another's host.
//
// Stats() and Registry() cover every model; ListModels, RawStream and RawPost
// need one host, which ForModel returns.
func FromEnvModels(cfg Config, models ...string) (*Client, error) {
	if len(models) == 0 {
		return nil, errors.New("llmwire: FromEnvModels needs at least one model")
	}
	reg := cfg.Registry
	if reg == nil {
		reg = Default()
	}
	for _, m := range models {
		if _, err := reg.Lookup(m); err != nil {
			return nil, err
		}
	}
	ids := slices.Clone(models)
	slices.Sort(ids)
	ids = slices.Compact(ids)

	type group struct {
		cfg    Config
		models []string
		ep     []envResolved
	}
	groups := map[string]*group{}
	var order []string
	add := func(key string, gcfg Config, m string, ep envResolved) {
		g, ok := groups[key]
		if !ok {
			g = &group{cfg: gcfg}
			groups[key] = g
			order = append(order, key)
		}
		g.models = append(g.models, m)
		g.ep = append(g.ep, ep)
	}

	var missing []error
	if cfg.BaseURL != "" {
		cfg.Registry = reg
		for _, m := range ids {
			p, _ := reg.Lookup(m)
			add("", cfg, m, envResolved{cfg: cfg, profile: p})
		}
	} else {
		get := envGetter(cfg.Lookup)
		var routed bool
		var err error
		if reg, routed, err = routeGateway(reg, get); err != nil {
			return nil, err
		}
		cfg.Registry = reg
		named, providers := map[string]bool{}, map[string]bool{}
		for _, m := range ids {
			ep, err := envEndpoint(m, cfg, get, routed)
			if err != nil {
				return nil, err
			}
			for _, e := range ep.missing {
				// One line per variable, however many models need it.
				key := e.Error()
				if me := (*MissingEnvError)(nil); errors.As(e, &me) && me.Var != "" {
					key = me.Var
				}
				if !named[key] {
					named[key] = true
					missing = append(missing, e)
				}
			}
			// no_api_key is per profile, so a keyless model on a keyed
			// provider gets a client of its own: one shared client would
			// either drop the key for the keyed models or send it to the
			// keyless one.
			key := ep.profile.Provider
			if ep.profile.NoAPIKey {
				key += "/no_api_key"
			}
			providers[ep.profile.Provider] = true
			add(key, ep.cfg, m, ep)
		}
		if cfg.APIKey != "" && len(providers) > 1 {
			return nil, fmt.Errorf("llmwire: Config.APIKey is set, but models %v span providers %v; "+
				"leave it empty so each provider's LLMWIRE_<PROVIDER>_API_KEY is read", ids, sortedKeys(providers))
		}
	}
	if len(missing) > 0 {
		return nil, errors.Join(missing...)
	}

	// The front client holds no host, no key and no session: every call is
	// handed to a provider's client before it could use one.
	front := cfg
	front.BaseURL, front.APIKey, front.EmulateOpenCode = "", "", false
	c := New(front)
	c.serving = setOf(ids)
	c.routes = make(map[string]*Client, len(ids))
	for _, key := range order {
		g := groups[key]
		sub := New(g.cfg)
		sub.stats = c.stats
		sub.serving = setOf(g.models)
		for i, m := range g.models {
			c.routes[m] = sub
			sub.logSettings(m, g.ep[i].profile, g.ep[i].keyVar)
		}
	}
	return c, nil
}

func setOf(ids []string) map[string]bool {
	out := make(map[string]bool, len(ids))
	for _, id := range ids {
		out[id] = true
	}
	return out
}

// ModelNotConfiguredError is a request for a model the client was not built
// for. Named rather than routed anywhere: a model missing from the list is a
// configuration mistake, and guessing a host for it could send it with another
// provider's key.
type ModelNotConfiguredError struct {
	Model      string
	Configured []string
}

func (e *ModelNotConfiguredError) Error() string {
	return fmt.Sprintf("llmwire: model %q is not one this client was built for; it serves %s",
		e.Model, strings.Join(e.Configured, ", "))
}

// checkServes refuses a model outside the client's list, where it has one.
func (c *Client) checkServes(model string) error {
	if c.serving == nil || c.serving[model] {
		return nil
	}
	return &ModelNotConfiguredError{Model: model, Configured: sortedKeys(c.serving)}
}

// ForModel returns the client that talks to model's host: itself on a New or
// FromEnv client, the model's provider client on a FromEnvModels one. That
// client serves only the models on its provider, so it cannot carry its key to
// another host. Use it for the host-level calls: ListModels, RawStream,
// RawPost.
func (c *Client) ForModel(model string) (*Client, error) {
	if err := c.checkServes(model); err != nil {
		return nil, err
	}
	if c.routes == nil {
		return c, nil
	}
	return c.routes[model], nil
}

// errSeveralHosts is a host-level call on a client that has several.
func (c *Client) errSeveralHosts(call string) error {
	return fmt.Errorf("llmwire: %s needs one host, and this client serves %s across several; call it on ForModel(model)",
		call, strings.Join(sortedKeys(c.serving), ", "))
}

// The dispatch every model-level entry point starts with on a FromEnvModels
// client. Kept beside each other so a new entry point is added to all of them.

func (c *Client) routedChat(ctx context.Context, req ChatRequest) (*ChatResponse, []Warning, error) {
	sub, err := c.ForModel(req.Model)
	if err != nil {
		return nil, nil, err
	}
	return sub.Chat(ctx, req)
}

func (c *Client) routedChatStream(ctx context.Context, req ChatRequest) (*Stream, []Warning, error) {
	sub, err := c.ForModel(req.Model)
	if err != nil {
		return nil, nil, err
	}
	return sub.ChatStream(ctx, req)
}

func (c *Client) routedEmbed(ctx context.Context, req EmbedRequest) (*EmbedResponse, []Warning, error) {
	sub, err := c.ForModel(req.Model)
	if err != nil {
		return nil, nil, err
	}
	return sub.Embed(ctx, req)
}
