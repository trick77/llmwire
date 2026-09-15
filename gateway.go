package llmwire

import (
	"fmt"
	"strings"
)

// GatewayModelsEnv lists the profiles a self-hosted gateway serves, and under
// which names. Comma-separated `<profile id>[=<name on the gateway>]`; a bare
// id means the gateway takes the public name unchanged. A listed profile is
// reached through the litellm provider (LLMWIRE_LITELLM_BASE_URL and
// LLMWIRE_LITELLM_API_KEY), an unlisted one through its own vendor as before.
//
// The names are the operator's, never the library's: a gateway is a
// deployment, its aliases are that deployment's, and putting them in
// profiles.yaml would publish one company's proxy layout as everyone's. What
// the library owns is the model behind the alias, which is why the left-hand
// side must be a profile id: the route inherits that profile's capabilities
// and validates against them, exactly as a YAML route does.
const GatewayModelsEnv = "LLMWIRE_LITELLM_MODELS"

// gatewayProvider is the provider every env-declared route goes through. It is
// the one provider profiles.yaml ships without a host, so its URL comes from
// the environment by the existing rule.
const gatewayProvider = "litellm"

// parseGatewayModels reads GatewayModelsEnv's value into id → wire name. An
// entry that is not `id` or `id=name` is an error naming it, never skipped:
// a typo that silently falls through would route the model to its vendor
// with the vendor's key, which may not even be configured.
func parseGatewayModels(v string) (map[string]string, error) {
	out := map[string]string{}
	for _, raw := range strings.Split(v, ",") {
		entry := strings.TrimSpace(raw)
		if entry == "" {
			continue
		}
		id, name, hasName := strings.Cut(entry, "=")
		id = strings.TrimSpace(id)
		name = strings.TrimSpace(name)
		if id == "" || (hasName && name == "") || strings.Contains(name, "=") {
			return nil, fmt.Errorf("llmwire: %s entry %q is not <profile id>[=<gateway name>]", GatewayModelsEnv, entry)
		}
		if !hasName {
			name = id
		}
		if _, dup := out[id]; dup {
			return nil, fmt.Errorf("llmwire: %s lists %q twice", GatewayModelsEnv, id)
		}
		out[id] = name
	}
	return out, nil
}

// viaGateway returns a copy of the registry in which the named profile is
// reached through the gateway under wireModelID. The entry keeps its id, so a
// caller that names the model keeps naming it; what changes is the route: the
// litellm provider, the alias on the wire, and no cost block, since a proxy
// prices its own calls (see Profile.resolve).
//
// Built the way a YAML route is, through resolve and validate against the
// base, so an env-declared route cannot claim what a file-declared one could
// not. The base must be unbased for the same reason the loader demands it;
// that includes a registry this function already routed, so a Client's own
// Registry() handed back in as Config.Registry is refused rather than routed
// twice. FromEnv starts from Default() or the caller's untouched document.
func (r *Registry) viaGateway(id, wireModelID string) (*Registry, error) {
	base, ok := r.byID[id]
	if !ok {
		return nil, &UnknownModelError{ID: id, Known: sortedIDs(r.byID)}
	}
	if base.Base != "" {
		return nil, fmt.Errorf("llmwire: %s names %q, which is already a route (base %q); list the model, not a route",
			GatewayModelsEnv, id, base.Base)
	}
	pv, ok := r.providers[gatewayProvider]
	if !ok {
		return nil, fmt.Errorf("llmwire: this registry has no %q provider, so %s cannot route through it",
			gatewayProvider, GatewayModelsEnv)
	}
	route := Profile{ID: id, Base: id, Gateway: gatewayProvider, Provider: gatewayProvider, WireModelID: wireModelID}
	// The stored base already carries its defaults, BaseURL and
	// EmulateOpenCode; resolve replaces provider and clears cost, and the
	// host fields are re-read from the gateway's provider below.
	p := route.resolve(*base)
	applyDefaults(&p)
	if err := p.validate(); err != nil {
		return nil, fmt.Errorf("llmwire: %s route for %q: %w", GatewayModelsEnv, id, err)
	}
	p.BaseURL = pv.BaseURL
	p.EmulateOpenCode = pv.EmulateOpenCode

	out := &Registry{byID: make(map[string]*Profile, len(r.byID)), providers: r.providers}
	for k, v := range r.byID {
		out.byID[k] = v
	}
	out.byID[id] = &p
	return out, nil
}
