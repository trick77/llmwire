package llmwire

import (
	"fmt"
	"sort"
	"strings"
)

// Model profiles: what each model on the other end of an OpenAI-compatible
// endpoint actually accepts.
//
// The protocol is one wire format, but no two deployments implement the same
// subset of it, and the differences are not discoverable from the spec. Holding
// them as data rather than as branches at call sites is what lets a wrong
// assumption fail locally, before a request is sent, instead of arriving as a
// 400 three network hops away — or worse, as a parameter silently ignored.
//
// Every field here is measured, not read. Where a vendor doc and a measurement
// disagree the measurement wins; see FINDINGS.md, where six such disagreements
// are recorded. A field whose value has not been measured is absent, and absent
// means "assume nothing", never "assume the default".

// Endpoint is the route a profile is for. It gates the whole request path, and
// separating it means a tool or a temperature aimed at an embeddings model is
// refused by the schema rather than by the endpoint.
type Endpoint string

const (
	EndpointChat       Endpoint = "chat"
	EndpointEmbeddings Endpoint = "embeddings"
)

// ReasoningControl is HOW a model's reasoning depth is expressed on the wire.
// Models that reason differ in the knob, not just in its values: one takes a
// nested object, another an effort string, a third a token budget.
type ReasoningControl string

const (
	ControlNone ReasoningControl = "none"
	// ControlToggleObject is thinking:{"type":"enabled"|"disabled"}.
	ControlToggleObject ReasoningControl = "toggle_object"
	// ControlEffort is reasoning_effort:"<level>".
	ControlEffort ReasoningControl = "effort"
	// ControlBudget is a token budget. No model in the registry uses it yet; it
	// exists so adding one is data rather than a new concept.
	ControlBudget ReasoningControl = "budget_tokens"
)

// ToolFormat is how a model emits a tool call.
type ToolFormat string

const (
	// FormatNative populates the OpenAI tool_calls field.
	FormatNative ToolFormat = "native"
	// FormatXML emits markup inside the content stream instead. Measured as NOT
	// occurring on the MiMo deployments (see FINDINGS.md), but retained because
	// loom observed it in production against a longer history than a probe
	// reproduces, and because deleting the capability would make its return a
	// rewrite rather than a flag.
	FormatXML ToolFormat = "xml"
)

// Reasoning describes a model's thinking controls.
//
// EnabledByDefault and CanBeDisabled are SEPARATE booleans, and that is the most
// important decision in this file. "Can this model stop reasoning?" is not
// answerable from whether "none" appears in a list of effort levels: published
// catalogues infer exactly that and get glm-5.3-flash wrong, because it rejects
// "none" AND rejects the disable toggle AND rejects three other levels, all with
// one overloaded error code. Three separate projects independently concluded the
// same thing and gave the flag three different names.
type Reasoning struct {
	Supported bool `yaml:"supported"`
	// EnabledByDefault: the model reasons when the knob is omitted entirely.
	EnabledByDefault bool `yaml:"enabled_by_default"`
	// CanBeDisabled: the model accepts being told not to reason at all.
	CanBeDisabled bool             `yaml:"can_be_disabled"`
	Control       ReasoningControl `yaml:"control"`
	// EffortValues is the exact accepted set, never a superset. The vendor's
	// global enum is wider than any single model takes: glm-5.3-flash accepts
	// three of the seven values its provider documents.
	EffortValues []string `yaml:"effort_values"`
	// DefaultEffort is what the model uses when the field is omitted. Empty
	// means the vendor does not say.
	DefaultEffort string `yaml:"default_effort"`
	// StreamField names the delta field reasoning arrives on: "reasoning_content"
	// on every model measured so far, "reasoning" on some servers. Empty means
	// reasoning is not streamed separately.
	StreamField string `yaml:"stream_field"`
	// BudgetParam is the wire key a token budget is sent under, and is required
	// when Control is ControlBudget. There is no default because the vendors
	// that take a budget disagree on the name, and guessing it would send a key
	// the endpoint ignores — the same silent failure max_tokens_param exists to
	// prevent. Must be empty for every other control.
	BudgetParam string `yaml:"budget_param"`
}

// Accepts reports whether effort is in the model's accepted set.
func (r Reasoning) Accepts(effort string) bool {
	for _, v := range r.EffortValues {
		if v == effort {
			return true
		}
	}
	return false
}

// Sampling describes temperature and top_p handling, which is subtler than a
// bool on every model in the registry.
type Sampling struct {
	// Supported: the parameter may be sent at all.
	Supported bool `yaml:"supported"`
	// ForcedValue, when non-nil, is the ONLY value accepted. Sending anything
	// else is a 400. This is distinct from unsupported: omitting the parameter
	// is fine, and sending the forced value is fine.
	ForcedValue *float64 `yaml:"forced_value"`
	// InertWhileReasoning: the parameter is accepted and then overridden by the
	// model while it is thinking. Neither rejected nor honoured — which is why
	// it cannot be expressed as Supported alone. Sending it is harmless;
	// believing it took effect is not.
	InertWhileReasoning bool `yaml:"inert_while_reasoning"`
	// RecommendedValue is the vendor's tuned operating point, sent when the
	// caller expresses no preference. It matters because omitting the parameter
	// does NOT mean "the model's default" on every endpoint: at least one falls
	// back to a materially lower value than the model was tuned for.
	RecommendedValue *float64 `yaml:"recommended_value"`
}

// Tools describes tool-calling support.
type Tools struct {
	Supported bool       `yaml:"supported"`
	Format    ToolFormat `yaml:"format"`
	// ToolChoiceValues is the accepted set. Several endpoints take only "auto"
	// and SILENTLY DROP anything else, so a caller asking for a forced call gets
	// an unforced one with no error.
	ToolChoiceValues []string `yaml:"tool_choice_values"`
	// SupportsForcedChoice is whether a specific tool can be compelled. Always
	// false for FormatXML: with no tool_calls field there is nothing for the
	// server to force.
	SupportsForcedChoice bool `yaml:"supports_forced_choice"`
	SupportsParallel     bool `yaml:"supports_parallel"`
	// RecoverInlineMarkup runs the inline-XML recovery even when Format is
	// native. Off for every profile measured; kept because loom observed markup
	// leaking from a deployment that otherwise returns native calls.
	RecoverInlineMarkup bool `yaml:"recover_inline_markup"`
}

// Output describes structured-output support. json_object and json_schema are
// tracked separately because they are separately supported, and because a
// downgrade from one to the other is a real, lossy coercion a caller deserves to
// be warned about.
type Output struct {
	JSONObject bool `yaml:"json_object"`
	JSONSchema bool `yaml:"json_schema"`
	// StrictSchema is whether "strict": true is accepted inside a json_schema
	// block. Some compat servers reject it.
	StrictSchema bool `yaml:"strict_schema"`
}

// Streaming describes SSE behaviour.
type Streaming struct {
	Supported bool `yaml:"supported"`
	// NeedsIncludeUsage: without stream_options.include_usage the endpoint
	// reports no usage at all. Measured false on both direct endpoints in the
	// registry; forced true behind a gateway that strips the usage chunk from
	// a client that did not ask for it.
	NeedsIncludeUsage bool `yaml:"needs_include_usage"`
	// AcceptsStreamOptions: whether the parameter may be sent. An endpoint that
	// rejects it must never receive it, even when harmless elsewhere.
	AcceptsStreamOptions bool `yaml:"accepts_stream_options"`
}

// Embedding describes embeddings-only behaviour. Separate from the chat
// capabilities so a chat profile declaring it fails the load, the same way an
// embeddings profile declaring tools does.
type Embedding struct {
	// Dimensions: the model accepts the dimensions parameter, which shortens the
	// returned vectors. Only the -3 generation does; older models reject it
	// outright, so this cannot be inferred from the endpoint.
	Dimensions bool `yaml:"dimensions"`
	// DefaultDimensions is the length of a vector this model returns when the
	// dimensions parameter is not sent. Required on an embeddings profile.
	//
	// It is here because it is a property of the MODEL, and because the callers
	// need it before they ever make a call: a vector column, a similarity index
	// and a stored corpus are all built to one width, and a model swapped for one
	// of a different width silently invalidates every row already written. An
	// application carrying its own copy of this number has two sources of truth
	// for one fact, and nothing to compare them against — which is precisely the
	// footgun this field removes.
	DefaultDimensions int `yaml:"default_dimensions"`
}

// Limits are the model's context and output bounds, in tokens.
type Limits struct {
	Context   int64 `yaml:"context"`
	MaxOutput int64 `yaml:"max_output"`
}

// Profile is everything known about one model at one endpoint, fully resolved.
//
// There are no optionals here beyond the explicit pointers: defaults are applied
// once at load, so no call site ever asks "what if this field is unset". A
// profile in memory is complete or the registry refused to load.
type Profile struct {
	// ID is what callers name. It is matched EXACTLY: no globbing, no prefix
	// matching. Prefix matching is how a new model silently inherits an older
	// family's quirks, and one published catalogue documents its own collision
	// between a model and that model's "-pro" variant.
	ID string `yaml:"id"`
	// Base names another profile to inherit from. See resolve for the rules.
	Base string `yaml:"base"`
	// Gateway marks a profile as reached through a proxy rather than directly.
	Gateway string `yaml:"gateway"`

	Endpoint Endpoint `yaml:"endpoint"`
	// WireModelID is what goes in the request body's "model" field, which is not
	// always the ID a caller uses: behind a gateway it is the proxy's alias.
	WireModelID string `yaml:"wire_model_id"`
	BaseURLEnv  string `yaml:"base_url_env"`
	APIKeyEnv   string `yaml:"api_key_env"`

	// MaxTokensParam is which output-cap parameter this endpoint honours.
	// Getting it wrong is not always an error: one endpoint accepts the wrong
	// name and IGNORES it, returning thirty times the requested tokens with no
	// signal at all.
	MaxTokensParam string `yaml:"max_tokens_param"`

	Reasoning   Reasoning `yaml:"reasoning"`
	Temperature Sampling  `yaml:"temperature"`
	TopP        Sampling  `yaml:"top_p"`
	Tools       Tools     `yaml:"tools"`
	Output      Output    `yaml:"output"`
	Streaming   Streaming `yaml:"streaming"`
	Limits      Limits    `yaml:"limits"`
	Vision      bool      `yaml:"vision"`
	Embedding   Embedding `yaml:"embedding"`

	// Cost is the published pay-as-you-go rate, or nil when no rate has been
	// verified for this model. Nil is not "free": pricing reports Unpriced and
	// warns, because a zero nobody can explain is worse than a missing number.
	// Absent for every gateway route by construction — see CostBlock.
	Cost *CostBlock `yaml:"cost"`

	// FinishReasonsExtra are vendor-specific finish_reason values beyond the
	// OpenAI vocabulary. Recorded for documentation: the parser treats the field
	// as an open set regardless, because an unknown value must never be fatal.
	FinishReasonsExtra []string `yaml:"finish_reasons_extra"`

	// Verified says how this profile's facts were established. "measured" means
	// a probe in eval_probe_test.go produced them; "source-derived" means they
	// were read from vendor source or documentation and never confirmed against
	// a live endpoint. A reader must be able to tell the two apart at a glance.
	Verified string `yaml:"verified"`
	// Notes carries per-profile caveats into godoc and error messages.
	Notes string `yaml:"notes"`
}

// Valid values for Verified.
const (
	VerifiedMeasured = "measured"
	VerifiedSource   = "source-derived"
)

// Wire names for the two output-cap parameters.
const (
	ParamMaxTokens           = "max_tokens"
	ParamMaxCompletionTokens = "max_completion_tokens"
)

// Keys a derived (based) profile is allowed to set. Everything else is a
// capability, and a capability is a property of the MODEL, not of the route
// taken to reach it.
//
// This is enforced as an allowlist rather than as a merge because the merge is
// where the ambiguity lives: in YAML an omitted bool and an explicit `false`
// decode identically, so "vision: false" on a derived profile is
// indistinguishable from saying nothing, and a merge has to guess which was
// meant. Refusing the field outright removes the guess. If a gateway genuinely
// exposes a narrower model, that is a different model and gets its own profile.
var derivedAllowedKeys = map[string]bool{
	"id": true, "base": true, "gateway": true,
	"wire_model_id": true, "base_url_env": true, "api_key_env": true,
	"endpoint": true, "max_tokens_param": true,
	"verified": true, "notes": true,
	// These two are allowed only for the specific sub-keys in
	// derivedAllowedNestedKeys. Admitting the whole mapping here and relying on
	// resolve to copy just the fields it knows about would silently DISCARD
	// every other sub-key: a gateway entry saying "streaming: {supported:
	// false}" would load clean and resolve to supported=true, which is exactly
	// the silent-config failure this package exists to prevent.
	"streaming": true,
	"tools":     true,
}

// Sub-keys a derived profile may set under the two mappings above. Both are
// TIGHTEN-ONLY: each can be switched on, never off.
//
//   - needs_include_usage: a proxy that strips the usage chunk from a client
//     which did not ask for it forces the parameter on, whatever the underlying
//     model needs.
//   - recover_inline_markup: defensive parsing, not a capability. Switching it
//     on can only cause more recovery to run, never claim support the model
//     lacks.
var derivedAllowedNestedKeys = map[string]map[string]bool{
	"streaming": {"needs_include_usage": true},
	"tools":     {"recover_inline_markup": true},
}

// resolve merges a base profile into this one.
//
// Identity and routing REPLACE: the wire model id, base URL and key variable are
// properties of the route, so a derived entry overrides them outright.
// Capabilities are INHERITED WHOLE and may not be restated — see
// derivedAllowedKeys for why a merge would have to guess.
//
// Single base, no chains: a base must itself be unbased. Inheritance here
// expresses "the same model, reached differently", not a type hierarchy, and a
// chain would make "which rule applied" unanswerable.
func (p Profile) resolve(base Profile) (Profile, error) {
	out := base

	out.ID = p.ID
	out.Base = p.Base
	if p.Gateway != "" {
		out.Gateway = p.Gateway
	}
	if p.WireModelID != "" {
		out.WireModelID = p.WireModelID
	}
	if p.BaseURLEnv != "" {
		out.BaseURLEnv = p.BaseURLEnv
	}
	if p.APIKeyEnv != "" {
		out.APIKeyEnv = p.APIKeyEnv
	}
	if p.Endpoint != "" {
		out.Endpoint = p.Endpoint
	}
	if p.MaxTokensParam != "" {
		out.MaxTokensParam = p.MaxTokensParam
	}
	// Provenance is deliberately NOT inherited. A derived profile describes a
	// different ROUTE to the model — a gateway alias, a forced parameter — and
	// none of that was exercised by whatever probe measured the base. Letting
	// "measured" carry across would make a profile claim evidence that does not
	// exist for it. Absent here means the loader defaults it to source-derived,
	// which is the honest answer until someone probes the route itself.
	out.Verified = p.Verified
	if p.Notes != "" {
		out.Notes = p.Notes
	}

	// Tighten-only, in the two places a route legitimately differs from its
	// model.
	if p.Streaming.NeedsIncludeUsage {
		out.Streaming.NeedsIncludeUsage = true
	}
	if p.Tools.RecoverInlineMarkup {
		out.Tools.RecoverInlineMarkup = true
	}

	// A gateway route inherits no price. The proxy reports its own per-call
	// spend, and it knows things this table cannot: which deployment actually
	// ran, and what it is charged at. Inheriting the base's list rate would
	// produce a confident figure for a call nobody priced — so the block is
	// dropped rather than carried, and a gateway entry stating one of its own is
	// refused by CostBlock.validate.
	if out.Gateway != "" {
		out.Cost = nil
	}
	return out, nil
}

// validate checks a fully-resolved profile for internal contradictions.
//
// These run at load, so a malformed profile is a failing test rather than a
// production 400. Every rule here encodes a combination that cannot be true of
// a real endpoint, which is what makes them worth checking rather than
// documenting.
func (p Profile) validate() error {
	if p.ID == "" {
		return fmt.Errorf("profile has no id")
	}
	bad := func(format string, args ...any) error {
		return fmt.Errorf("profile %q: "+format, append([]any{p.ID}, args...)...)
	}

	switch p.Endpoint {
	case EndpointChat, EndpointEmbeddings:
	default:
		return bad("endpoint %q is not %q or %q", p.Endpoint, EndpointChat, EndpointEmbeddings)
	}
	if p.WireModelID == "" {
		return bad("wire_model_id is empty; it is what goes on the wire and is not defaulted from id")
	}
	switch p.Verified {
	case VerifiedMeasured, VerifiedSource:
	default:
		return bad("verified must be %q or %q, got %q", VerifiedMeasured, VerifiedSource, p.Verified)
	}

	if p.Endpoint == EndpointEmbeddings {
		// An embeddings profile carrying chat capabilities is a copy-paste
		// error, and one that would otherwise surface as a confusing 400 from
		// the wrong route. Every sampling and generation knob is checked, not a
		// sample of them: a guard that covers most of the set reads as complete
		// while leaving a hole.
		switch {
		case p.Reasoning.Supported:
			return bad("an embeddings profile must not declare reasoning")
		case p.Tools.Supported:
			return bad("an embeddings profile must not declare tools")
		case p.Temperature.Supported:
			return bad("an embeddings profile must not declare temperature")
		case p.TopP.Supported:
			return bad("an embeddings profile must not declare top_p")
		case p.Vision:
			return bad("an embeddings profile must not declare vision")
		case p.Output.JSONObject || p.Output.JSONSchema || p.Output.StrictSchema:
			return bad("an embeddings profile must not declare structured output")
		}
		// Required, not optional. A caller sizes a vector column and an index to
		// this number before it ever makes a call, so leaving it absent would push
		// the figure back into every application that uses the model — which is
		// the duplication this field exists to end.
		if p.Embedding.DefaultDimensions <= 0 {
			return bad("an embeddings profile needs a positive default_dimensions; " +
				"a caller sizes its vector storage to it before the first call")
		}
		if p.Cost != nil {
			if err := p.Cost.validate(p); err != nil {
				return err
			}
		}
		// The limits check below applies to both endpoints, so fall through to
		// it rather than returning early.
		return p.validateLimits()
	}
	if p.Embedding.Dimensions || p.Embedding.DefaultDimensions != 0 {
		return bad("a chat profile must not declare embedding settings")
	}

	if p.MaxTokensParam != ParamMaxTokens && p.MaxTokensParam != ParamMaxCompletionTokens {
		return bad("max_tokens_param must be %q or %q, got %q",
			ParamMaxTokens, ParamMaxCompletionTokens, p.MaxTokensParam)
	}

	r := p.Reasoning
	// budget_param is the wire NAME of the budget knob, so it is meaningful for
	// exactly one control and nowhere else. Checked in both directions: missing
	// where it is required leaves the renderer with no key to send, and present
	// anywhere else is a profile describing a knob the model does not have.
	if r.Control == ControlBudget && r.Supported {
		if r.BudgetParam == "" {
			return bad("reasoning control is %q but budget_param is empty; there is no default, "+
				"because the vendors that take a budget disagree on the name", ControlBudget)
		}
	} else if r.BudgetParam != "" {
		return bad("budget_param is set but reasoning control is %q, which takes no budget", r.Control)
	}
	if !r.Supported {
		if r.Control != "" && r.Control != ControlNone {
			return bad("reasoning is unsupported but control is %q", r.Control)
		}
		if len(r.EffortValues) > 0 {
			return bad("reasoning is unsupported but effort_values is set")
		}
		if r.EnabledByDefault || r.CanBeDisabled {
			return bad("reasoning is unsupported but enabled_by_default/can_be_disabled is set")
		}
	} else {
		switch r.Control {
		case ControlEffort:
			if len(r.EffortValues) == 0 {
				return bad("reasoning control is %q but effort_values is empty; "+
					"the accepted set must be stated, never inferred", ControlEffort)
			}
			if r.DefaultEffort != "" && !r.Accepts(r.DefaultEffort) {
				return bad("default_effort %q is not in effort_values %v", r.DefaultEffort, r.EffortValues)
			}
		case ControlToggleObject, ControlBudget:
		default:
			return bad("reasoning is supported but control is %q", r.Control)
		}
		// The inference this schema exists to prevent, checked in both
		// directions: a model that cannot be disabled must not advertise a
		// disabling value, and one that can must offer a way to do it.
		if !r.CanBeDisabled && r.Accepts("none") {
			return bad(`can_be_disabled is false but effort_values contains "none"`)
		}
		if r.CanBeDisabled && r.Control == ControlEffort && !r.Accepts("none") {
			return bad(`can_be_disabled is true but effort_values has no "none" and control is %q`, ControlEffort)
		}
		if !r.EnabledByDefault && !r.CanBeDisabled {
			return bad("reasoning is supported but is neither on by default nor switchable; " +
				"that combination describes a model that never reasons")
		}
	}

	if p.Tools.Supported {
		if p.Tools.Format != FormatNative && p.Tools.Format != FormatXML {
			return bad("tools.format must be %q or %q, got %q", FormatNative, FormatXML, p.Tools.Format)
		}
		if p.Tools.Format == FormatXML && p.Tools.SupportsForcedChoice {
			return bad("tools.format is %q, which has no tool_calls field to force, "+
				"but supports_forced_choice is true", FormatXML)
		}
		if len(p.Tools.ToolChoiceValues) == 0 {
			return bad("tools are supported but tool_choice_values is empty; " +
				"several endpoints accept only \"auto\" and silently drop the rest")
		}
	} else if len(p.Tools.ToolChoiceValues) > 0 || p.Tools.SupportsForcedChoice {
		return bad("tools are unsupported but tool settings are declared")
	}

	if p.Output.StrictSchema && !p.Output.JSONSchema {
		return bad("strict_schema is set but json_schema is not supported")
	}
	if p.Streaming.NeedsIncludeUsage && !p.Streaming.AcceptsStreamOptions {
		return bad("needs_include_usage is true but accepts_stream_options is false; " +
			"the parameter would have to be sent to an endpoint that rejects it")
	}
	if p.Cost != nil {
		if err := p.Cost.validate(p); err != nil {
			return err
		}
	}
	return p.validateLimits()
}

// validateLimits is shared by both endpoint kinds.
func (p Profile) validateLimits() error {
	if p.Limits.MaxOutput > 0 && p.Limits.Context > 0 && p.Limits.MaxOutput > p.Limits.Context {
		return fmt.Errorf("profile %q: max_output (%d) exceeds context (%d)",
			p.ID, p.Limits.MaxOutput, p.Limits.Context)
	}
	return nil
}

// String renders a profile for an error message: enough to identify which
// deployment is meant, without dumping the whole struct.
func (p Profile) String() string {
	var b strings.Builder
	b.WriteString(p.ID)
	if p.WireModelID != p.ID {
		fmt.Fprintf(&b, " (wire: %s)", p.WireModelID)
	}
	if p.Gateway != "" {
		fmt.Fprintf(&b, " via %s", p.Gateway)
	}
	return b.String()
}

// sortedIDs is a stable list for error messages and for Models().
func sortedIDs(m map[string]*Profile) []string {
	out := make([]string, 0, len(m))
	for id := range m {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// clone returns a deep-enough copy that a caller mutating the result cannot
// affect the registry.
//
// Registry is cached for the life of the process and its profiles are shared by
// every caller, so handing out the stored pointer makes one caller's
// experiment everyone else's configuration. Slices and pointer fields are
// copied too: a shallow struct copy still shares their backing storage, which
// would leave exactly the same hole one level down.
func (p *Profile) clone() *Profile {
	out := *p
	out.Reasoning.EffortValues = append([]string(nil), p.Reasoning.EffortValues...)
	out.Tools.ToolChoiceValues = append([]string(nil), p.Tools.ToolChoiceValues...)
	out.FinishReasonsExtra = append([]string(nil), p.FinishReasonsExtra...)
	out.Temperature.ForcedValue = copyFloat(p.Temperature.ForcedValue)
	out.Temperature.RecommendedValue = copyFloat(p.Temperature.RecommendedValue)
	out.TopP.ForcedValue = copyFloat(p.TopP.ForcedValue)
	out.TopP.RecommendedValue = copyFloat(p.TopP.RecommendedValue)
	out.Cost = p.Cost.clone()
	return &out
}

func copyFloat(p *float64) *float64 {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
