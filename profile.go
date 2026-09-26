package llmwire

import (
	"fmt"
	"regexp"
	"slices"
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

// The endpoints a profile can describe.
const (
	EndpointChat       Endpoint = "chat"
	EndpointEmbeddings Endpoint = "embeddings"
)

// ReasoningControl is HOW a model's reasoning depth is expressed on the wire.
// Models that reason differ in the knob, not just in its values: one takes a
// nested object, another an effort string, a third a token budget.
type ReasoningControl string

const (
	// ControlNone is a model that exposes no reasoning knob.
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
	//
	// Listed shallowest first, in effortLadder order, and checked at load:
	// ReasoningMinimal reads the first entry as the model's floor.
	//
	// Required for ControlEffort. Optional for ControlToggleObject, where a
	// non-empty set says the model ALSO takes reasoning_effort beside its
	// on/off switch: ReasoningEffort(level) then renders the field instead of
	// being refused, and ReasoningOff() still renders the toggle. Empty on a
	// toggle model means the switch is the only knob.
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
	// LeaksCloseTag: the model writes a stray "</think>" into content, after a
	// draft of its answer, even with thinking disabled. Chat then cuts content
	// at the first such tag, on requests with thinking off and no JSON format,
	// and moves the part before it to Reasoning. Observed,
	// not documented by any vendor; see profiles.yaml for the case. Applies to
	// Chat only: a stream has already sent the draft by the time the tag
	// arrives.
	LeaksCloseTag bool `yaml:"leaks_close_tag"`
	// Balanced is the level ReasoningBalanced resolves to: fast but not
	// shallow. Must be one of EffortValues, and never "none". Empty means the
	// intent sends nothing and the model runs at its own default.
	Balanced string `yaml:"balanced"`
	// Overhead is the reasoning tokens to budget beside a MaxAnswerTokens
	// answer, per resolved level: an effort level, "off", or "default" for a
	// request that sends no knob (DefaultEffort's entry wins there when both
	// exist). A level with no entry on a thinking request takes
	// DefaultReasoningOverhead; thinking off takes 0. A budget request is its
	// own overhead. Absent unless measured, like every other bit.
	Overhead map[string]int `yaml:"overhead"`
	// MinBudget is the budget ReasoningMinimal sends on a budget_tokens model
	// that cannot be disabled. Forbidden on every other control. Absent there,
	// that intent is refused: a guessed floor is either rejected by the
	// endpoint or not a floor.
	MinBudget int `yaml:"min_budget"`
}

// DefaultReasoningOverhead is the reasoning allowance MaxAnswerTokens adds for a
// thinking request whose level has no reasoning.overhead entry. A budget, not a
// measurement: the one measured point is that 1024 was NOT enough at
// glm-5.3-flash's deepest level, which spent a 1024-token cap on reasoning and
// returned no content. A profile with a measured figure overrides it per level.
const DefaultReasoningOverhead = 1024

// effortLadder orders every effort level a profile may list, shallowest first.
// A level is placed here once, so "shallowest" is a property of the word and
// effort_values can be checked against it rather than trusted. A vendor level
// not listed fails the load until it is placed.
var effortLadder = []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"}

// Accepts reports whether effort is in the model's accepted set.
func (r Reasoning) Accepts(effort string) bool { return slices.Contains(r.EffortValues, effort) }

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
	// BuffersToolArgs: the model emits a tool call's name, then goes silent
	// while it serializes the whole argument server-side, then flushes it in
	// one burst. A caller offering tools that carry large arguments widens
	// ChatRequest.ToolCallIdleTimeout for those turns.
	BuffersToolArgs bool `yaml:"buffers_tool_args"`
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
	// DisplayName is the human label ("GLM 5.3 Flash"), for a model picker or a
	// log a person reads. Defaults to ID at load, so a document written before
	// the field still loads; every shipped profile sets it. A route keeps its
	// model's label unless it names its own.
	DisplayName string `yaml:"display_name"`
	// Base names another profile to inherit from. See resolve for the rules.
	Base string `yaml:"base"`
	// Gateway marks a profile as reached through a proxy rather than directly.
	Gateway string `yaml:"gateway"`

	Endpoint Endpoint `yaml:"endpoint"`
	// WireModelID is what goes in the request body's "model" field, which is not
	// always the ID a caller uses: behind a gateway it is the proxy's alias.
	WireModelID string `yaml:"wire_model_id"`
	// Provider names the host and account this model is reached through:
	// zai, mimo, openai, litellm. It decides the environment variables
	// FromEnv reads, LLMWIRE_<PROVIDER>_BASE_URL and _API_KEY, so two profiles
	// on one host cannot name different variables. Lowercase letters and
	// digits only; it is uppercased into the variable name.
	Provider string `yaml:"provider"`
	// NoAPIKey marks a host that authenticates nothing: FromEnv sends no
	// Authorization header and does not look for the key variable.
	NoAPIKey bool `yaml:"no_api_key"`
	// BaseURL is the provider's endpoint root, resolved at load from the
	// providers: map. Not a profile key: the host belongs to the provider, and
	// a profile restating it would be the per-model copy this field removes.
	// Empty when the provider ships no host; FromEnv then requires the
	// LLMWIRE_<PROVIDER>_BASE_URL variable, which it otherwise refuses.
	BaseURL string `yaml:"-"`
	// EmulateOpenCode follows the provider the same way BaseURL does.
	EmulateOpenCode bool `yaml:"-"`

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
	"id": true, "display_name": true, "base": true, "gateway": true,
	"wire_model_id": true, "provider": true, "no_api_key": true,
	"max_tokens_param": true,
	"verified":         true, "notes": true,
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
func (p Profile) resolve(base Profile) Profile {
	out := base

	out.ID = p.ID
	out.Base = p.Base
	if p.DisplayName != "" {
		out.DisplayName = p.DisplayName
	}
	if p.Gateway != "" {
		out.Gateway = p.Gateway
	}
	if p.WireModelID != "" {
		out.WireModelID = p.WireModelID
	}
	if p.Provider != "" {
		out.Provider = p.Provider
		// The flag belongs to the host, not the model: a new provider is a
		// new host, and whether it authenticates is the derived profile's to
		// say. Without this a keyed gateway in front of a keyless base would
		// inherit "no key" and be answered with a 401.
		out.NoAPIKey = p.NoAPIKey
	} else if p.NoAPIKey {
		out.NoAPIKey = true
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
	// dropped rather than carried. A derived entry restating one is refused by
	// checkDerivedKeys, an unbased gateway entry by CostBlock.validate.
	if out.Gateway != "" {
		out.Cost = nil
	}
	return out
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
	if p.Provider != "" && !providerName.MatchString(p.Provider) {
		return bad("provider %q must be lowercase letters and digits, it becomes the LLMWIRE_<PROVIDER>_BASE_URL variable name", p.Provider)
	}
	switch p.Verified {
	case VerifiedMeasured, VerifiedSource:
	default:
		return bad("verified must be %q or %q, got %q", VerifiedMeasured, VerifiedSource, p.Verified)
	}

	var err error
	if p.Endpoint == EndpointEmbeddings {
		err = p.validateEmbeddings(bad)
	} else {
		err = p.validateChat(bad)
	}
	if err != nil {
		return err
	}
	if p.Cost != nil {
		if err := p.Cost.validate(p); err != nil {
			return err
		}
	}
	return p.validateLimits()
}

// validateEmbeddings refuses chat capabilities on an embeddings profile.
//
// Such a profile is a copy-paste error, and one that would otherwise surface
// as a confusing 400 from the wrong route. Every sampling and generation knob
// is checked, not a sample of them, and not only the `supported` bits: a
// guard that covers most of the set reads as complete while leaving a hole.
func (p Profile) validateEmbeddings(bad func(string, ...any) error) error {
	r := p.Reasoning
	reasoningDeclared := r.Supported || r.EnabledByDefault || r.CanBeDisabled || r.Control != "" ||
		len(r.EffortValues) > 0 || r.DefaultEffort != "" || r.StreamField != "" || r.BudgetParam != "" || r.LeaksCloseTag ||
		r.Balanced != "" || len(r.Overhead) > 0 || r.MinBudget != 0
	t := p.Tools
	toolsDeclared := t.Supported || t.Format != "" || len(t.ToolChoiceValues) > 0 ||
		t.SupportsForcedChoice || t.SupportsParallel || t.RecoverInlineMarkup
	switch {
	case reasoningDeclared:
		return bad("an embeddings profile must not declare reasoning")
	case toolsDeclared:
		return bad("an embeddings profile must not declare tools")
	case p.Temperature.declared():
		return bad("an embeddings profile must not declare temperature")
	case p.TopP.declared():
		return bad("an embeddings profile must not declare top_p")
	case p.Vision:
		return bad("an embeddings profile must not declare vision")
	case p.Output.JSONObject || p.Output.JSONSchema || p.Output.StrictSchema:
		return bad("an embeddings profile must not declare structured output")
	case p.Streaming.Supported || p.Streaming.NeedsIncludeUsage || p.Streaming.AcceptsStreamOptions || p.Streaming.BuffersToolArgs:
		return bad("an embeddings profile must not declare streaming")
	case p.MaxTokensParam != "":
		return bad("an embeddings profile must not declare max_tokens_param; the route takes no output cap")
	}
	// Required, not optional. A caller sizes a vector column and an index to
	// this number before it ever makes a call, so leaving it absent would push
	// the figure back into every application that uses the model — which is
	// the duplication this field exists to end.
	if p.Embedding.DefaultDimensions <= 0 {
		return bad("an embeddings profile needs a positive default_dimensions; " +
			"a caller sizes its vector storage to it before the first call")
	}
	return nil
}

// declared reports whether any sampling field is set, `supported` included.
func (s Sampling) declared() bool {
	return s.Supported || s.ForcedValue != nil || s.RecommendedValue != nil || s.InertWhileReasoning
}

// validateSampling checks one sampling parameter's own consistency.
func validateSampling(bad func(string, ...any) error, name string, s Sampling) error {
	if !s.Supported && (s.ForcedValue != nil || s.RecommendedValue != nil || s.InertWhileReasoning) {
		// The value would be read by nothing: checkSampling drops the
		// parameter before it looks at either, so the profile would describe a
		// knob the request path never sends.
		return bad("%s is unsupported but forced_value, recommended_value or inert_while_reasoning is set", name)
	}
	if s.ForcedValue != nil && s.RecommendedValue != nil && *s.ForcedValue != *s.RecommendedValue {
		// The recommended value is what goes out when the caller says
		// nothing, and a forced value is the only one the endpoint takes:
		// disagreeing, the silent default would be a 400.
		return bad("%s recommended_value %g differs from forced_value %g, which is the only value the endpoint accepts",
			name, *s.RecommendedValue, *s.ForcedValue)
	}
	return nil
}

// validateChat checks the capability block of a chat profile.
func (p Profile) validateChat(bad func(string, ...any) error) error {
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
	if dup := firstDuplicate(r.EffortValues); dup != "" {
		return bad("effort_values lists %q twice", dup)
	}
	if err := checkEffortOrder(bad, r.EffortValues); err != nil {
		return err
	}
	// Meaningful for one control only, like budget_param: the floor
	// ReasoningMinimal sends to a budget model it cannot switch off. Not
	// required, so a document from before the field loads; without it that
	// intent is refused on such a model rather than sent with a guessed floor.
	switch {
	case r.MinBudget < 0:
		return bad("min_budget %d is negative", r.MinBudget)
	case r.MinBudget != 0 && r.Control != ControlBudget:
		return bad("min_budget is set but reasoning control is %q, which takes no budget", r.Control)
	}
	if !r.Supported {
		if r.Balanced != "" || len(r.Overhead) > 0 {
			return bad("reasoning is unsupported but balanced or overhead is set")
		}
		if r.Control != "" && r.Control != ControlNone {
			return bad("reasoning is unsupported but control is %q", r.Control)
		}
		if len(r.EffortValues) > 0 {
			return bad("reasoning is unsupported but effort_values is set")
		}
		if r.EnabledByDefault || r.CanBeDisabled {
			return bad("reasoning is unsupported but enabled_by_default/can_be_disabled is set")
		}
		if r.DefaultEffort != "" || r.StreamField != "" || r.LeaksCloseTag {
			return bad("reasoning is unsupported but default_effort, stream_field or leaks_close_tag is set")
		}
	} else {
		switch r.Control {
		case ControlEffort:
			if len(r.EffortValues) == 0 {
				return bad("reasoning control is %q but effort_values is empty; "+
					"the accepted set must be stated, never inferred", ControlEffort)
			}
		case ControlToggleObject:
			// Levels beside a toggle only make sense on a model that is
			// thinking to begin with: there is no on-switch constructor, so a
			// level sent to a model that is off would be inert.
			if len(r.EffortValues) > 0 && !r.EnabledByDefault {
				return bad("effort_values is set on a %s model that is not enabled_by_default; a level cannot switch thinking on", ControlToggleObject)
			}
		case ControlBudget:
			if len(r.EffortValues) > 0 {
				return bad("reasoning control is %q but effort_values is set; a budget model takes no levels", ControlBudget)
			}
			if r.DefaultEffort != "" {
				return bad("default_effort is set but reasoning control is %q, which takes no levels", ControlBudget)
			}
		default:
			return bad("reasoning is supported but control is %q", r.Control)
		}
		// A default that is not in the accepted set describes a model
		// contradicting itself; checked once here for both level-taking
		// controls rather than per case.
		if r.DefaultEffort != "" && !r.Accepts(r.DefaultEffort) {
			return bad("default_effort %q is not in effort_values %v", r.DefaultEffort, r.EffortValues)
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
		if r.Balanced != "" && (r.Balanced == "none" || !r.Accepts(r.Balanced)) {
			return bad("balanced %q must be one of effort_values %v other than none; "+
				"none is off, not balanced", r.Balanced, r.EffortValues)
		}
		if err := r.checkOverhead(bad); err != nil {
			return err
		}
	}

	if err := validateSampling(bad, "temperature", p.Temperature); err != nil {
		return err
	}
	if err := validateSampling(bad, "top_p", p.TopP); err != nil {
		return err
	}

	t := p.Tools
	if t.Supported {
		if t.Format != FormatNative && t.Format != FormatXML {
			return bad("tools.format must be %q or %q, got %q", FormatNative, FormatXML, t.Format)
		}
		if t.Format == FormatXML && t.SupportsForcedChoice {
			return bad("tools.format is %q, which has no tool_calls field to force, "+
				"but supports_forced_choice is true", FormatXML)
		}
		if len(t.ToolChoiceValues) == 0 {
			return bad("tools are supported but tool_choice_values is empty; " +
				"several endpoints accept only \"auto\" and silently drop the rest")
		}
		// The list holds the STRING modes only. A forced function is an
		// object on the wire and has its own bit, so "function" in the list
		// would be a spelling the request path never matches, and a typo
		// (a misspelt "required") would silently make that mode a relaxation.
		for _, v := range t.ToolChoiceValues {
			if !slices.Contains(toolChoiceStringModes, v) {
				return bad("tool_choice_values entry %q is not one of %v; a forced function is supports_forced_choice",
					v, toolChoiceStringModes)
			}
		}
		if dup := firstDuplicate(t.ToolChoiceValues); dup != "" {
			return bad("tool_choice_values lists %q twice", dup)
		}
	} else if len(t.ToolChoiceValues) > 0 || t.SupportsForcedChoice || t.SupportsParallel ||
		t.Format != "" || t.RecoverInlineMarkup {
		return bad("tools are unsupported but tool settings are declared")
	}

	if p.Output.StrictSchema && !p.Output.JSONSchema {
		return bad("strict_schema is set but json_schema is not supported")
	}
	if p.Streaming.NeedsIncludeUsage && !p.Streaming.AcceptsStreamOptions {
		return bad("needs_include_usage is true but accepts_stream_options is false; " +
			"the parameter would have to be sent to an endpoint that rejects it")
	}
	if p.Streaming.BuffersToolArgs && (!p.Streaming.Supported || !p.Tools.Supported) {
		return bad("buffers_tool_args is set but the model does not stream tool calls; " +
			"it describes a streamed tool-call argument")
	}
	return nil
}

// checkEffortOrder requires every level on the ladder and the list shallowest
// first. ReasoningMinimal takes the first entry as the floor, so a list in
// vendor-doc order would make "minimal" the deepest setting without a word.
func checkEffortOrder(bad func(string, ...any) error, values []string) error {
	prev := -1
	for _, v := range values {
		rank := slices.Index(effortLadder, v)
		if rank < 0 {
			return bad("effort_values entry %q is not on the depth ladder %v; place it in effortLadder first", v, effortLadder)
		}
		if rank < prev {
			return bad("effort_values %v must be listed shallowest first (ladder %v)", values, effortLadder)
		}
		prev = rank
	}
	return nil
}

// checkOverhead refuses a figure nothing would read: a level the model does not
// take, "off" on a model that cannot be switched off, "default" on one that does
// not think unasked.
func (r Reasoning) checkOverhead(bad func(string, ...any) error) error {
	for _, level := range sortedKeys(r.Overhead) {
		switch n := r.Overhead[level]; {
		case n < 0:
			return bad("overhead %q is %d; a reasoning allowance cannot be negative", level, n)
		case level == "off" && !r.CanBeDisabled,
			level == "default" && !r.EnabledByDefault,
			level != "off" && level != "default" && !r.Accepts(level):
			return bad("overhead names %q, which no request to this model resolves to "+
				"(effort_values %v, off only if can_be_disabled, default only if enabled_by_default)",
				level, r.EffortValues)
		}
	}
	return nil
}

// toolChoiceStringModes are the tool_choice values that go on the wire as a
// bare string. The forced-function form is an object and is described by
// Tools.SupportsForcedChoice instead.
var toolChoiceStringModes = []string{string(ToolChoiceAuto), string(ToolChoiceNone), string(ToolChoiceRequired)}

// firstDuplicate returns the first value that appears twice, or "".
func firstDuplicate(values []string) string {
	seen := make(map[string]bool, len(values))
	for _, v := range values {
		if seen[v] {
			return v
		}
		seen[v] = true
	}
	return ""
}

// validateLimits is shared by both endpoint kinds.
func (p Profile) validateLimits() error {
	if p.Limits.Context < 0 || p.Limits.MaxOutput < 0 {
		return fmt.Errorf("profile %q: limits must not be negative (context %d, max_output %d)",
			p.ID, p.Limits.Context, p.Limits.MaxOutput)
	}
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
	if p.Reasoning.Overhead != nil {
		out.Reasoning.Overhead = make(map[string]int, len(p.Reasoning.Overhead))
		for k, v := range p.Reasoning.Overhead {
			out.Reasoning.Overhead[k] = v
		}
	}
	out.Tools.ToolChoiceValues = append([]string(nil), p.Tools.ToolChoiceValues...)
	out.FinishReasonsExtra = append([]string(nil), p.FinishReasonsExtra...)
	out.Temperature.ForcedValue = copyPtr(p.Temperature.ForcedValue)
	out.Temperature.RecommendedValue = copyPtr(p.Temperature.RecommendedValue)
	out.TopP.ForcedValue = copyPtr(p.TopP.ForcedValue)
	out.TopP.RecommendedValue = copyPtr(p.TopP.RecommendedValue)
	out.Cost = p.Cost.clone()
	return &out
}

// copyPtr returns a pointer to a copy of *p, or nil for nil: the one-line
// deep copy every clone in this package needs for its pointer fields.
func copyPtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

var providerName = regexp.MustCompile(`^[a-z][a-z0-9]*$`)

// BaseURLEnv is LLMWIRE_<PROVIDER>_BASE_URL: required when BaseURL is empty,
// refused when it is not. Empty when the profile names no provider.
func (p Profile) BaseURLEnv() string { return p.providerVar("BASE_URL") }

// APIKeyEnv is the variable FromEnv reads the bearer token from:
// LLMWIRE_<PROVIDER>_API_KEY. Empty when the profile names no provider or
// takes no key.
func (p Profile) APIKeyEnv() string {
	if p.NoAPIKey {
		return ""
	}
	return p.providerVar("API_KEY")
}

func (p Profile) providerVar(field string) string {
	if p.Provider == "" {
		return ""
	}
	return "LLMWIRE_" + strings.ToUpper(p.Provider) + "_" + field
}
