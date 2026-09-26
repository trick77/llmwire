package llmwire

import "fmt"

// Intent resolution and the answer budget.
//
// Both run in plan, before any check reads the request, so everything after
// them sees a concrete variant and a concrete cap: an intent is validated,
// rendered, logged and priced exactly as what it became, and there is no
// second place a resolution could be re-derived and disagree.

// resolveReasoning turns an intent into the concrete request this profile
// takes. Concrete requests and nil pass through untouched. nil out means
// "send nothing": the model runs at its own default.
func resolveReasoning(want ReasoningRequest, r Reasoning) ReasoningRequest {
	switch want.(type) {
	case reasoningMinimal:
		switch {
		case !r.Supported:
			// Already as shallow as it gets; there is no knob to send.
			return nil
		case r.CanBeDisabled:
			return reasoningOff{}
		case len(r.EffortValues) > 0:
			// Shallowest first, checked at load (checkEffortOrder).
			return reasoningEffort{level: r.EffortValues[0]}
		case r.Control == ControlBudget && r.MinBudget > 0:
			return reasoningBudget{tokens: r.MinBudget}
		case r.Control == ControlBudget:
			// No floor declared: left as the intent, which checkReasoning
			// refuses by name.
			return want
		default:
			// A toggle that cannot be switched off and takes no level: the
			// default is the only setting there is.
			return nil
		}
	case reasoningBalanced:
		if !r.Supported || r.Balanced == "" {
			return nil
		}
		return reasoningEffort{level: r.Balanced}
	default:
		return want
	}
}

// reasoningOverhead is the reasoning allowance a MaxAnswerTokens request adds,
// for the request as resolved and coerced.
func reasoningOverhead(want ReasoningRequest, r Reasoning) int {
	if !r.Supported {
		return 0
	}
	switch w := want.(type) {
	case nil:
		if !r.EnabledByDefault {
			return 0
		}
		if n, ok := r.Overhead[r.DefaultEffort]; ok && r.DefaultEffort != "" {
			return n
		}
		return overheadOr(r, "default", DefaultReasoningOverhead)
	case reasoningOff:
		return overheadOr(r, "off", 0)
	case reasoningEffort:
		if w.level == "none" {
			return overheadOr(r, "off", 0)
		}
		return overheadOr(r, w.level, DefaultReasoningOverhead)
	case reasoningBudget:
		// The budget IS the allowance; a zero budget is the off switch.
		return w.tokens
	default:
		// Unreachable: intents are resolved before this runs. Budgeted as a
		// thinking request, so a fifth variant over-allows rather than starves.
		return DefaultReasoningOverhead
	}
}

func overheadOr(r Reasoning, level string, def int) int {
	if n, ok := r.Overhead[level]; ok {
		return n
	}
	return def
}

// checkAnswerBudget turns MaxAnswerTokens into the wire cap. Runs after
// checkReasoning, so the overhead is that of the knob actually going out,
// BestEffort drops included.
func (v *validation) checkAnswerBudget(req ChatRequest) {
	if req.MaxAnswerTokens == nil {
		return
	}
	if req.MaxTokens != nil {
		// Structural: two caps with different meanings and one wire field.
		v.reject("max_answer_tokens", "set MaxTokens or MaxAnswerTokens, not both")
		return
	}
	answer := *req.MaxAnswerTokens
	if answer <= 0 {
		v.reject("max_answer_tokens", fmt.Sprintf("an answer budget of %d is not positive; leave MaxAnswerTokens nil for no cap", answer))
		return
	}
	max := v.profile.Limits.MaxOutput
	// A budget is a hard allotment, not an estimate: a budget endpoint refuses
	// a completion cap at or below it, so the clamp below must never reach it.
	if b, ok := v.out.Reasoning.(reasoningBudget); ok && b.tokens > 0 && max > 0 {
		switch {
		case int64(b.tokens) >= max:
			if v.refuse("reasoning", fmt.Sprintf("a reasoning budget of %d leaves no room for an answer "+
				"under this model's output limit of %d", b.tokens, max), nil) {
				return
			}
			v.dropReasoning()
		case int64(answer) <= max && int64(answer)+int64(b.tokens) > max:
			v.warn(WarnCompatibility, "max_answer_tokens",
				fmt.Sprintf("budget %d plus answer %d exceed the output limit of %d; the answer gets %d",
					b.tokens, answer, max, max-int64(b.tokens)))
		}
	}
	wire := int64(answer) + int64(reasoningOverhead(v.out.Reasoning, v.profile.Reasoning))
	if max > 0 {
		if int64(answer) > max {
			v.warn(WarnCompatibility, "max_answer_tokens",
				fmt.Sprintf("%d exceeds this model's output limit of %d; the cap sent is the limit", answer, max))
		}
		// Otherwise clamped silently: a level's overhead is an allowance, not
		// an allotment, and the endpoint would clamp anyway. A budget always
		// stays below the clamp (checked above).
		wire = min(wire, max)
	}
	n := int(wire)
	v.out.MaxTokens = &n
}
