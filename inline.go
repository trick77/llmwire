package llmwire

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Inline tool-call recovery.
//
// Some deployments emit a tool call as markup inside the content (or reasoning)
// channel instead of populating the native tool_calls field. Two syntaxes are
// known, both observed on MiMo in production against a long, tool-saturated
// history that a probe does not reproduce (see profiles.yaml, mimo-v2.5-pro):
//
//	<tool_call>
//	<function=tavily__tavily_search>
//	<parameter=query>colossus forbin project</parameter>
//	</function>
//	</tool_call>
//
// and an undocumented variant no prompt teaches:
//
//	<tool_invocation name="generate_image" arguments={"prompt": "…", "height": 1024} />
//
// A profile opts in with tools.recover_inline_markup (or tools.format: xml). With
// it on, the parser withholds the markup from streamed content and reasoning
// deltas, recovers the calls once the answer is complete and cuts the markup from
// the accumulated text, so no caller ever sees it. Recovery runs whether or not
// the request offered tools: the case that leaked was a tool-free call answered
// with a tool call.

const (
	inlineToolCallMarker   = "<tool_call>"
	inlineInvocationMarker = "<tool_invocation"
	// inlineToolCallIDPrefix names recovered calls; the endpoint minted no id.
	inlineToolCallIDPrefix = "inline_call_"
)

// inlineToolCallMarkers are the opening markers of every inline syntax. The gate
// withholds streamed text from the earliest of these.
var inlineToolCallMarkers = []string{inlineToolCallMarker, inlineInvocationMarker}

var (
	inlineToolCallBlock = regexp.MustCompile(`(?s)<tool_call>(.*?)</tool_call>`)
	inlineFunctionName  = regexp.MustCompile(`<function=([^>]+)>`)
	inlineParameter     = regexp.MustCompile(`(?s)<parameter=([^>]+)>(.*?)</parameter>`)

	// For the <tool_invocation …/> variant. The arguments value is raw JSON with
	// nested braces and quotes, so it is located by balanced-brace scanning (see
	// scanJSONObject) rather than a regex; only the name attribute is matched
	// here. The \b anchors the match to the `name` attribute so a name-suffixed
	// attribute ahead of it (display_name="x" …) is not mistaken for the tool name.
	inlineInvocationName = regexp.MustCompile(`\bname\s*=\s*"([^"]*)"`)
	inlineInvocationArgs = regexp.MustCompile(`arguments\s*=\s*`)
)

// recoversInline reports whether this profile's answers are parsed for inline
// tool-call markup.
func (t Tools) recoversInline() bool {
	return t.RecoverInlineMarkup || t.Format == FormatXML
}

// inlineToolCallID is the synthetic id of the index-th (0-based) recovered call.
// The stream surfaces the first call's name early under this same id, so the
// full call parsed at the end updates that entry instead of duplicating it.
func inlineToolCallID(index int) string {
	return fmt.Sprintf("%s%d", inlineToolCallIDPrefix, index+1)
}

// firstInlineToolName extracts the name of the first inline call from a
// possibly still-streaming buffer, as soon as its name has fully arrived. MiMo
// emits the name right after the marker but flushes the argument tens of seconds
// later, so this is what lets a client name the running tool during that gap.
// Returns "" until the name is parseable.
func firstInlineToolName(content string) string {
	if m := inlineFunctionName.FindStringSubmatch(content); m != nil {
		return strings.TrimSpace(m[1])
	}
	if idx := strings.Index(content, inlineInvocationMarker); idx >= 0 {
		if m := inlineInvocationName.FindStringSubmatch(content[idx:]); m != nil {
			return strings.TrimSpace(m[1])
		}
	}
	return ""
}

// containsInlineToolMarker reports whether s carries any inline marker.
func containsInlineToolMarker(s string) bool {
	for _, m := range inlineToolCallMarkers {
		if strings.Contains(s, m) {
			return true
		}
	}
	return false
}

// cutAtFirstInlineMarker truncates s at the earliest inline marker, mirroring what
// the gate withholds live. It is the safety net for a block that failed to parse
// (truncated or malformed): the parser leaves that markup in place, and this keeps
// it out of the returned text. A no-op when no marker is present.
func cutAtFirstInlineMarker(s string) string {
	cut := -1
	for _, m := range inlineToolCallMarkers {
		if idx := strings.Index(s, m); idx >= 0 && (cut < 0 || idx < cut) {
			cut = idx
		}
	}
	if cut < 0 {
		return s
	}
	return strings.TrimSpace(s[:cut])
}

// parseInlineToolCalls extracts every block of either syntax and returns the
// content with the blocks removed. When no call is present it returns
// (nil, content) unchanged.
func parseInlineToolCalls(content string) ([]ToolCall, string) {
	var calls []ToolCall
	cleaned := content

	// Syntax 1: <tool_call><function=NAME><parameter=k>v</parameter>…</tool_call>.
	if blocks := inlineToolCallBlock.FindAllStringSubmatchIndex(content, -1); len(blocks) > 0 {
		for _, block := range blocks {
			inner := content[block[2]:block[3]]
			m := inlineFunctionName.FindStringSubmatch(inner)
			if m == nil {
				continue
			}
			name := strings.TrimSpace(m[1])
			if name == "" {
				continue
			}
			calls = append(calls, ToolCall{Type: "function", Name: name, Arguments: inlineArguments(inner)})
		}
		if len(calls) > 0 {
			cleaned = inlineToolCallBlock.ReplaceAllString(cleaned, "")
		}
	}

	// Syntax 2: <tool_invocation name="NAME" arguments={…json…} />.
	if invCalls, invCleaned := parseInvocationTags(cleaned); len(invCalls) > 0 {
		calls = append(calls, invCalls...)
		cleaned = invCleaned
	}

	if len(calls) == 0 {
		return nil, content
	}
	// Ids by overall order, so the first call keeps the id the stream surfaced
	// early regardless of which syntax produced it.
	for i := range calls {
		calls[i].ID = inlineToolCallID(i)
	}
	return calls, strings.TrimSpace(cleaned)
}

// parseInvocationTags extracts every <tool_invocation name="…" arguments={…} />
// tag. Tags that do not parse cleanly (no name, truncated, unbalanced JSON) are
// left in place so the unparsed-markup warning can name them.
func parseInvocationTags(content string) ([]ToolCall, string) {
	if !strings.Contains(content, inlineInvocationMarker) {
		return nil, content
	}
	var calls []ToolCall
	var b strings.Builder
	i := 0
	for {
		rel := strings.Index(content[i:], inlineInvocationMarker)
		if rel < 0 {
			b.WriteString(content[i:])
			break
		}
		start := i + rel
		call, end, ok := parseInvocationAt(content, start)
		if !ok {
			// Keep the marker text and step past it so the scan cannot loop.
			b.WriteString(content[i : start+len(inlineInvocationMarker)])
			i = start + len(inlineInvocationMarker)
			continue
		}
		b.WriteString(content[i:start])
		calls = append(calls, call)
		i = end
	}
	return calls, b.String()
}

// parseInvocationAt parses one <tool_invocation …/> tag beginning at start and
// returns the call and the index just past the closing '>'. ok is false when the
// tag has no usable name, never closes, or carries a present-but-malformed
// arguments value: a broken value must not degrade to {} and dispatch a call
// with its arguments dropped.
func parseInvocationAt(content string, start int) (ToolCall, int, bool) {
	seg := content[start:]

	// The name must belong to THIS tag: before its arguments= when it has one,
	// and before its closing '>' otherwise. Without the bound a nameless tag
	// borrows the name of the next one and is dispatched under it.
	limit := len(seg)
	if gt := strings.IndexByte(seg, '>'); gt >= 0 {
		limit = gt
	}
	if loc := inlineInvocationArgs.FindStringIndex(seg); loc != nil && loc[0] < limit {
		limit = loc[0]
	}
	nameMatch := inlineInvocationName.FindStringSubmatchIndex(seg)
	if nameMatch == nil || nameMatch[0] > limit {
		return ToolCall{}, 0, false
	}
	name := strings.TrimSpace(seg[nameMatch[2]:nameMatch[3]])
	if name == "" {
		return ToolCall{}, 0, false
	}

	// The first '>' after the name closes a no-arguments tag, and for a tag with
	// arguments it is a sentinel: this tag's own arguments= sits before it (the
	// '>' is then inside the JSON), a LATER tag's arguments= sits after it.
	gt := strings.IndexByte(seg[nameMatch[1]:], '>')
	firstGt := -1
	if gt >= 0 {
		firstGt = nameMatch[1] + gt
	}

	args := "{}"
	afterAttrs := nameMatch[1]
	if loc := inlineInvocationArgs.FindStringIndex(seg); loc != nil && (firstGt < 0 || loc[0] < firstGt) {
		valStart := loc[1]
		if valStart >= len(seg) || seg[valStart] != '{' {
			return ToolCall{}, 0, false
		}
		jsonEnd, ok := scanJSONObject(seg, valStart)
		if !ok {
			return ToolCall{}, 0, false
		}
		raw := seg[valStart:jsonEnd]
		if !json.Valid([]byte(raw)) {
			return ToolCall{}, 0, false
		}
		args = raw
		afterAttrs = jsonEnd
	}

	closeRel := strings.IndexByte(seg[afterAttrs:], '>')
	if closeRel < 0 {
		return ToolCall{}, 0, false
	}
	end := start + afterAttrs + closeRel + 1
	return ToolCall{Type: "function", Name: name, Arguments: args}, end, true
}

// scanJSONObject returns the index just past the '}' closing the object that
// opens at s[open], respecting quoted strings and escapes so braces inside string
// values do not end the scan early. ok is false if the object never closes.
func scanJSONObject(s string, open int) (int, bool) {
	depth := 0
	inString := false
	escaped := false
	for i := open; i < len(s); i++ {
		c := s[i]
		if inString {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i + 1, true
			}
		}
	}
	return 0, false
}

// inlineArguments renders the <parameter=key>value</parameter> pairs of one
// block as a JSON object in the order they appear. Values stay strings; the
// caller's argument coercion handles typing.
func inlineArguments(inner string) string {
	params := inlineParameter.FindAllStringSubmatch(inner, -1)
	var b strings.Builder
	b.WriteByte('{')
	for i, p := range params {
		if i > 0 {
			b.WriteByte(',')
		}
		writeJSONString(&b, strings.TrimSpace(p[1]))
		b.WriteByte(':')
		writeJSONString(&b, strings.TrimSpace(p[2]))
	}
	b.WriteByte('}')
	return b.String()
}

// writeJSONString encodes s without HTML escaping, so a query keeps readable &,
// <, > instead of \u00xx sequences.
func writeJSONString(b *strings.Builder, s string) {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(s)
	b.WriteString(strings.TrimRight(buf.String(), "\n"))
}

// inlineGate withholds streamed text from the first inline marker on. Prose
// before the marker streams normally, token by token; a trailing suffix that
// could be the start of a marker is held until it either grows into one or is
// proven to be text. Once a marker is seen the remainder is withheld: a model
// that has started a tool call emits nothing but tool calls afterwards.
type inlineGate struct {
	buffer     string
	suppressed bool
}

// push returns the portion of delta that is safe to hand on now.
func (g *inlineGate) push(delta string) string {
	if g.suppressed {
		return ""
	}
	g.buffer += delta
	earliest := -1
	for _, m := range inlineToolCallMarkers {
		if idx := strings.Index(g.buffer, m); idx >= 0 && (earliest < 0 || idx < earliest) {
			earliest = idx
		}
	}
	if earliest >= 0 {
		out := g.buffer[:earliest]
		g.buffer = ""
		g.suppressed = true
		return out
	}
	hold := 0
	for _, m := range inlineToolCallMarkers {
		if h := partialMarkerSuffixLen(g.buffer, m); h > hold {
			hold = h
		}
	}
	out := g.buffer[:len(g.buffer)-hold]
	g.buffer = g.buffer[len(g.buffer)-hold:]
	return out
}

// flush returns what is still held once the stream ends. A suffix that never
// grew into a marker is real text and must surface.
func (g *inlineGate) flush() string {
	if g.suppressed {
		return ""
	}
	out := g.buffer
	g.buffer = ""
	return out
}

// partialMarkerSuffixLen returns the length of the longest suffix of s that is a
// proper prefix of marker ("abc<too" against "<tool_call>" is 4).
func partialMarkerSuffixLen(s, marker string) int {
	max := len(marker) - 1
	if len(s) < max {
		max = len(s)
	}
	for n := max; n > 0; n-- {
		if strings.HasPrefix(marker, s[len(s)-n:]) {
			return n
		}
	}
	return 0
}

// inlineMarkerCount counts the opening markers in s, which is how many blocks
// the model started.
func inlineMarkerCount(s string) int {
	n := 0
	for _, m := range inlineToolCallMarkers {
		n += strings.Count(s, m)
	}
	return n
}

// inlineRecovery is what recovering an answer's inline markup produced.
type inlineRecovery struct {
	calls              []ToolCall
	content, reasoning string
	// channel names where the calls were found: "content" or "reasoning".
	channel string
	// blocks is how many markers the recovered channel carried; fewer calls
	// than blocks means some were truncated or malformed.
	blocks int
	// cut is set when text was withheld from a channel without a call coming
	// out of it: a marker the model started and never finished, or one it
	// merely quoted while its real call went out natively.
	cut bool
}

// recoverInline parses inline calls out of a finished answer, preferring the
// content channel, and cuts both channels at their first marker so the returned
// text is exactly what the gate let through live. Native calls win: when the
// endpoint populated tool_calls nothing is recovered, though the text is still
// cut, and the warning says so.
func recoverInline(content, reasoning string, native int) inlineRecovery {
	r := inlineRecovery{}
	if native == 0 {
		r.channel = "content"
		r.calls, _ = parseInlineToolCalls(content)
		if len(r.calls) == 0 {
			r.channel = "reasoning"
			r.calls, _ = parseInlineToolCalls(reasoning)
		}
		if len(r.calls) == 0 {
			r.channel = ""
		} else {
			r.blocks = inlineMarkerCount(map[string]string{"content": content, "reasoning": reasoning}[r.channel])
		}
	}
	r.content = cutAtFirstInlineMarker(content)
	r.reasoning = cutAtFirstInlineMarker(reasoning)
	// Text was cut from a channel that produced nothing.
	r.cut = (containsInlineToolMarker(content) && r.channel != "content") ||
		(containsInlineToolMarker(reasoning) && r.channel != "reasoning")
	return r
}

// warnings phrases the recovery for the caller. A recovered call is worth a
// warning because the endpoint's native field was empty; text cut without a
// call coming out of it is a dead end the caller would otherwise never see.
func (r inlineRecovery) warnings() []Warning {
	var ws []Warning
	if len(r.calls) > 0 {
		details := fmt.Sprintf("recovered %d inline tool call(s) from the %s channel; the endpoint sent no native tool_calls", len(r.calls), r.channel)
		if r.blocks > len(r.calls) {
			details += fmt.Sprintf("; %d of %d blocks did not parse (truncated or malformed) and were cut", r.blocks-len(r.calls), r.blocks)
		}
		ws = append(ws, Warning{Kind: WarnOther, Feature: "tool_calls", Details: details})
	}
	if r.cut {
		ws = append(ws, Warning{
			Kind:    WarnOther,
			Feature: "tool_calls",
			Details: "inline tool-call markup was cut from the answer without a call parsing out of it (truncated, malformed, or quoted alongside a native call)",
		})
	}
	return ws
}
