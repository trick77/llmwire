package llmwire

import "strings"

// The close tag, and the open tag that makes a block of it.
const (
	strayCloseTag = "</think>"
	thinkOpenTag  = "<think>"
)

// leakCanApply reports whether a reply to req can carry the leak at all. It
// was observed with thinking switched off, where no reasoning block belongs in
// content. A JSON reply is never cut: a string value may hold the tag, and a
// cut would leave a fragment that no longer parses.
func leakCanApply(req ChatRequest) bool {
	if _, off := req.Reasoning.(reasoningOff); !off {
		return false
	}
	return req.ResponseFormat.Kind == FormatUnset || req.ResponseFormat.Kind == FormatText
}

// cutStrayCloseTag splits content at the FIRST close tag when it ends a
// draft: what follows is the answer, what precedes it joins reasoning, so a
// caller reading the reasoning channel still sees it and one parsing the
// answer does not. The first, not the last: an answer that quotes the tag
// after the leak keeps its opening. An open tag anywhere but the very start
// means the pair is quoted text, not a leak, and content is left alone.
func cutStrayCloseTag(content, reasoning string) (string, string, bool) {
	i := strings.Index(content, strayCloseTag)
	if i < 0 {
		return content, reasoning, false
	}
	head := strings.TrimSpace(content[:i])
	draft, opened := strings.CutPrefix(head, thinkOpenTag)
	if strings.Contains(draft, thinkOpenTag) || (!opened && strings.Contains(head, thinkOpenTag)) {
		return content, reasoning, false
	}
	if draft = strings.TrimSpace(draft); draft != "" {
		if reasoning != "" {
			reasoning += "\n"
		}
		reasoning += draft
	}
	return strings.TrimSpace(content[i+len(strayCloseTag):]), reasoning, true
}
