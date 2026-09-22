package llmwire

import "strings"

// strayCloseTag is the closing half of a reasoning block. A model that
// leaks it writes a draft, this tag and then its answer, all into content,
// with no opening tag and no reasoning channel.
const strayCloseTag = "</think>"

// cutStrayCloseTag splits content at the LAST stray close tag: what follows
// is the answer, what precedes it is a draft and joins reasoning, so a caller
// reading the reasoning channel still sees it and one parsing the answer does
// not. Content without the tag is returned unchanged.
func cutStrayCloseTag(content, reasoning string) (string, string, bool) {
	i := strings.LastIndex(content, strayCloseTag)
	if i < 0 {
		return content, reasoning, false
	}
	draft := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(content[:i]), "<think>"))
	if draft != "" {
		if reasoning != "" {
			reasoning += "\n"
		}
		reasoning += draft
	}
	return strings.TrimSpace(content[i+len(strayCloseTag):]), reasoning, true
}
