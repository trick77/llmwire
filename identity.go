package llmwire

import (
	"crypto/rand"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Presenting as opencode.
//
// Some of the endpoints this library targets are sold as the backend for a
// particular client. MiMo's token-plan host is an opencode-facing product: a
// neutral User-Agent is not what its production traffic looks like, and three
// backends independently arrived at sending opencode's own client string plus
// the session header pair it sends. Each carried its own copy of the string, the
// header names and the id-minting routine — which is the per-repo relearning
// this library exists to end.
//
// Config.EmulateOpenCode is the whole interface. It changes headers only, never
// a request body, and it takes no session id from the caller: a Client mints its
// own at construction and rotates it after an idle gap (see sessionIdleRotation),
// the way one running opencode is one session until its user steps away. An
// application that wants calls grouped differently makes another Client, which
// is cheap when it shares the *http.Client.
//
// On an endpoint that does not care — Z.ai's general host, OpenAI — the headers
// are inert and cost nothing. On one that does, they are the difference between
// being served and being refused as a bot.

// OpenCodeUserAgent is the client string opencode sends: the app, the SDK that
// built the request, and the runtime it ran on. Pinned to one release rather
// than tracking opencode's, since the point is a plausible client, not the
// latest one.
const OpenCodeUserAgent = "opencode/1.18.11 ai-sdk/openai-compatible/3.0.20 ai-sdk/provider-utils/5.0.18 runtime/bun/1.3.14"

// The two headers opencode sends, both carrying the same id. The upstream sends
// the pair back too; "affinity" is what it is for — pinning the many calls one
// session makes to a single upstream node.
const (
	HeaderSessionID       = "X-Session-Id"
	HeaderSessionAffinity = "X-Session-Affinity"
)

// Session ids mirror the shape the upstream issues, e.g.
//
//	ses_ 0367809bfffe ejtHKm95o6rU4mQ
//	│    └─12 hex────┘ └─14 base62───┘
//	│    timestamp+counter   random
//	prefix
//
// The 12 hex digits are the bitwise inversion of (millis << 12 | counter),
// truncated to 48 bits and written big-endian — a 12-bit per-process counter
// keeps ids minted in the same millisecond distinct, and the inversion is what
// gives upstream ids their characteristic trailing f's.
const (
	sessionIDPrefix   = "ses_"
	sessionIDRandomLn = 14
	sessionIDAlphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

var sessionCounter atomic.Uint64

// newSessionID mints an id in the shape opencode's upstream issues.
func newSessionID() string {
	millis := uint64(time.Now().UnixMilli())
	counter := sessionCounter.Add(1) & 0xFFF // 12 bits
	stamp := ^(millis<<12 | counter) & 0xFFFFFFFFFFFF
	return fmt.Sprintf("%s%012x%s", sessionIDPrefix, stamp, randomBase62(sessionIDRandomLn))
}

// randomBase62 draws n characters from the base62 alphabet. Rejection sampling
// keeps the draw unbiased; if the system entropy source fails the id degrades to
// the alphabet's first character rather than failing a model call, since this is
// an opaque routing token and not a secret.
func randomBase62(n int) string {
	const limit = 256 - (256 % len(sessionIDAlphabet)) // largest unbiased byte range
	out := make([]byte, 0, n)
	buf := make([]byte, n)
	for len(out) < n {
		if _, err := rand.Read(buf); err != nil {
			for len(out) < n {
				out = append(out, sessionIDAlphabet[0])
			}
			break
		}
		for _, b := range buf {
			if int(b) >= limit {
				continue
			}
			out = append(out, sessionIDAlphabet[int(b)%len(sessionIDAlphabet)])
			if len(out) == n {
				break
			}
		}
	}
	return string(out)
}

// sessionIdleRotation is how long a Client may go without a call before its
// next call starts a new session.
//
// A session id minted once and carried for the life of a process looks nothing
// like the traffic it is imitating: a person's session has a start and an end,
// and a server that runs for weeks would otherwise present one session with ten
// thousand calls in it. Rotating on an idle GAP rather than on a clock keeps the
// property that matters — a burst of related calls seconds apart stays pinned to
// one upstream node, which is what the affinity header is for — while a process
// that sat quiet overnight comes back as a fresh session, which is what a person
// opening the app again would be. Thirty minutes is a judgment about what
// "stepped away" means, not a measurement of anything.
const sessionIdleRotation = 30 * time.Minute

// session is the opencode session state one Client carries: the current id and
// when it was last used.
type session struct {
	mu       sync.Mutex
	id       string
	lastUsed time.Time
	now      func() time.Time
}

func newSession(now func() time.Time) *session {
	return &session{id: newSessionID(), lastUsed: now(), now: now}
}

// current returns the id to send on a call made now, rotating it first if the
// client has been idle past sessionIdleRotation.
func (s *session) current() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	if now.Sub(s.lastUsed) > sessionIdleRotation {
		s.id = newSessionID()
	}
	s.lastUsed = now
	return s.id
}
