package llmwire

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// SpoolTransport writes every response it carries to a directory, one file per
// response, as the status line, the headers and the body exactly as they came
// off the wire. It is the debugging aid for the question no log line answers:
// what did the endpoint actually send. A stream is written once the body is
// closed, so the file holds every frame the client read, including a truncated
// tail when the read failed.
//
// Handed to Config.HTTPClient as the transport:
//
//	hc := &http.Client{Transport: llmwire.NewSpoolTransport("logs/llm", nil, nil)}
//	client := llmwire.New(llmwire.Config{HTTPClient: hc})
//
// Files are named <UTC timestamp>-<sequence>.http. The directory is created on
// first use with mode 0700 and the files with 0600: the bodies are the
// caller's prompts and the model's answers. A file that cannot be written is
// dropped silently; a diagnostic must never fail the request it observes.
//
// The request body is not spooled. The response is what the caller could not
// otherwise see; the request is what the caller built.
type SpoolTransport struct {
	dir  string
	next http.RoundTripper
	skip func(*http.Request) bool
	seq  atomic.Uint64
}

// NewSpoolTransport wraps next, writing responses under dir. A nil next means
// http.DefaultTransport. skip, when set, is asked per request and a true
// answer leaves that response unwritten — the hook for a request whose body
// must not touch disk, such as a turn the caller promised to keep ephemeral.
func NewSpoolTransport(dir string, next http.RoundTripper, skip func(*http.Request) bool) *SpoolTransport {
	if next == nil {
		next = http.DefaultTransport
	}
	return &SpoolTransport{dir: dir, next: next, skip: skip}
}

// RoundTrip implements http.RoundTripper.
func (t *SpoolTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.next.RoundTrip(req)
	if err != nil || resp == nil || resp.Body == nil {
		return resp, err
	}
	if t.skip != nil && t.skip(req) {
		return resp, nil
	}
	resp.Body = &spoolBody{
		ReadCloser: resp.Body,
		resp:       resp,
		path:       filepath.Join(t.dir, fmt.Sprintf("%s-%06d.http", time.Now().UTC().Format("20060102T150405.000000000Z"), t.seq.Add(1))),
	}
	return resp, nil
}

// spoolBody tees the body into a buffer and writes the file on Close. On
// Close, not on EOF: a stream that failed mid-way never reaches EOF, and its
// partial body is exactly the evidence worth keeping.
//
// The buffer is guarded: net/http tolerates a Close racing a Read from another
// goroutine, and a caller using this transport on a bare http.Client may do
// exactly that from a watchdog.
type spoolBody struct {
	io.ReadCloser
	resp *http.Response
	path string
	mu   sync.Mutex
	body bytes.Buffer
	once sync.Once
}

func (b *spoolBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if n > 0 {
		b.mu.Lock()
		b.body.Write(p[:n])
		b.mu.Unlock()
	}
	return n, err
}

func (b *spoolBody) Close() error {
	err := b.ReadCloser.Close()
	b.once.Do(b.write)
	return err
}

func (b *spoolBody) write() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(b.path), 0o700); err != nil {
		return
	}
	var out bytes.Buffer
	proto := b.resp.Proto
	if proto == "" {
		proto = "HTTP/1.1"
	}
	fmt.Fprintf(&out, "%s %s\n", proto, b.resp.Status)
	if err := b.resp.Header.Write(&out); err != nil {
		return
	}
	out.WriteString("\n")
	b.body.WriteTo(&out)
	_ = os.WriteFile(b.path, out.Bytes(), 0o600)
}
