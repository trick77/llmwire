package llmwire

import (
	"fmt"
	"math/big"
	"net/http"
	"net/url"
	"strings"
	"time"

	// Embeds the IANA timezone database. Off-peak windows are evaluated in the
	// provider's own zone, and LoadLocation fails on an image that carries no
	// tzdata — a scratch container. Without this import that failure would turn
	// a priced call into an unpriced one on some deployments and not others,
	// which is the least debuggable kind of difference. Stdlib, so the
	// one-dependency rule is untouched; the cost is roughly 450KB of binary.
	_ "time/tzdata"

	"gopkg.in/yaml.v3"
)

// Pricing.
//
// ONE unit, always: USD, at the vendor's published pay-as-you-go list rate.
// Subscription flat rates are deliberately not modelled. A call actually billed
// as plan credits on a token-plan host still reports the list USD rate here,
// because a credit figure is not comparable between models and reporting nothing
// at all leaves a budget cap with nothing to check.
//
// TWO sources, no third: this table, embedded in profiles.yaml, and a gateway's
// own per-call figure from a response header. There is no runtime catalogue
// fetch and no caller-supplied rate map — that would buy a network call, a
// cache, a staleness window and a fallback table, and this table IS the fallback
// table. A stale rate is fixed by a library release, deliberately.
//
// Integer arithmetic end to end. A float total drifts in the digits displayed,
// which is exactly where a billing figure is read.

// Cost is what one call cost.
//
// A FromTable figure is the vendor's published pay-as-you-go LIST rate in USD. It
// is a comparable equivalent, NOT an invoice: a call billed as subscription
// credits on a token-plan host still reports the list rate here, because a credit
// figure cannot be compared between models and an Unpriced one leaves a budget
// cap with nothing to check.
type Cost struct {
	// NanoUSD is billionths of one US dollar. Integer end to end, because a float
	// total drifts in the digits displayed — which is where a billing figure is
	// read.
	NanoUSD int64 `json:"nano_usd"`
	// Provenance is what gives NanoUSD its meaning, and its zero value is
	// Unpriced: a forgotten fill then reads "unknown", never "the table said
	// zero", which would read as free.
	Provenance CostProvenance `json:"provenance"`
	// AppliedAt is the clock value captured at REQUEST START. Which instant bills
	// is unspecified by both vendors, so this records the one that was used.
	AppliedAt time.Time `json:"applied_at,omitempty"`
	// Window and Tier name the multipliers that applied, empty when none did.
	// Always empty under Reported: a gateway's figure is already final.
	Window string `json:"window,omitempty"`
	Tier   string `json:"tier,omitempty"`
}

// CostProvenance says where a figure came from. There is no unit here: every
// figure is USD, and a one-value enum would be a switch nobody should write.
type CostProvenance int

const (
	// Unpriced is the ZERO VALUE, deliberately: no rate was available. NanoUSD is
	// 0, and 0 means unknown, never free.
	Unpriced CostProvenance = iota
	// FromTable was computed from the embedded list rate. See the Cost doc
	// comment: a list-rate equivalent, not an invoice.
	FromTable
	// Reported came from the gateway, which knows what it actually charged.
	Reported
)

func (p CostProvenance) String() string {
	switch p {
	case FromTable:
		return "from-table"
	case Reported:
		return "reported"
	default:
		return "unpriced"
	}
}

// MarshalText renders the provenance as its name. A bare integer in a log or a
// stored record would be unreadable exactly where it matters most.
func (p CostProvenance) MarshalText() ([]byte, error) { return []byte(p.String()), nil }

// UnmarshalText reads back what MarshalText wrote, so a stored Cost round-trips.
// Without it the name marshals out fine and then fails to decode, which turns
// "keep a record of what this call cost" into a runtime error at read time.
//
// An unrecognised name decodes to Unpriced rather than failing: a record written by
// a later version that knows a fourth source should read as "provenance unknown",
// which is true, rather than making the whole record unreadable.
func (p *CostProvenance) UnmarshalText(text []byte) error {
	switch string(text) {
	case "from-table":
		*p = FromTable
	case "reported":
		*p = Reported
	default:
		*p = Unpriced
	}
	return nil
}

// rate is a token price in nano-USD per MILLION tokens: 1e-9 dollars, per 1e6
// tokens. Both scales are needed. Per-million is how every vendor publishes, and
// nano is the smallest unit any of them quotes — a cache-read lane of $0.0036
// per million is 3_600_000 here, and the same figure in nano-USD per TOKEN would
// be 3.6, which no integer can hold.
type rate int64

// nanoPerUSD is the nano-USD in one dollar.
const nanoPerUSD = 1_000_000_000

// UnmarshalYAML reads the vendor's own decimal spelling ("0.15", "0.0036") and
// converts it EXACTLY, from the literal text.
//
// Never through float64: 0.0036 has no exact binary representation, so the
// integer that came out would depend on the rounding of the very value a test
// pins. yaml.Node.Value hands over the characters the author typed, and the
// conversion below is a digit shift.
func (r *rate) UnmarshalYAML(node *yaml.Node) error {
	s := strings.TrimSpace(node.Value)
	if s == "" {
		return fmt.Errorf("empty rate")
	}
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(strings.TrimPrefix(s, "-"), "+")

	intPart, fracPart, _ := strings.Cut(s, ".")
	if intPart == "" {
		intPart = "0"
	}
	if !allDigits(intPart) || (fracPart != "" && !allDigits(fracPart)) {
		// The exponent form is legal YAML and unreadable in a price table, so it
		// gets its own message: refusing it keeps every rate in the file
		// comparable by eye against the vendor's page, which is the only way the
		// hand cross-check works. Checked here rather than up front, so a word
		// that merely contains an "e" is reported as what it is.
		if isExponentForm(s) {
			return fmt.Errorf("rate %q: write the decimal the vendor publishes, not scientific notation",
				node.Value)
		}
		return fmt.Errorf("rate %q is not a decimal number", node.Value)
	}
	// Nine fractional digits is exactly nano precision. A tenth cannot be
	// represented, so it is refused rather than rounded away: a silently dropped
	// digit in a price is the same class of failure as a silently dropped
	// parameter.
	if len(fracPart) > 9 {
		return fmt.Errorf("rate %q has more than 9 decimal places, which is finer than nano-USD", node.Value)
	}
	frac := fracPart + strings.Repeat("0", 9-len(fracPart))

	whole, err := parseDigits(intPart)
	if err != nil {
		return fmt.Errorf("rate %q: %w", node.Value, err)
	}
	fracVal, err := parseDigits(frac)
	if err != nil {
		return fmt.Errorf("rate %q: %w", node.Value, err)
	}
	// Checked here too, not only inside parseDigits: the scaling multiply is where
	// a plausible-looking integer part overflows, and some wrapped values come back
	// POSITIVE — a silently wrong rate that the negative-rate check never sees.
	if whole > (1<<62)/nanoPerUSD {
		return fmt.Errorf("rate %q is too large to hold in nano-USD", node.Value)
	}
	v := whole*nanoPerUSD + fracVal
	if neg {
		v = -v
	}
	*r = rate(v)
	return nil
}

// isExponentForm reports whether s looks like 1.5e-01: a decimal mantissa, an
// e, and a signed integer exponent.
func isExponentForm(s string) bool {
	mant, exp, found := strings.Cut(strings.ToLower(s), "e")
	if !found {
		return false
	}
	intPart, fracPart, _ := strings.Cut(mant, ".")
	if intPart != "" && !allDigits(intPart) {
		return false
	}
	if fracPart != "" && !allDigits(fracPart) {
		return false
	}
	exp = strings.TrimPrefix(strings.TrimPrefix(exp, "-"), "+")
	return allDigits(exp)
}

func allDigits(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

// parseDigits converts an all-digit string, refusing an overflow rather than
// wrapping: a wrapped rate could come out negative, and a negative rate credits
// the caller for tokens they spent.
func parseDigits(s string) (int64, error) {
	var v int64
	for _, c := range s {
		d := int64(c - '0')
		if v > (1<<62)/10 {
			return 0, fmt.Errorf("value is too large")
		}
		v = v*10 + d
	}
	return v, nil
}

// CostBlock is one model's published rate, per lane.
//
// Whole block or nothing: every lane a chat model bills is mandatory. A
// "missing cache rate falls back to the input rate" rule would be a call-site
// default, and this registry resolves everything at load precisely so that no
// call site invents a value. A lane the vendor genuinely gives away is an
// explicit 0.
type CostBlock struct {
	// Lanes, in nano-USD per million tokens. Pointers so an absent lane is
	// distinguishable from an explicit zero: absent is a load refusal, zero is
	// the vendor saying free.
	Input      *rate `yaml:"input"`
	CacheRead  *rate `yaml:"cache_read"`
	CacheWrite *rate `yaml:"cache_write"`
	Output     *rate `yaml:"output"`

	// Tiers raise the rate for a large prompt. Selected by the request.
	Tiers []CostTier `yaml:"tiers"`
	// Windows move it by wall-clock time, in the provider's own zone. None ships
	// attached today — see the note on glm-5.3-flash in profiles.yaml.
	Windows []CostWindow `yaml:"windows"`
	// OutsideWindowsPermille applies when no window matches. Defaults to 1000
	// (the list rate) at load, because both vendors express the rule as "peak is
	// list, everything else is cheaper".
	OutsideWindowsPermille int32 `yaml:"outside_windows_permille"`

	// SourceURL and VerifiedOn are mandatory. Two models in this registry have
	// prices differing by 2x between the vendor's page and a published
	// catalogue, and a rate whose origin is not recorded cannot be audited when
	// that happens again.
	SourceURL  string `yaml:"source_url"`
	VerifiedOn string `yaml:"verified_on"`
}

// CostTier is a surcharge for a large prompt, selected by the request.
type CostTier struct {
	Name string `yaml:"name"`
	// MinInputTokens is the threshold on the prompt's TOTAL tokens, cached ones
	// included: vendors threshold on how much context they had to hold, not on
	// what they ended up billing at the full rate.
	MinInputTokens int64 `yaml:"min_input_tokens"`
	// MultiplierPermille applies to the whole call. At least 1000: a tier making
	// a call cheaper would under-report, and under-reporting is the direction
	// that silently raises what a budget cap allows.
	MultiplierPermille int32 `yaml:"multiplier_permille"`
}

// CostWindow is a wall-clock multiplier in the provider's own timezone.
type CostWindow struct {
	Name string `yaml:"name"`
	// Zone is an IANA name, e.g. Asia/Singapore. Resolved at load, so a bad zone
	// fails the load and pricing never fails at call time.
	Zone string `yaml:"zone"`
	// Days are Mon..Sun; empty means every day.
	Days []string `yaml:"days"`
	// From is inclusive, To is EXCLUSIVE, both "HH:MM". No midnight wrap:
	// express that as two windows, so "which window applied" stays answerable.
	From               string `yaml:"from"`
	To                 string `yaml:"to"`
	MultiplierPermille int32  `yaml:"multiplier_permille"`

	// Resolved at load.
	loc     *time.Location
	days    [7]bool
	fromMin int
	toMin   int
}

var weekdayNames = map[string]time.Weekday{
	"sun": time.Sunday, "mon": time.Monday, "tue": time.Tuesday,
	"wed": time.Wednesday, "thu": time.Thursday, "fri": time.Friday,
	"sat": time.Saturday,
}

// validate resolves and checks a cost block. Called from Profile.validate, so a
// malformed rate fails `go test` rather than a production invoice.
func (b *CostBlock) validate(p Profile) error {
	bad := func(format string, args ...any) error {
		return fmt.Errorf("profile %q cost: "+format, append([]any{p.ID}, args...)...)
	}

	// A gateway reports its own per-call spend and is never priced from a table:
	// the proxy knows the real deployment and whatever discount applies to it,
	// and this table does not. Checked here for an unbased gateway entry;
	// resolve drops an INHERITED block for the same reason.
	if p.Gateway != "" {
		return bad("a gateway route is priced from its response header, never from the table")
	}

	if b.SourceURL == "" {
		return bad("source_url is required; a rate whose origin is not recorded cannot be audited")
	}
	u, err := url.Parse(b.SourceURL)
	if err != nil || u.Scheme != "https" || u.Host == "" {
		return bad("source_url %q must be an https URL", b.SourceURL)
	}
	if b.VerifiedOn == "" {
		return bad("verified_on is required")
	}
	// Deliberately NOT compared against the wall clock. A "must not be in the
	// future" rule would make the load pass or fail depending on the day it ran,
	// which is the one kind of check this package refuses to build.
	if _, err := time.Parse("2006-01-02", b.VerifiedOn); err != nil {
		return bad("verified_on %q is not a YYYY-MM-DD date", b.VerifiedOn)
	}

	lanes := []struct {
		name string
		r    *rate
	}{
		{"input", b.Input}, {"output", b.Output},
		{"cache_read", b.CacheRead}, {"cache_write", b.CacheWrite},
	}
	for _, l := range lanes {
		if l.r != nil && *l.r < 0 {
			return bad("%s rate is negative", l.name)
		}
	}

	if p.Endpoint == EndpointEmbeddings {
		// One lane, and the others must be absent rather than zero: a zero
		// output rate on an embeddings model reads as "measured free" when what
		// it means is "there is no such lane".
		if b.Input == nil {
			return bad("input rate is required")
		}
		for _, l := range lanes[1:] {
			if l.r != nil {
				return bad("an embeddings profile must not declare a %s rate", l.name)
			}
		}
	} else {
		for _, l := range lanes {
			if l.r == nil {
				return bad("%s rate is required; a cost block is whole or absent, because a lane "+
					"falling back to another lane's rate is a call-site default", l.name)
			}
		}
		if *b.CacheRead > *b.Input {
			return bad("cache_read (%d) is dearer than input (%d); a cache discount cannot cost more "+
				"than fresh tokens, so this is a transposition", *b.CacheRead, *b.Input)
		}
	}

	if err := b.validateTiers(bad); err != nil {
		return err
	}
	return b.validateWindows(bad)
}

func (b *CostBlock) validateTiers(bad func(string, ...any) error) error {
	seenName := map[string]bool{}
	seenThreshold := map[int64]bool{}
	for i := range b.Tiers {
		t := &b.Tiers[i]
		switch {
		case t.Name == "":
			return bad("tier #%d has no name; Cost.Tier reports it", i)
		case seenName[t.Name]:
			return bad("duplicate tier name %q", t.Name)
		case t.MinInputTokens <= 0:
			return bad("tier %q needs a positive min_input_tokens", t.Name)
		case seenThreshold[t.MinInputTokens]:
			return bad("two tiers share min_input_tokens %d, so which one applies is unanswerable",
				t.MinInputTokens)
		case t.MultiplierPermille < 1000:
			return bad("tier %q multiplier_permille is %d; a tier below 1000 would make a big prompt "+
				"cheaper and under-report spend", t.Name, t.MultiplierPermille)
		}
		seenName[t.Name] = true
		seenThreshold[t.MinInputTokens] = true
	}
	return nil
}

func (b *CostBlock) validateWindows(bad func(string, ...any) error) error {
	seenName := map[string]bool{}
	for i := range b.Windows {
		w := &b.Windows[i]
		if w.Name == "" {
			return bad("window #%d has no name; Cost.Window reports it", i)
		}
		if seenName[w.Name] {
			return bad("duplicate window name %q", w.Name)
		}
		seenName[w.Name] = true
		if w.Zone == "" {
			return bad("window %q has no zone; a wall-clock rule without a timezone is ambiguous", w.Name)
		}
		loc, err := time.LoadLocation(w.Zone)
		if err != nil {
			return bad("window %q zone %q: %v", w.Name, w.Zone, err)
		}
		w.loc = loc
		if w.MultiplierPermille <= 0 {
			return bad("window %q needs a positive multiplier_permille", w.Name)
		}
		if w.fromMin, err = parseClock(w.From); err != nil {
			return bad("window %q from: %v", w.Name, err)
		}
		if w.toMin, err = parseClock(w.To); err != nil {
			return bad("window %q to: %v", w.Name, err)
		}
		if w.fromMin >= w.toMin {
			return bad("window %q runs %s to %s; from must be before to, and a window crossing "+
				"midnight is written as two windows", w.Name, w.From, w.To)
		}
		for _, d := range w.Days {
			wd, ok := weekdayNames[strings.ToLower(d)]
			if !ok {
				return bad("window %q day %q is not Mon..Sun", w.Name, d)
			}
			w.days[wd] = true
		}
		if len(w.Days) == 0 {
			for d := range w.days {
				w.days[d] = true
			}
		}
	}
	// One zone per block. A cost block belongs to one vendor, and comparing clock
	// ranges across zones would need every pair mapped onto a common timeline — so
	// instead of an overlap check that silently skips the cross-zone pairs (leaving
	// "the first match" a document-order accident for exactly those), mixing zones
	// is refused outright.
	for i := range b.Windows {
		if b.Windows[i].Zone != b.Windows[0].Zone {
			return bad("windows %q and %q are in different zones (%s, %s); one cost block is one "+
				"vendor's schedule, and clock ranges in different zones cannot be compared for "+
				"overlap", b.Windows[0].Name, b.Windows[i].Name, b.Windows[0].Zone, b.Windows[i].Zone)
		}
	}
	// Overlaps are refused so "the first matching window" is a deterministic
	// rule rather than a document-order accident.
	for i := range b.Windows {
		for j := i + 1; j < len(b.Windows); j++ {
			a, c := &b.Windows[i], &b.Windows[j]
			if a.fromMin >= c.toMin || c.fromMin >= a.toMin {
				continue
			}
			for d := 0; d < 7; d++ {
				if a.days[d] && c.days[d] {
					return bad("windows %q and %q overlap in %s", a.Name, c.Name, a.Zone)
				}
			}
		}
	}

	if b.OutsideWindowsPermille < 0 {
		return bad("outside_windows_permille is negative")
	}
	if b.OutsideWindowsPermille == 0 {
		b.OutsideWindowsPermille = 1000
	}
	if len(b.Windows) == 0 && b.OutsideWindowsPermille != 1000 {
		return bad("outside_windows_permille is set but no window is defined, which makes it an " +
			"unconditional multiplier; put that in the lane rates instead")
	}
	return nil
}

// priceCall prices one completed call.
//
// One entry point for both lanes, so the rule that a gateway route is NEVER
// priced from the table is enforced in a single place rather than at four call
// sites. hdr and status are the response's; at is the clock value captured at
// request start.
//
// The returned Cost is always meaningful — its zero value is Unpriced — and the
// returned warnings belong on the call's own []Warning.
func priceCall(p *Profile, u Usage, hdr http.Header, status int, at time.Time) (Cost, []Warning) {
	unpriced := func(format string, args ...any) (Cost, []Warning) {
		return Cost{AppliedAt: at}, []Warning{{
			Kind:    WarnOther,
			Feature: "cost",
			Details: fmt.Sprintf(format, args...),
		}}
	}

	// Defensive, and load-bearing: a gateway sends the cost header even on some
	// failures, and "0" under a non-2xx means ERRORED, not free. Callers are not
	// supposed to price an error path at all, and this is where that can never be
	// misread.
	if status < 200 || status >= 300 {
		return Cost{AppliedAt: at}, nil
	}

	if p.Gateway != "" {
		return priceFromGateway(p, hdr, at)
	}
	if p.Cost == nil {
		return unpriced("no verified rate for model %q; the cost is unknown, not zero", p.ID)
	}
	// A total that was never reported makes the call unpriceable. Treating a
	// missing count as zero would under-report, and under-reporting is what
	// silently raises the ceiling a budget cap enforces.
	if u.Input.Total == nil {
		return unpriced("model %q reported no prompt tokens, so the call cannot be priced", p.ID)
	}
	if p.Endpoint == EndpointChat && u.Output.Total == nil {
		return unpriced("model %q reported no completion tokens, so the call cannot be priced", p.ID)
	}

	b := p.Cost
	// Cache reads are SUBTRACTED, not added: prompt_tokens already includes
	// cached_tokens, and parseUsage has derived and clamped NoCache. Reasoning is
	// never a lane of its own — completion_tokens already contains it, so pricing
	// Output.Reasoning as well would double-count every call these models make.
	//
	// Summed in big.Int, exactly once. The intermediate is tokens x
	// nano-per-million x permille x permille, which for a million-token prompt at
	// a dollar per million runs to 1e21 — twenty times past int64. Rounding each
	// lane to nano first would fit, but it would round four times instead of once
	// and over-report every call by up to four nano.
	scaled := new(big.Int)
	addLane(scaled, u.Input.NoCache, b.Input)
	addLane(scaled, u.Input.CacheRead, b.CacheRead)
	addLane(scaled, u.Input.CacheWrite, b.CacheWrite)
	addLane(scaled, u.Output.Total, b.Output)

	tier, tierPermille := b.selectTier(*u.Input.Total)
	window, windowPermille := b.selectWindow(at)
	scaled.Mul(scaled, big.NewInt(int64(tierPermille)))
	scaled.Mul(scaled, big.NewInt(int64(windowPermille)))

	// ONE rounding site, on the summed total, rounding UP so llmwire never
	// under-reports. The divisor is the three scales the numerator carries: a
	// million tokens per rate, and a thousand per multiplier, twice.
	nano, err := ceilDivBig(scaled, big.NewInt(1_000_000*1_000*1_000))
	if err != nil {
		return unpriced("model %q: the call is too large to price: %v", p.ID, err)
	}

	return Cost{
		NanoUSD:    nano,
		Provenance: FromTable,
		AppliedAt:  at,
		Window:     window,
		Tier:       tier,
	}, nil
}

// addLane adds one lane's tokens x rate into the running total. A nil count
// contributes nothing: "not reported" is not "zero", but a lane nobody reported
// cannot be charged for either.
func addLane(sum *big.Int, tokens *int64, r *rate) {
	if tokens == nil || r == nil {
		return
	}
	term := new(big.Int).Mul(big.NewInt(*tokens), big.NewInt(int64(*r)))
	sum.Add(sum, term)
}

// ceilDivBig divides, rounding away from zero, and refuses a result that does not
// fit rather than wrapping — a wrapped total could come out negative, which would
// credit the caller for tokens they spent.
func ceilDivBig(n, d *big.Int) (int64, error) {
	q, m := new(big.Int), new(big.Int)
	q.DivMod(n, d, m)
	if m.Sign() != 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsInt64() {
		return 0, fmt.Errorf("the cost does not fit in nano-USD")
	}
	return q.Int64(), nil
}

// selectTier returns the highest threshold the prompt met.
//
// Thresholded on the prompt TOTAL, cached tokens included: vendors price by how
// much context they had to hold, not by what they ended up billing at full rate.
func (b *CostBlock) selectTier(inputTotal int64) (string, int32) {
	name, permille := "", int32(1000)
	var best int64
	for _, t := range b.Tiers {
		if inputTotal >= t.MinInputTokens && t.MinInputTokens > best {
			best, name, permille = t.MinInputTokens, t.Name, t.MultiplierPermille
		}
	}
	return name, permille
}

// selectWindow evaluates the instant in each window's OWN zone.
//
// Half-open [from, to): 17:59 is inside a window ending at 18:00 and 18:00 is
// not. Overlaps are refused at load, so "the first match" is deterministic rather
// than a document-order accident.
func (b *CostBlock) selectWindow(at time.Time) (string, int32) {
	for _, w := range b.Windows {
		local := at.In(w.loc)
		if !w.days[local.Weekday()] {
			continue
		}
		min := local.Hour()*60 + local.Minute()
		if min >= w.fromMin && min < w.toMin {
			return w.Name, w.MultiplierPermille
		}
	}
	if len(b.Windows) == 0 {
		return "", 1000
	}
	return "", b.OutsideWindowsPermille
}

// litellmCostHeader carries the gateway's own per-call spend.
const litellmCostHeader = "x-litellm-response-cost"

// priceFromGateway reads a proxy's reported cost.
//
// Never falls back to the table: the proxy knows which deployment ran and what it
// is charged at, and a list rate substituted here would be a confident figure for
// a call nobody priced. An absent header is normal — a LiteLLM STREAM carries no
// cost header at all, because the headers are built before the stream is consumed
// — so that case degrades to Unpriced rather than to a guess.
func priceFromGateway(p *Profile, hdr http.Header, at time.Time) (Cost, []Warning) {
	warn := func(format string, args ...any) (Cost, []Warning) {
		return Cost{AppliedAt: at}, []Warning{{
			Kind:    WarnOther,
			Feature: "cost",
			Details: fmt.Sprintf(format, args...),
		}}
	}
	raw := hdr.Get(litellmCostHeader)
	if raw == "" || isNoneLiteral(raw) {
		return warn("gateway %q reported no cost for model %q (a stream never does), so the cost "+
			"is unknown, not zero", p.Gateway, p.ID)
	}
	nano, err := parseReportedCost(raw)
	if err != nil {
		return warn("gateway %q reported an unusable cost %q: %v", p.Gateway, raw, err)
	}
	return Cost{NanoUSD: nano, Provenance: Reported, AppliedAt: at}, nil
}

// parseReportedCost converts a gateway's decimal cost into nano-USD.
//
// big.Rat rather than strconv.ParseFloat, for two reasons. ParseFloat ACCEPTS
// "NaN" and "±Inf", and a NaN cost makes every `total >= limit` comparison false —
// silently disabling the budget cap it was supposed to feed. And the gateway sends
// scientific notation ("1.23e-05"), which big.Rat converts exactly, so the
// integer does not depend on float rounding of the value under test.
func parseReportedCost(s string) (int64, error) {
	s = strings.TrimSpace(s)
	r, ok := new(big.Rat).SetString(s)
	if !ok {
		// This is the branch NaN and Inf land in, which is the whole reason for
		// the type choice.
		return 0, fmt.Errorf("not a finite decimal number")
	}
	if r.Sign() < 0 {
		return 0, fmt.Errorf("negative cost")
	}
	r.Mul(r, new(big.Rat).SetInt64(nanoPerUSD))
	// Round up, the same direction as the table lane: a sub-nano charge reports 1
	// rather than disappearing.
	nano, err := ceilDivBig(r.Num(), r.Denom())
	if err != nil {
		return 0, fmt.Errorf("cost %q: %w", s, err)
	}
	return nano, nil
}

// parseClock reads "HH:MM" into minutes since midnight.
func parseClock(s string) (int, error) {
	t, err := time.Parse("15:04", s)
	if err != nil {
		return 0, fmt.Errorf("%q is not HH:MM", s)
	}
	return t.Hour()*60 + t.Minute(), nil
}

// clone deep-copies a cost block. The resolved *time.Location is immutable and
// is shared rather than copied.
func (b *CostBlock) clone() *CostBlock {
	if b == nil {
		return nil
	}
	out := *b
	out.Input = copyRate(b.Input)
	out.CacheRead = copyRate(b.CacheRead)
	out.CacheWrite = copyRate(b.CacheWrite)
	out.Output = copyRate(b.Output)
	out.Tiers = append([]CostTier(nil), b.Tiers...)
	out.Windows = append([]CostWindow(nil), b.Windows...)
	return &out
}

func copyRate(r *rate) *rate {
	if r == nil {
		return nil
	}
	v := *r
	return &v
}
