"""Coverage radar for the context wiki: find concepts the wiki should hold but does not.

Why this exists: the daily merge judges each batch of items in isolation. A new concept
(for example Cloudflare's "Code Mode") shows up first as a blog post, then as passing
mentions in newsletters and link roundups. Any single mention looks too thin to earn a
claim, so nothing ever notices that the idea is spreading. This module adds the two
signals the merge cannot compute for itself:

  * Emerging terms: phrases that are frequent in the recent archive, spread across several
    distinct sources, nearly absent from older archive, and absent from the wiki.
  * Watchlist: concepts Charlie has said must be covered (watchlist.txt). A watchlist entry
    whose aliases appear nowhere in the wiki is a coverage gap.

Everything here is pure (strings and Items in, plain data out) so it is cheap to test and
has no paths, network, or LLM dependency. ai_research.py owns the I/O: it feeds archived
items in, runs targeted searches for what comes out, and passes a hint to the merge.

Key design decisions:
  * No NLP library. "New" is defined against the archive itself: a term that the older
    archive barely contains but the last few weeks contain often is new to us, whatever it
    is. The one outside resource is the system word list (/usr/share/dict), used to tell a
    coined single word ("codemode") from an ordinary one ("announcing"). Coinages need less
    evidence than phrases; without the word list there is no coinage lane, and single words
    must meet the phrase rules (more items, two sources).
  * Keys ignore case, spaces, and hyphens, so "code mode", "Code-Mode", and "codemode"
    are one term.
  * Candidates are ranked by how many distinct SOURCES carry them, not raw count, because
    one chatty source repeating itself is not a trend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

# Words that may not begin or end a multi-word candidate ("of the", "how to" are noise).
STOPWORDS = frozenset("""
a about above after again all also an and any are as at be because been before being but by can
could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its just like may might more most my new no nor not now of
off on once one only or other our out over own same she should so some such than that the their
them then there these they this those through to too under until up use used using very was we
were what when where which while who why will with would you your via per vs get got make makes
made see says said say show shows way ways what's it's don't can't
""".split())

CONTRACTION_RE = re.compile(r"['\u2019](?:s|t|re|ve|ll|d|m)\b", re.I)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#]*(?:-[A-Za-z0-9]+)*")

# A candidate must clear all of these to be reported. Tuned against the live archive so
# that "codemode" (2 sources, a handful of items, near-zero history) qualifies while
# ordinary vocabulary does not.
MIN_WINDOW_ITEMS = 3
MIN_WINDOW_DAYS = 2      # spread over time, so one burst of reposts does not count as a trend
MAX_BASELINE_ITEMS = 2
MIN_KEY_LEN = 6
# A single token that is not an English word is a coinage or a name ("codemode", "langgraph"),
# which is far more informative than a common phrase, so it needs less evidence.
MIN_COINED_ITEMS = 2
MIN_PHRASE_SOURCES = 2   # ordinary phrases need two independent sources; coinages do not (with a word list)
# Words that come from our own item templates and scraped social posts, never from the
# subject matter. Any candidate containing one is noise.
BOILERPLATE = frozenset("""points comments matched topic discussion subreddits twitters retweet
impression count expanded reply replied url ainews http https www com""".split())
# A concept can be a trend inside ONE source for weeks before a second one picks it up
# (Latent Space was alone in mentioning "codemode" for a week), so a second source is not
# required for a coinage. That lane needs a word list (/usr/share/dict/words, present on
# macOS; on Linux install wamerican): without one, single words get phrase rules.
# Precision comes from the baseline test plus the LLM filter in ai_research.


class ItemLike(Protocol):
    """The slice of ai_research.Item that radar reads."""
    source: str
    title: str
    body: str
    item_id: str


@dataclass
class Candidate:
    """An emerging term and the evidence for it."""
    term: str            # display form, lowercase with spaces
    key: str             # normalized key (lowercase, no spaces or hyphens)
    items: int           # distinct window items containing it
    sources: int         # distinct window sources containing it
    in_titles: int       # window items whose title contains it
    baseline: int        # distinct older items containing it
    days: int = 0        # distinct archive days the window items span
    score: float = 0.0


_DICT_WORDS: set[str] | None = None


def _dictionary() -> set[str]:
    """Lowercase English words from the system word list, or an empty set if there is none.

    Used only to reject a single ordinary word ("announcing") posing as a coined term. It is
    optional: without the file, has_dictionary() is False and the coinage lane is off.
    """
    global _DICT_WORDS
    if _DICT_WORDS is None:
        _DICT_WORDS = set()
        for path in ("/usr/share/dict/words", "/usr/share/dict/american-english"):
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    _DICT_WORDS = {w.strip().lower() for w in fh}
                break
            except OSError:
                continue
    return _DICT_WORDS


def has_dictionary() -> bool:
    """True if a system word list was found, so coinages can be told from ordinary words."""
    return bool(_dictionary())


def is_word(w: str) -> bool:
    """True if `w` (lowercase) is an ordinary English word or a plain inflection of one."""
    words = _dictionary()
    if w in words:
        return True
    for suf in ("ing", "ed", "es", "s", "ly", "er", "ers", "ness", "ment", "ments", "ings"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)]
            if stem in words or stem + "e" in words or (len(stem) > 3 and stem[-1] == stem[-2] and stem[:-1] in words):
                return True
    return False


def collapse(text: str) -> str:
    """Lowercase and drop everything but letters and digits: the key space."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _ngrams(text: str) -> dict[str, set[str]]:
    """Candidate phrases in `text` as {key: every spelling seen}. One to three tokens, no edge stopwords."""
    text = CONTRACTION_RE.sub("", text)   # "doesn't" -> "doesn", not a fragment candidate
    toks = TOKEN_RE.findall(text)
    low = [t.lower() for t in toks]
    out: dict[str, set[str]] = {}
    n = len(toks)
    for i in range(n):
        for size in (1, 2, 3):
            j = i + size
            if j > n:
                break
            words = low[i:j]
            if words[0] in STOPWORDS or words[-1] in STOPWORDS or BOILERPLATE & set(words):
                continue
            key = collapse("".join(words))
            if len(key) < MIN_KEY_LEN or key.isdigit():
                continue
            if size == 1 and (len(words[0]) < 5 or is_word(words[0])):
                continue
            shown = " ".join(words).replace("-", " ")
            out.setdefault(key, set()).add(shown)   # every spelling, so neither order wins
    return out


def _doc_text(it: ItemLike, body_chars: int = 3000) -> str:
    """Text a term is searched in: the title (weighted by the caller) plus a body prefix."""
    return f"{it.title}\n{it.body[:body_chars]}"


def emerging_terms(window: Iterable[ItemLike], baseline: Iterable[ItemLike],
                   wiki_text: str, limit: int = 60) -> list[Candidate]:
    """Terms that are frequent and multi-source recently, rare before, and not in the wiki.

    Args:
        window: Recent items (for example the last 21 days of archive).
        baseline: Older items, used to reject ordinary vocabulary.
        wiki_text: The assembled wiki; a term already in it is not a gap.
        limit: Maximum candidates to return.

    Returns:
        Candidates, strongest first. Empty if the window is too small to show a trend.
    """
    window = list(window)
    if len(window) < MIN_WINDOW_ITEMS:
        return []

    per_item: list[tuple[ItemLike, dict[str, set[str]], set[str]]] = []
    seen_items: dict[str, set[int]] = {}
    seen_sources: dict[str, set[str]] = {}
    seen_days: dict[str, set[str]] = {}
    one_token_items: dict[str, set[int]] = {}   # items that write the key as one token ("codemode")
    spaced_items: dict[str, set[int]] = {}      # items that write it as words ("code mode")
    spaced: dict[str, set[str]] = {}            # every multi-word spelling ("open ai agents", "openai agents")
    one_token: dict[str, set[str]] = {}         # every one-token spelling
    camel_items: dict[str, set[int]] = {}
    for idx, it in enumerate(window):
        for tok in re.findall(r"#?[A-Za-z][A-Za-z0-9]*", _doc_text(it)):
            if tok.startswith("#") or re.search(r"[a-z][A-Z]", tok):
                camel_items.setdefault(collapse(tok), set()).add(idx)
        grams = _ngrams(_doc_text(it))
        title_keys = set(_ngrams(it.title))
        per_item.append((it, grams, title_keys))
        for key, forms in grams.items():
            seen_items.setdefault(key, set()).add(idx)
            seen_sources.setdefault(key, set()).add(it.source)
            seen_days.setdefault(key, set()).add(it.item_id[:10])
            for shown in forms:
                if " " in shown:
                    spaced.setdefault(key, set()).add(shown)
                    spaced_items.setdefault(key, set()).add(idx)
                else:
                    one_token.setdefault(key, set()).add(shown)
                    one_token_items.setdefault(key, set()).add(idx)

    # Without a word list every long ordinary word would look coined, so the lane is off.
    use_coined = has_dictionary()

    def coined(key: str) -> bool:
        """A key written as one token in enough items on its own, that no dictionary knows.

        Counting only the one-token items, not every item with the key, keeps a single
        "#PromptCaching" from moving the ordinary phrase "prompt caching" into the lane.
        """
        ones = one_token_items.get(key, set())
        # A lowercase one-token spelling ("codemode") is a deliberate coinage; a CamelCase
        # hashtag ("#PromptCaching") is a phrase written for a tag, so it alone does not count.
        return (use_coined and bool(ones - camel_items.get(key, set()))
                and len(seen_items[key]) >= MIN_COINED_ITEMS and not is_word(key))

    def display(key: str) -> str:
        """The spelling most items use: the spaced form wins only as a strict majority."""
        if len(spaced_items.get(key, ())) > len(one_token_items.get(key, ())):
            return min(spaced[key])
        return min(one_token[key])

    survivors = {k for k, ix in seen_items.items()
                 if len(seen_days[k]) >= MIN_WINDOW_DAYS
                 and (len(ix) >= MIN_COINED_ITEMS if coined(k)
                      else len(ix) >= MIN_WINDOW_ITEMS and len(seen_sources[k]) >= MIN_PHRASE_SOURCES)}
    if not survivors:
        return []

    # Baseline pass touches only the survivors, so the older archive is scanned once and
    # cheaply no matter how many distinct phrases the window contains.
    # Word-boundary regexes, not collapsed substrings: "code mode" must not match
    # "decode mode", which a collapsed-text search does.
    # Built from every spaced spelling seen: alias_pattern("code mode") also matches
    # "codemode" and "code-mode", but alias_pattern("codemode") matches neither spaced form,
    # and alias_pattern("openai agents") misses "open ai agents", so any one spelling alone
    # can make an established term read as brand new.
    pats = {k: re.compile("|".join(alias_pattern(f).pattern for f in sorted(spaced.get(k) or one_token[k])), re.I)
            for k in survivors}
    base_count = dict.fromkeys(survivors, 0)
    for it in baseline:
        text = _doc_text(it)
        # Any match collapses to its key, so a key absent from the collapsed text cannot
        # match; this skips nearly every regex without changing a single count.
        flat = collapse(text)
        for key, pat in pats.items():
            if key in flat and pat.search(text):
                base_count[key] += 1

    # The wiki test is deliberately looser than the baseline test: also compare spelling-blind
    # ("Code Mode" in the wiki hides the coinage "codemode"). Over-excluding is the safe error.
    wiki_flat = collapse(wiki_text)
    cands: list[Candidate] = []
    for key in survivors:
        if base_count[key] > MAX_BASELINE_ITEMS or pats[key].search(wiki_text) or key in wiki_flat:
            continue
        title_hits = sum(1 for _, _, tk in per_item if key in tk)
        c = Candidate(display(key), key, len(seen_items[key]), len(seen_sources[key]),
                      title_hits, base_count[key], len(seen_days[key]))
        c.score = c.sources * 3 + c.items + 2 * c.in_titles
        cands.append(c)

    cands.sort(key=lambda c: (-c.score, c.term))
    coined_keys = {c.key for c in cands if coined(c.key)}
    # Drop a phrase contained in a longer one that is nearly as common ("programmatic tool"
    # inside "programmatic tool calling"); the longer form is the better search query.
    kept: list[Candidate] = []
    for c in cands:
        if any(c.key != o.key and c.key in o.key and o.items >= 0.8 * c.items for o in cands):
            continue
        kept.append(c)
    # Coinages and ordinary phrases are ranked in separate lanes. Mixed together, the
    # phrase lane (higher counts, mostly noise) crowds a two-mention coinage like
    # "codemode" out of any fixed-size list, which is exactly the case this exists for.
    lane_coined = [c for c in kept if c.key in coined_keys][: (2 * limit) // 3]
    lane_phrase = [c for c in kept if c.key not in coined_keys][: limit - len(lane_coined)]
    return lane_coined + lane_phrase


# ── Watchlist ───────────────────────────────────────────────────────────────

def parse_watchlist(text: str) -> list[list[str]]:
    """Parse watchlist.txt into alias groups.

    Format: one concept per line, aliases separated by `|`, `#` starts a comment.
    The first alias is the canonical name used in reports and searches.

        code mode | codemode | code execution with mcp
    """
    entries = []
    for ln in text.splitlines():
        ln = ln.split("#", 1)[0].strip()
        aliases = [a.strip() for a in ln.split("|") if a.strip()]
        if aliases:
            entries.append(aliases)
    return entries


def alias_pattern(alias: str) -> re.Pattern[str]:
    """Regex matching an alias however it is spaced or hyphenated ("code mode", "code-mode", "codemode")."""
    parts = [re.escape(p) for p in re.split(r"[\s\-_]+", alias.strip()) if p]
    return re.compile(r"\b" + r"[\s\-_]*".join(parts) + r"\b", re.I)


def covered(wiki_text: str, aliases: list[str]) -> bool:
    """True if any alias of the concept appears in the wiki."""
    return any(alias_pattern(a).search(wiki_text) for a in aliases)


def missing_watch(wiki_text: str, entries: list[list[str]]) -> list[list[str]]:
    """Watchlist concepts with no mention anywhere in the wiki."""
    return [e for e in entries if not covered(wiki_text, e)]


def lost_watch(old_wiki: str, new_wiki: str, entries: list[list[str]]) -> list[list[str]]:
    """Concepts the old wiki covered that the new one no longer does (a merge regression)."""
    return [e for e in entries if covered(old_wiki, e) and not covered(new_wiki, e)]


def hint_text(missing: list[list[str]], emerging: list[Candidate]) -> str:
    """Prompt addendum telling the merge which concepts are known gaps. Empty if there are none."""
    if not missing and not emerging:
        return ""
    lines = ["## Coverage radar (hints, not instructions)",
             "The wiki is checked for gaps after every run. These concepts are NOT in the wiki yet. "
             "If any item above explains one, extract the practice (apply the normal inclusion "
             "test) and add it. A primary source (the vendor or author who introduced it) outranks "
             "a passing mention. If no item supports one, ignore it."]
    if missing:
        lines.append("Must-cover concepts: " + "; ".join(e[0] for e in missing))
    if emerging:
        lines.append("Terms recurring across several sources lately: " + "; ".join(
            f"{c.term} ({c.items} items, {c.sources} sources)" for c in emerging))
    return "\n".join(lines)
