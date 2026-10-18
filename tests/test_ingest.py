#!/usr/bin/env python3
"""Offline tests for ingest/ai_research.py (no network, no LLM calls).

Covers the parts of the pipeline that protect the wiki from bad merges and bad
archive writes: the shrink guard and its retry, archive de-duplication and
re-offering of unmerged items, the scoping of the automatic git commit, the
topic-file split, and the per-card date binding of the html_index scraper.
Every test runs against a throwaway directory; the real wiki is never touched.

Run:  python tests/test_ingest.py        (or: pytest tests/)
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Point every configurable location at a throwaway dir BEFORE importing the module, so a
# real .env (which may aim WIKI_DATA_DIR at a live wiki) can never be touched by a test.
# Environment variables take precedence over .env, so this always wins.
_ISOLATED = tempfile.mkdtemp(prefix="context-wiki-tests-")
for _name in ("WIKI_DATA_DIR", "WIKI_ARCHIVE_DIR", "WIKI_STATE_DIR"):
    os.environ[_name] = _ISOLATED
os.environ["WIKI_GIT_COMMIT"] = "auto"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingest"))
import ai_research as a  # noqa: E402

PATH_NAMES = ("WIKI_DIR", "INDEX_PATH", "ARCHIVE_DIR", "STATE_PATH", "REVIEWED_PATH", "TOPICS_DIR", "GIT_COMMIT")


@contextlib.contextmanager
def sandbox():
    """Point every module-level path at a fresh temp dir, and restore them after."""
    tmp = Path(tempfile.mkdtemp())
    saved = {n: getattr(a, n) for n in PATH_NAMES if hasattr(a, n)}
    a.WIKI_DIR = tmp
    a.INDEX_PATH = tmp / "INDEX.md"
    a.ARCHIVE_DIR = tmp / "archive"
    a.STATE_PATH = tmp / "ingest" / "state.json"
    a.REVIEWED_PATH = tmp / "ingest" / "backfill_reviewed.json"
    if "TOPICS_DIR" in saved:
        a.TOPICS_DIR = tmp / "topics"
    a.GIT_COMMIT = "auto"
    (tmp / "ingest").mkdir()
    try:
        yield tmp
    finally:
        for n, v in saved.items():
            setattr(a, n, v)


@contextlib.contextmanager
def quiet_git():
    """Make git usable in temp repos: fixed identity, and no global commit hooks."""
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "/dev/null",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def doc(claims: int, words_per_claim: int = 20, section: str = "Tooling & frameworks", tag: str = "ab") -> str:
    """A wiki document with `claims` bullets, each citing a distinct item id."""
    filler = " ".join(["word"] * words_per_claim)
    bullets = "\n".join(f"- {filler} - src, 2026-01-01 [2026-01-01-{tag}-{i:02d}]" for i in range(claims))
    return f"## {section}\n\n{bullets}\n"


def item(slug: str, url: str, title: str = "t") -> a.Item:
    return a.Item("src", slug, title, url, datetime.now(timezone.utc), "body text")


# ── shrink guard ────────────────────────────────────────────────────────────


def test_claim_floor_scales_with_the_cap():
    """A file under the cap needs 75% of its claims kept; an over-cap file is allowed to trim."""
    under = doc(47, words_per_claim=24)               # about 1,200 words, near the cap
    assert a.claim_floor(under) == 36                 # ceil(0.75 * 47)
    over = doc(66, words_per_claim=a.WORD_CAP // 66 * 2)   # about twice the cap
    assert a.claim_floor(over) < 0.75 * 66            # scaled down, so trimming toward the cap passes
    assert a.claim_floor(doc(5)) == 0                 # tiny files are exempt


def test_merge_checked_accepts_retries_and_rejects():
    """Good answer: 1 call. Bad then good: retry with the dropped claims listed. Bad twice: rejected."""
    old, good, bad = doc(47), doc(60), doc(20)
    calls: list[str] = []
    real = a.merge_index

    def stub(*answers):
        it = iter(answers)

        def f(index_md, items, creds, correction=""):
            calls.append(correction)
            return next(it)
        return f

    try:
        a.merge_index = stub(good); calls.clear()
        assert a.merge_checked(old, [], {}) == good and len(calls) == 1

        a.merge_index = stub(bad, good); calls.clear()
        assert a.merge_checked(old, [], {}) == good and len(calls) == 2
        assert "CORRECTION" in calls[1] and "kept only 20" in calls[1]

        a.merge_index = stub(bad, bad); calls.clear()
        try:
            a.merge_checked(old, [], {})
            raise AssertionError("expected MergeRejected")
        except a.MergeRejected:
            pass

        a.merge_index = stub(""); calls.clear()
        assert a.merge_checked(old, [], {}) == "" and len(calls) == 1   # empty is the caller's problem
    finally:
        a.merge_index = real


# ── archive ─────────────────────────────────────────────────────────────────


def test_write_archive_dedupes_within_a_batch_and_reoffers_unmerged():
    """One archived copy per URL; an archived-but-unmerged item is re-offered under its original id."""
    with sandbox() as tmp:
        A, B, N, A2 = item("hn", "https://x/a"), item("sw", "https://x/b"), item("mn", "", "a note"), item("mf", "https://x/a")
        a.write_archive([A, B, N, A2], "2026-10-01")
        assert (A.item_id, B.item_id, N.item_id) == ("2026-10-01-hn-01", "2026-10-01-sw-02", "2026-10-01-mn-03")
        assert A2.item_id == ""                                    # same URL twice in one batch

        a.INDEX_PATH.write_text(f"## S\n\n- claim [{A.item_id}]\n")    # A was merged, B never was
        A3, B3, C3 = item("hn", "https://x/a"), item("sw", "https://x/b"), item("ls", "https://x/c")
        a.write_archive([A3, B3, C3], "2026-10-02")
        assert A3.item_id == ""                                    # merged already: skipped
        assert B3.item_id == "2026-10-01-sw-02"                    # unmerged: re-offered, original id
        assert C3.item_id == "2026-10-02-ls-01"                    # new: archived
        ids = [i.item_id for i in a.read_archive()]
        assert len(ids) == len(set(ids)) == 4                      # no second copy of anything


# ── git commit scoping ──────────────────────────────────────────────────────


def _seed_repo(tmp: Path, ignore_data: bool) -> None:
    git(tmp, "init", "-q", "-b", "main")
    (tmp / "archive").mkdir(); (tmp / "archive" / "d.md").write_text("### x\n")
    (tmp / "INDEX.md").write_text("## S\n\n- c [2026-01-01-ab-01]\n")
    (tmp / "ingest" / "ai_research.py").write_text("print('v1')\n")
    (tmp / "other.txt").write_text("1\n")
    if ignore_data:
        (tmp / ".gitignore").write_text("INDEX.md\narchive/\ntopics/\ningest/state.json\ningest/backfill_reviewed.json\n")
        git(tmp, "add", ".gitignore", "ingest/ai_research.py", "other.txt")
    else:
        git(tmp, "add", "-A")
    git(tmp, "commit", "-q", "-m", "seed")


def test_commit_stages_only_data_files():
    """Code edits and unrelated staged files must not be swept into the automatic commit."""
    with sandbox() as tmp, quiet_git():
        _seed_repo(tmp, ignore_data=False)
        (tmp / "INDEX.md").write_text("## S\n\n- c [2026-01-01-ab-01]\n- d [2026-01-01-ab-02]\n")
        (tmp / "ingest" / "ai_research.py").write_text("print('v2 edited')\n")      # unstaged code edit
        (tmp / "other.txt").write_text("2\n"); git(tmp, "add", "other.txt")          # unrelated staged file
        a.commit("2026-10-01", 2)
        files = git(tmp, "show", "--name-only", "--format=", "HEAD").split()
        assert files == ["INDEX.md"], files
        assert "other.txt" in git(tmp, "diff", "--cached", "--name-only")           # still staged
        assert "ai_research.py" in git(tmp, "diff", "--name-only")                  # still just modified


def test_commit_skips_when_data_is_gitignored_or_not_a_repo():
    """The published code repo ignores the data; a plain directory is not a repo. Both are no-ops."""
    with sandbox() as tmp, quiet_git():
        _seed_repo(tmp, ignore_data=True)
        before = git(tmp, "rev-parse", "HEAD")
        (tmp / "INDEX.md").write_text("changed\n")
        a.commit("2026-10-01", 1)
        assert git(tmp, "rev-parse", "HEAD") == before
    with sandbox() as tmp, quiet_git():
        (tmp / "INDEX.md").write_text("x\n")
        a.commit("2026-10-01", 1)          # must not raise


# ── topic-file split ────────────────────────────────────────────────────────

BIG = "Agent architecture & harness design"     # a TOPIC_META title, so it gets a fixed slug
SMALL = "Retrieval & RAG"


@contextlib.contextmanager
def split_at(words: int):
    old = a.SECTION_SPLIT_WORDS
    a.SECTION_SPLIT_WORDS = words
    try:
        yield
    finally:
        a.SECTION_SPLIT_WORDS = old


def ids(md: str) -> list[str]:
    return sorted(a.ID_RE.findall(md))


def two_sections(big_claims: int = 10, small_claims: int = 2) -> str:
    return doc(big_claims, 10, BIG, "ab") + "\n" + doc(small_claims, 10, SMALL, "cd")


def test_only_the_oversized_section_moves_to_a_topic_file():
    """The big section becomes a stub pointing at topics/<slug>.md; the small one stays inline."""
    with sandbox() as tmp, split_at(60):
        md = two_sections()
        a.write_wiki(md)
        index = a.INDEX_PATH.read_text()
        assert "<!-- topic-file: topics/agent-architecture.md -->" in index
        assert "10 claims, updated 2026-01-01" in index and "Covers:" in index
        assert "## Retrieval & RAG" in index and "2026-01-01-cd-00" in index      # small section stays inline
        assert "2026-01-01-ab-00" not in index                                     # big section's claims do not
        assert "2026-01-01-ab-09" in (tmp / "topics" / "agent-architecture.md").read_text()


def test_assembled_wiki_is_identical_to_what_was_written():
    """read_wiki() must give back every claim, so the merge never notices where sections live."""
    with sandbox(), split_at(60):
        md = two_sections()
        a.write_wiki(md)
        got = a.read_wiki()
        assert ids(got) == ids(md) and a.count_claims(got) == a.count_claims(md) == 12
        assert [t for t, _ in a.parse_sections(got)[1]] == [BIG, SMALL]           # canonical section order


def test_a_split_section_stays_split_until_it_shrinks_a_lot():
    """Hysteresis: between half and the full threshold it stays split; below half it comes back inline."""
    with sandbox() as tmp, split_at(60):
        a.write_wiki(two_sections(10))
        assert (tmp / "topics" / "agent-architecture.md").exists()
        a.write_wiki(two_sections(3))                    # about 45 words: under 60 but over 30
        assert (tmp / "topics" / "agent-architecture.md").exists()
        a.write_wiki(two_sections(1))                    # about 15 words: under 30
        assert not (tmp / "topics").exists()             # file and now-empty directory both removed
        assert "2026-01-01-ab-00" in a.INDEX_PATH.read_text()


def test_a_legacy_single_file_wiki_migrates():
    """INDEX.md written by the old code (everything inline) reads fine and splits on the next write."""
    with sandbox() as tmp, split_at(60):
        md = two_sections()
        a.INDEX_PATH.write_text(md)
        assert ids(a.read_wiki()) == ids(md)
        a.write_wiki(a.read_wiki())
        assert (tmp / "topics" / "agent-architecture.md").exists()
        assert ids(a.read_wiki()) == ids(md)


def test_a_missing_topic_file_is_an_error_not_an_empty_section():
    """If the file a stub points at is gone, the merge must not proceed as if the section were empty."""
    with sandbox() as tmp, split_at(60):
        a.write_wiki(two_sections())
        (tmp / "topics" / "agent-architecture.md").unlink()
        try:
            a.read_wiki()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "agent-architecture.md" in str(exc)


def test_an_unknown_section_title_gets_a_slug_and_round_trips():
    with sandbox() as tmp, split_at(60):
        md = doc(10, 10, "Weird & New Topic!", "ef")
        a.write_wiki(md)
        assert (tmp / "topics" / "weird-new-topic.md").exists()
        assert ids(a.read_wiki()) == ids(md)


def test_ids_inside_topic_files_count_as_merged():
    """merged_ids() drives re-offering of unmerged items, so it must look inside topic files too."""
    with sandbox(), split_at(60):
        a.write_wiki(two_sections())
        found = a.merged_ids()
        assert "2026-01-01-ab-07" in found and "2026-01-01-cd-01" in found


def test_commit_includes_topic_files_and_their_deletion():
    with sandbox() as tmp, quiet_git(), split_at(60):
        _seed_repo(tmp, ignore_data=False)
        a.write_wiki(two_sections()); a.commit("2026-10-01", 1)
        assert "topics/agent-architecture.md" in git(tmp, "show", "--name-only", "--format=", "HEAD")
        a.write_wiki(two_sections(1)); a.commit("2026-10-02", 1)                  # un-split: file deleted
        assert "D\ttopics/agent-architecture.md" in git(tmp, "show", "--name-status", "--format=", "HEAD")


# ── html index dates ────────────────────────────────────────────────────────


def _card(path: str, title: str, date: str) -> str:
    """One newest-first index card as the Anthropic engineering page lays it out:
    image link, title link, teaser, then the date. Both links point at the same article."""
    teaser = "A teaser paragraph about the article. " * 4
    return (f'<article><a href="{path}"><img alt=""></a>'
            f'<a href="{path}"><h3>{title}</h3></a><p>{teaser}</p>'
            f'<span>{date}</span></article>\n')


def test_html_index_dates_bind_to_their_own_card():
    """Each article gets the date from its own card, not the previous (newer) card's date."""
    cards = [("/engineering/newest", "Newest", "Apr 23, 2026", "2026-04-23"),
             ("/engineering/middle", "Middle", "Jan 9, 2026", "2026-01-09"),
             ("/engineering/oldest", "Oldest", "Sep 19, 2024", "2024-09-19")]
    index_html = "<html><body><nav>Engineering</nav>\n" + "".join(
        _card(p, t, d) for p, t, d, _ in cards) + "<footer>Copyright 2026</footer></body></html>"
    prose = "<p>" + ("This sentence is long enough to count as article prose rather than navigation chrome. " * 3) + "</p>"
    article_html = "<html><body><h1>{title}</h1>" + prose * 3 + "</body></html>"

    def fake_get(url: str) -> bytes:
        if url == "https://www.anthropic.com/engineering":
            return index_html.encode()
        return article_html.format(title=url.rsplit("/", 1)[-1].title()).encode()

    # index_card_dates alone: the pure binding step
    got = a.index_card_dates(index_html)
    assert {p: d.date().isoformat() for p, d in got.items()} == {p: iso for p, _, _, iso in cards}

    # and through the fetcher, with the network stubbed and an empty archive
    real_get, real_archived = a.http_get, a.archived_urls
    a.http_get, a.archived_urls = fake_get, lambda: set()
    try:
        with sandbox():
            items = a.fetch_html_index("anthropic-eng", "ae", "https://www.anthropic.com/engineering",
                                       datetime(2000, 1, 1, tzinfo=timezone.utc))
    finally:
        a.http_get, a.archived_urls = real_get, real_archived
    by_url = {i.url: i for i in items}
    assert len(by_url) == 3
    for p, title, _, iso in cards:
        it = by_url["https://www.anthropic.com" + p]
        assert it.published.date().isoformat() == iso, (p, it.published)
        assert it.title == title


# ── settings and locations ──────────────────────────────────────────────────


def test_settings_come_from_env_then_dotenv_then_default():
    """Real environment variables win over the repo .env, which wins over the built-in default."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / ".env").write_text('# comment\nWIKI_X_ONE=from-file\nWIKI_X_TWO="quoted"\nKEY=a#b\n')
    saved = a._LOCAL_SETTINGS
    try:
        a._LOCAL_SETTINGS = a._read_env_file(tmp / ".env")
        os.environ["WIKI_X_ONE"] = "from-env"
        assert a.setting("WIKI_X_ONE", "dflt") == "from-env"
        del os.environ["WIKI_X_ONE"]
        assert a.setting("WIKI_X_ONE", "dflt") == "from-file"
        assert a.setting("WIKI_X_TWO", "dflt") == "quoted"           # one pair of quotes stripped
        assert a.setting("KEY", "dflt") == "a#b"                     # inline # is part of the value
        assert a.setting("WIKI_X_MISSING", "dflt") == "dflt"
        assert a._read_env_file(tmp / "absent.env") == {}
    finally:
        a._LOCAL_SETTINGS = saved
        os.environ.pop("WIKI_X_ONE", None)


def test_index_md_gets_a_generated_header_that_reading_ignores():
    """INDEX.md warns human editors, and the header never leaks into what the merge sees."""
    with sandbox(), split_at(10_000):
        md = doc(3)
        a.write_wiki(md)
        assert a.INDEX_PATH.read_text().startswith("<!-- Generated by context-wiki")
        assert "<!--" not in a.read_wiki() and ids(a.read_wiki()) == ids(md)


def test_commit_can_be_switched_off():
    """WIKI_GIT_COMMIT=0 (a vault whose own sync job commits) means no commit at all."""
    with sandbox() as tmp, quiet_git():
        _seed_repo(tmp, ignore_data=False)
        before = git(tmp, "rev-parse", "HEAD")
        (tmp / "INDEX.md").write_text("## S\n\n- changed [2026-01-01-ab-09]\n")
        a.GIT_COMMIT = "0"
        a.commit("2026-10-01", 1)
        assert git(tmp, "rev-parse", "HEAD") == before


def test_commit_skips_data_that_lives_outside_the_wiki_repo():
    """Archive and state kept elsewhere (the default) are left out instead of failing the commit."""
    with sandbox() as tmp, quiet_git():
        _seed_repo(tmp, ignore_data=False)
        outside = Path(tempfile.mkdtemp())
        a.ARCHIVE_DIR = outside / "archive"; a.ARCHIVE_DIR.mkdir()
        (a.ARCHIVE_DIR / "x.md").write_text("### y\n")
        (tmp / "INDEX.md").write_text("## S\n\n- c [2026-01-01-ab-01]\n- e [2026-01-01-ab-03]\n")
        a.commit("2026-10-01", 1)
        assert git(tmp, "show", "--name-only", "--format=", "HEAD").split() == ["INDEX.md"]


# ── URL safety (http_get / --add-url) ───────────────────────────────────────
# No test here touches the network: refused URLs fail before any request, literal
# IPs resolve without DNS, and named hosts use a stubbed getaddrinfo.


def refused(url: str, allow_private: bool = True) -> bool:
    """True if check_url raises UnsafeURLError for this URL."""
    try:
        a.check_url(url, allow_private)
        return False
    except a.UnsafeURLError:
        return True


@contextlib.contextmanager
def fake_dns(*addresses: str):
    """Make socket.getaddrinfo return the given addresses for any host."""
    real = a.socket.getaddrinfo

    def stub(host, port, *args, **kwargs):
        return [(a.socket.AF_INET6 if ":" in ip else a.socket.AF_INET, a.socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in addresses]
    a.socket.getaddrinfo = stub
    try:
        yield
    finally:
        a.socket.getaddrinfo = real


def test_only_http_and_https_urls_are_fetched():
    """file://, ftp://, data: and scheme-less URLs are refused; http(s) to the built-in sources passes."""
    for url in ("file:///Users/x/ai/ai-router/.env", "FILE:///etc/passwd", "ftp://example.com/x",
                "data:text/plain,hi", "/etc/passwd", "https:///no-host"):
        assert refused(url), url
    assert not refused("https://www.anthropic.com/engineering")
    assert not refused("http://example.com/feed")


def test_add_url_refuses_loopback_private_and_link_local_ips():
    """Literal non-public addresses are refused for user URLs but still allowed for built-in sources."""
    for url in ("http://127.0.0.1:4000/v1/models", "http://10.0.0.5/", "http://192.168.1.1/",
                "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
                "http://0.0.0.0/", "http://100.64.0.1/"):
        assert refused(url, allow_private=False), url
        assert not refused(url, allow_private=True), url
    assert not refused("https://8.8.8.8/", allow_private=False)        # public literal passes


def test_add_url_refuses_a_name_with_any_private_address():
    """Every resolved address is checked: one private answer among public ones is enough to refuse."""
    with fake_dns("93.184.216.34", "10.0.0.5"):
        assert refused("https://example.com/post", allow_private=False)
    with fake_dns("127.0.0.1"):
        assert refused("http://localhost:4000/", allow_private=False)
    with fake_dns("93.184.216.34", "2606:2800:220:1::1"):
        assert not refused("https://example.com/post", allow_private=False)


def test_add_url_refuses_a_host_that_does_not_resolve():
    """A resolution failure is a refusal, not an uncaught socket error."""
    real = a.socket.getaddrinfo

    def fail(*args, **kwargs):
        raise a.socket.gaierror("nodename nor servname provided")
    a.socket.getaddrinfo = fail
    try:
        assert refused("https://no-such-host.invalid/", allow_private=False)
    finally:
        a.socket.getaddrinfo = real


def test_redirect_targets_are_checked():
    """A redirect to file://, ftp://, or (for user URLs) a private host is refused before it is followed."""
    req = a.urllib.request.Request("https://example.com/")
    strict, lenient = a._CheckedRedirectHandler(False), a._CheckedRedirectHandler(True)
    for handler, target in ((lenient, "file:///etc/passwd"), (lenient, "ftp://example.com/x"),
                            (strict, "http://127.0.0.1:4000/v1"), (strict, "http://169.254.169.254/")):
        try:
            handler.redirect_request(req, None, 302, "Found", {}, target)
            raise AssertionError(f"expected UnsafeURLError for {target}")
        except a.UnsafeURLError:
            pass
    assert lenient.redirect_request(req, None, 302, "Found", {}, "https://example.com/b") is not None


def test_http_get_refuses_before_any_request_and_caps_the_body():
    """A refused URL never builds an opener; an oversized body is cut to MAX_FETCH_BYTES."""
    opened: list[str] = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return b"x" * n

    class Opener:
        def open(self, req, timeout=None):
            opened.append(req.full_url)
            return Resp()

    real_build, real_cap = a.urllib.request.build_opener, a.MAX_FETCH_BYTES
    a.urllib.request.build_opener = lambda *handlers: Opener()
    a.MAX_FETCH_BYTES = 10
    try:
        try:
            a.http_get("file:///etc/passwd")
            raise AssertionError("expected UnsafeURLError")
        except a.UnsafeURLError:
            pass
        assert opened == []
        assert a.http_get("https://example.com/feed") == b"x" * 10
        assert opened == ["https://example.com/feed"]
    finally:
        a.urllib.request.build_opener, a.MAX_FETCH_BYTES = real_build, real_cap


def test_add_url_refusal_is_a_clean_error_and_archives_nothing():
    """--add-url file://... returns exit code 1 and writes no archive file."""
    with sandbox():
        assert a.add_manual("file:///Users/x/ai/ai-router/.env", None, None, None, {}) == 1
        assert not a.ARCHIVE_DIR.exists() or not any(a.ARCHIVE_DIR.iterdir())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
