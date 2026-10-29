# context-wiki

A small pipeline that keeps a maintained **build manual for AI engineering** and hands it to your coding agents, so they stop re-researching the same ground every session.

Every day it pulls new posts from practitioner feeds, archives each one in full, and makes **one LLM call** that merges what matters into a topic-organized markdown wiki. Agents read that wiki (a few hundred to a couple of thousand tokens) instead of paying to search, fetch, and synthesize again.

The idea: do the expensive work **once, at write time**, and make reading cheap.

> This repository contains the code and the scheduled job only. The wiki it produces (`INDEX.md`, `topics/`, `archive/`) is generated on your machine, stays there, and is gitignored. The archive holds full text of other people's articles, so it should not be published.

## How it works

```
sources: ~20 RSS/Atom feeds (practitioners + vendor engineering blogs), Hacker News,
         HF papers, Anthropic engineering blog
   |   date watermark per source: only items newer than the last successful pull
   |   + coverage radar: targeted searches for concepts the wiki is missing (see below)
   |   + thin items get the linked page's text fetched
   v
archive/YYYY-MM-DD.md     every item in full, with a stable id like 2026-10-07-sw-03   (append-only)
   |
   v   LLM merges, 15 items per call, each call against the previous result
   |   guards: shrink guard, watchlist guard, output guard (+ retries), then prune to the word cap
   v
INDEX.md  +  topics/<topic>.md        what agents read
```

Design decisions worth knowing:

- **Merge, never append.** Each run rewrites the whole wiki from the current wiki plus new items, so overlapping findings are merged into one claim that cites every source, and superseded claims are replaced. A word cap forces the model to decide what earns its place.
- **A build manual, not a digest.** The merge prompt only admits a claim if an engineer could act on it while writing code. Funding news, launches, benchmarks-as-marketing, and paper abstracts with no reproducible technique are rejected.
- **Content-based pruning.** Nothing expires on a timer. A claim leaves only when it is superseded, absorbed into another claim, or no longer decision-relevant.
- **Greppable ids.** Every claim ends with an attribution (source, date, and item ids in brackets). `grep -rn "^### <item-id>$" archive/` returns the stored article and its URL, so follow-up questions need no network call.
- **Shrink guard.** A merge that keeps under 75% of the existing claims is retried once with the dropped claims listed, then rejected, leaving the wiki untouched. A model can ignore "the word cap is a ceiling, not a target"; code cannot. The floor is set once per run (once per `--backfill`) from the wiki the run started with, and every merge chunk and the prune that follows must keep it, so small cuts cannot add up across chunks or passes.
- **Watchlist guard.** `watchlist.txt` lists concepts that must be covered (one per line, aliases separated by `|`). Once the wiki mentions a concept, a merge that removes every mention is retried and then rejected, so a technique cannot be pruned away by accident.
- **Output guard.** The wiki is read as trusted context by every agent session, so a merge result that adds hidden text (an HTML comment, a `<!...>` or `<?...?>` form, or a Markdown link reference definition such as `[//]: # (...)`: invisible when rendered, read in full by agents), an HTML tag that loads, runs, or hides something (`<img>`, `<script>`, `<details>`, `<template>`, an `on*=`/`src=`/`style=`/`hidden` attribute, also after a slash as in `<img/src=x>`), a Markdown image, or a list item or paragraph that cites no item id is retried once and then rejected. Backticked code (an escaped backtick opens no code span), XML-style prompt tags such as `<document>`, and an attribute-less opening tag whose name is capitalised or one letter (a generic or component like `List<Object>` or `<Button>`) pass, unless the element hides its content. The retry names each finding by kind and position and never quotes it back. This is a cheap shape filter, not injection detection: an instruction written as a normal cited claim passes. Findings already in the stored wiki do not block merges, and a pre-existing id-less claim the merge lightly rewords (no longer, at most three new words or symbols) replaces it one for one; lint reports them.
- **Chunked merging.** Items go to the model 15 at a time, each chunk merging into the result of the last. One 60-item prompt made the model either rewrite the wiki wholesale (tripping the guards) or ignore the new items. A chunk the guards reject is skipped, and the sources of its items keep their old watermark so the next run refetches them (the Anthropic corpus, which has no watermark, refetches any archived article no merge has seen yet), ranks them ahead of new items for their source's quota, and merges up to four of them alone, one per call, before the usual chunks. The guards reject a whole chunk, so only a guard refusal of an item merged alone counts against it (a connection error or timeout does not): one refused alone twice is given up on (`ingest/merge_retry.json`) and left to `--backfill`, so one poisoned item cannot sink a chunk every day and its chunk-mates are not given up with it.
- **Pruning by deletion.** The word cap is enforced in code. When the wiki is over it, the model is shown the numbered claims and returns only the numbers to delete; the code deletes them, so every surviving claim is byte-for-byte unchanged, and never deletes the sole claim covering a watchlist concept. (Asking the model to retype a shorter wiki cut whole topics at random.)
- **Topic files.** A section over 500 words moves to `topics/<slug>.md` and `INDEX.md` keeps a one-line pointer (claim count, last updated, scope). Small sections stay inline. This is purely storage: the merge always sees one assembled document.
- **Lint.** Reports duplicate item ids, ids the model invented, cap overruns, single-source claims whose date differs from the archive's, claims built on an item that was archived without a body, claims that lead with a research-artifact name instead of a practice, and bullets containing a URL, a pipe into a shell (`| sh`, `| bash`), HTML, or no item id. The archive is the ground truth for ids and dates. Reported, never auto-fixed.
- **Untrusted input.** Feed titles and bodies are wrapped in an explicit "data, not instructions" block before they reach the model.

## Requirements

- Python 3.10 or newer, and the `openai` package
- An OpenAI-compatible chat endpoint and a model alias that is good at editing a long document (for example a LiteLLM proxy, or any provider with a compatible API)
- macOS for the bundled scheduler template; any cron-style scheduler works elsewhere

## Quick start

```bash
git clone https://github.com/CharlieTolleson/context-wiki.git
cd context-wiki
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# credentials for your endpoint (read from .env, or from real environment variables)
cat > .env <<'ENV'
LITELLM_BASE_URL=http://localhost:4000/v1
LITELLM_MASTER_KEY=your-key
ENV
export WIKI_MODEL=your-model-alias        # default: claude-sonnet-5

.venv/bin/python ingest/ai_research.py --dry-run   # fetch and list what is new; no credentials needed
.venv/bin/python ingest/ai_research.py             # first real run
```

The variable names say LiteLLM, but any OpenAI-compatible base URL and key work.

The first run looks back 14 days (`LOOKBACK_CAP_DAYS`) and writes `INDEX.md`, `archive/`, and, once a section is large enough, `topics/` in the repo root (all gitignored). To write the wiki somewhere else, see [Keeping the wiki in a notes vault](#keeping-the-wiki-in-a-notes-vault).

## Commands

| Command | What it does |
|---|---|
| `ai_research.py` | Normal daily run: fetch, archive, merge, store |
| `--dry-run` | Fetch and print what is new. No LLM call, no writes, watermarks unchanged |
| `--since YYYY-MM-DD` | Override every source's watermark (manual backfill of fetching) |
| `--backfill` | Merge already-archived items that never reached the wiki, in batches (capped at 8 passes) |
| `--backfill --since YYYY-MM-DD` | Same, limited to archive items from that run date onward |
| `--add-url URL [--source NAME]` | Fetch one page by hand and run it through the same archive and merge path. Only public `http`/`https` URLs are accepted: `file://` and hosts that resolve to loopback, private, or link-local addresses (including IPv6 forms such as NAT64 that embed one) are refused, checked again at connect time so a host cannot rebind its DNS between the check and the request. Only HTML or plain-text pages are fetched, and the whole request, redirects included, must finish within two minutes |
| `--add-note TEXT --title TITLE [--source NAME]` | Store a finding that has no URL (something you learned yourself) |
| `--radar` | Print the coverage report (watchlist gaps, emerging terms, recent removals) and write `RECALL.md` |
| `--prune` | Bring an over-cap wiki back under the word cap by model-selected deletion |
| `--rebuild-index` | Re-store the wiki through the topic-file layout. No fetching, no LLM call |

Manual adds follow exactly the same rules as the scheduled job: the same inclusion test, merge rules, word cap, and attribution format. Do not edit `INDEX.md` or `topics/` by hand; the next merge will overwrite your edit.

## Keeping recall high: the coverage radar

A merge that sees each batch alone cannot notice that an idea is spreading, or that its originating blog is not a source. The radar covers that, using three cheap signals. It is advisory, so a failure in it never stops a run.

- **Watchlist.** A concept in `watchlist.txt` that the wiki does not mention is searched for on Hacker News at any date every run (relevance-ranked, so the originating post is found even months later), and the merge is told it is a known gap.
- **Emerging terms.** Phrases and coined words that recur in the last 21 days of archive, were rare before, and are not in the wiki. A small model (`WIKI_RADAR_MODEL`) keeps the ones that name techniques, tools, or protocols; those are searched for too. A one-word coinage ("codemode") needs less evidence than a phrase, which relies on a word list at `/usr/share/dict/words` (present on macOS; on Debian or Ubuntu install `wamerican`). Without one, single words get the stricter phrase rules and the log says so.
- **Recall report.** Every claim a merge removes is appended to `dropped_claims.jsonl`, and each run writes `RECALL.md` (watchlist gaps, emerging terms, recent removals) and one `RECALL ...` line to the log. `ai_research.py --radar` prints it on demand.

The example that motivated it: Cloudflare's "Code Mode" (agents write code against MCP tools instead of calling them one by one) was ingested once, pruned a day later, never re-found because the originating blog was not a source, and mentioned only in passing afterwards.

## Configuration

Each setting can be an environment variable or a `KEY=VALUE` line in `.env` at the repo root (gitignored; the environment wins):

| Setting | Default | Purpose |
|---|---|---|
| `WIKI_DATA_DIR` | repo root | Where the readable wiki is written: `INDEX.md` and `topics/` |
| `WIKI_ARCHIVE_DIR` | `archive/` in the repo | The raw archive of fetched articles |
| `WIKI_STATE_DIR` | `ingest/` in the repo | Watermarks (`state.json`), the backfill ledger, and the run lock |
| `WIKI_MODEL` | `claude-sonnet-5` | Model alias used for the merge and the prune |
| `WIKI_RADAR_MODEL` | `cheap` | Small model alias that filters emerging-term candidates (never writes the wiki) |
| `WIKI_WATCHLIST` | `watchlist.txt` in the repo | Concepts that must be covered |
| `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY` | none | Endpoint and key |
| `WIKI_ENV_FILE` | the repo `.env` | Optional separate file to read the two endpoint values from, so secrets can stay in one shared place |
| `WIKI_GIT_COMMIT` | `auto` | `auto` commits the wiki after a run when its folder is a git repo and the data is not ignored; `0` never commits |
| `WIKI_SECTION_SPLIT_WORDS` | `500` | Section size that moves it into its own topic file |

Constants at the top of `ingest/ai_research.py`:

| Constant | Default | Meaning |
|---|---|---|
| `SOURCES` | 23 sources | Add a tuple `(name, two-letter-id, kind, url)`. Kinds: `feed` (RSS or Atom), `hn`, `hf_papers` (new by listing day, since a day's list carries papers published weeks earlier), `html_index`. The id must be exactly two lowercase letters |
| `SOURCE_KEYWORDS` | 9 sources | General engineering blogs whose items are kept only when the title or opening matches `RELEVANCE_RE` |
| `SOURCE_QUOTA` | papers 3, HN 10 | Max items a high-volume source may put in one merge batch, however small the batch. The rest stay archived |
| `WORD_CAP` | `2000` | Hard cap on the whole wiki. The merge prompt, lint, and shrink guard all follow it |
| `LOOKBACK_CAP_DAYS` | `14` | Longest backfill after an outage |
| `TOPIC_META` | 7 sections | Section titles, file slugs, and one-line scopes. Sections the model invents are handled too |
| `MERGE_CHUNK` | `15` | Items per merge call |
| `ENRICH_LIMIT`, `ENRICH_MIN_BODY` | 40, 500 | Thin items (body under 500 characters) get the linked page fetched, up to 40 per run |
| `HN_QUERIES`, `HN_MIN_POINTS` | 16 topics, 75 | Hacker News is queried by topic, not by front page |

The merge prompt is the `MERGE_SYSTEM` string. Tune it for your own definition of "worth keeping". Keep its few-shot examples fictional (they use made-up "Acme" names and say so): a real artifact name in an example tends to come back as a wiki claim, cited against whatever item looked closest.

## Keeping the wiki in a notes vault

The readable wiki and the raw archive have separate settings so the wiki can live where you already keep notes (an Obsidian vault, for example) while the archive stays local:

```
# .env
WIKI_DATA_DIR=~/notes/Resources/Wiki     # INDEX.md and topics/ land here
WIKI_GIT_COMMIT=0                        # if the vault already has its own sync job that commits
WIKI_ENV_FILE=~/path/to/shared/.env      # optional: read LITELLM_* from an existing secrets file
```

Leave `WIKI_ARCHIVE_DIR` out of anything that syncs, indexes, or mirrors the vault: it holds full text of other people's articles, it grows every day, and an embedding pipeline would bury your own notes under it. The default (`archive/` in this repo, gitignored) is a good place.

Do not let two things commit the same folder. If the vault has its own sync job, set `WIKI_GIT_COMMIT=0` so its history becomes the wiki's undo log.

## Scheduling

**macOS (launchd).** A template is in `launchd/`:

```bash
mkdir -p logs
sed "s#__REPO_DIR__#$PWD#g" launchd/com.example.context-wiki.plist.template \
  > ~/Library/LaunchAgents/com.example.context-wiki.plist
launchctl load ~/Library/LaunchAgents/com.example.context-wiki.plist
launchctl start com.example.context-wiki      # fire once now to test
```

**cron or anything else.** Run the script once a day from the repo root:

```
0 7 * * * cd /path/to/context-wiki && .venv/bin/python ingest/ai_research.py >> logs/context-wiki.log 2>&1
```

Things learned the hard way:

- **Sleeping laptops.** A job that fires while the Mac is asleep starts in a short maintenance wake and is put back to sleep mid-run, which shows up as socket timeouts and an LLM call that finishes hours late. The template wraps the job in `caffeinate -i` for this reason.
- **One run at a time.** Every mode that writes takes a lock (`run.lock` in `WIKI_STATE_DIR`), so a manual `--add-url` started during the scheduled job waits for it to finish instead of interleaving with it. A run gives up with an error after two hours (`LOCK_WAIT_SECONDS`), and the log names the PID holding the lock. Files are replaced atomically, so a killed run never leaves a half-written `INDEX.md`; the temp files start with `.ctxwiki.` and the next run removes any a killed run left behind. If a vault sync job commits the wiki folder, add `.ctxwiki.*` to the vault's `.gitignore` so one is never committed mid-run.
- **Missed days heal themselves.** Each source keeps its own watermark and only advances it on success, so a skipped day or a failed source is picked up on the next run (up to `LOOKBACK_CAP_DAYS`).
- **launchd and time zones.** launchd's calendar scheduler can keep using the time zone it started with. After travelling across zones, a job may fire hours off schedule until you reboot or log out and back in.

## Using the wiki from an agent

Point your agent's standing instructions (for example a `CLAUDE.md`) at the wiki, and say *when* to read it. A bare "this file exists" does not beat an agent's habit of reaching for web search; naming the trigger topics does.

```markdown
## Context wiki

`<WIKI_DATA_DIR>/INDEX.md` is a maintained knowledge base of current engineering
findings, refreshed daily. Read it before researching AI systems, agent architecture,
retrieval, context engineering, evals, or LLM tooling from scratch, and before reaching
for web search on those topics.

A large section lives in `topics/<topic>.md`; INDEX.md then shows a one-line pointer for it.
Open that file when the topic fits your task. Each claim ends with a source, a date, and
item ids in brackets. For more detail, `grep -rn "^### <item-id>$" <WIKI_ARCHIVE_DIR>`
returns the stored article and its URL. Fetch the URL only if the stored body is thin.
```

The read path is deliberately tiered so most questions cost few tokens: `INDEX.md`, then a topic file if a pointer matches, then `grep` an item id in `archive/`, and only then the live URL.

## Data and git

The data files are gitignored here. After a run the script tries to commit the wiki so a bad merge is one `git revert` away. It quietly skips when `WIKI_GIT_COMMIT=0`, when the data is ignored, or when the folder is not a repository. Only the data files are ever staged, never code or anything else you have staged, and only those inside the wiki folder's own repo.

To keep that history, point `WIKI_DATA_DIR` at a folder in a **private** repo (or a vault that syncs to one).

## Tests

```bash
.venv/bin/python tests/test_ingest.py     # offline: no network, no LLM calls
```

They cover the shrink guard and its retry, the output guard (HTML and id-less bullets), the radar and watchlist guard, chunked merging, deletion-based pruning, feed repair and page enrichment, archive de-duplication and re-offering of unmerged items, per-source quotas on small batches, the papers fetcher, the lint's checks against the archive (fabricated ids, changed dates, bodiless sources, artifact names), commit scoping and the commit switch, settings precedence, the topic-file split (including a round trip, the missing-file safety check, and the rejection of pointers that lead outside `topics/`), URL safety for fetches, and the date binding of the `html_index` scraper (each article takes the date from its own card), and the merge prompt's examples (fictional names, marked as illustrations). The suite points every location at a temp directory before importing, so it cannot touch a real wiki.

## Known limitations

- **Page enrichment is best effort.** Thin items get the linked page's text, but paywalled, JavaScript-rendered, or blocking sites leave the original teaser, and at most 40 items are fetched per run.
- **Some sites block scripted fetches.** `openai.com/blog` returns 403 to plain HTTP clients, and its working news feed is mostly customer stories, so OpenAI is not a source. The Anthropic news page has no feed either.
- **The emerging-term detector is statistical and modest.** It needs a term to appear in two or three archive items on different days, so a concept mentioned once is invisible to it; the watchlist and the vendor-blog sources are the main defence, and the detector is the net under them.
- **Merge quality depends on the model.** Lint reports the failures a model keeps making (duplicate ids, artifact-name bullets, a citation date the model changed); it does not fix them. The artifact-name check keeps a small allowlist (`ARTIFACT_ALLOWLIST`) for products and generic terms that legitimately head a practice.
- **No license file yet.** Until one is added, the default is all rights reserved.
