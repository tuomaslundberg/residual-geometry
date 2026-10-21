# Coding the B reading lists

> ⛔ **Embargo (Tuomas, 2026-10-21):** no session reports coding results to Tuomas — any
> coder's codes, counts or agreement — until he states his annotation is finished. Files
> named `EMBARGOED-*` hold such results. **Also not to be opened before coding:**
> `coding_key.tsv` and the `<encoder>.coding_sheet.tsv` files, which show each document's
> end and score.

**File to code: `coding_blind.tsv`** (423 documents: every document that appears at either end
of the web or cultural direction, for either encoder, once). Shuffled; shows only the
language and the first ~800 characters, so the codes are not steered by which end a document
came from. `coding_key.tsv` maps items back to encoder, direction, end, rank and score; don't
open it until coding is done.

Fill `code_place` with one of:

| Code | Use when the text is mainly about … |
|---|---|
| `FI` | Finland: Finnish places, institutions, people, events, Finnish everyday life |
| `US` | the United States |
| `ANG` | another English-speaking country (UK, Ireland, Canada, Australia, NZ …) |
| `ELSE` | some other place (another country, the EU, the world in general, a named elsewhere) |
| `NONE` | no particular place (generic how-to, product text, abstract topic) |
| `WRONG` | not in the expected language (wrong-language page the filter missed) |

Judge what the text is *about*, not the language it is written in or the site it comes from.
If two apply, pick the dominant one and write the other in `code_note`. Anything odd (spam,
machine translation, boilerplate) can go in `code_note` too.

The order is random, so stopping early still gives a fair subset: any first N rows are a
random draw from all lists. At ~10–15 s per item, all 423 take about 1.5 hours; the first
150 already give usable counts.

Save as tab-separated text with the same columns. **Tool:** `open experiments/thesis/annotate.html` (Mac browser), load `coding_blind.tsv`; keys `1`–`6` or `F U A E N W` code and advance, `/` opens the note, progress persists in the browser, *Export TSV* writes the same columns. It shows only the text and language — no model codes. Afterwards `coding_key.tsv` joins the codes
to the lists, and the counts (share of FI / US per end, per encoder and direction) go next to
the automatic mention proxy in `thesis_analyses.md` § B.
