# EMBARGOED-claude-codes.tsv — how these codes were made

> ⛔ **Embargoed** (Tuomas, 2026-10-21): do not report, summarise or count these codes to
> Tuomas, or join them to `coding_key.tsv`, until he states his own annotation is finished.
> Rule in `~/.context/corrections.md`.

- **Coder:** a fresh Claude sub-agent (general-purpose; Claude Opus 5.5), spawned
  2026-10-21 by the pipeline session at Tuomas's request, as a blind second coder.
- **What it saw:** only `CODING.md` and `coding_blind.tsv` (shuffled, text only; no end,
  score, encoder, site or key). It was told not to open any other file, run git, or search
  the web, and to judge each text only by its own content. It reported opening, in addition,
  only the organisation's mandatory PCP working-agreement document (general, unrelated to the
  study) and its own scratch copies (deleted afterwards).
- **Why a sub-agent:** the pipeline session built the lists and read the documents at both
  ends, so its own codes would not be blind.
- **Instructions:** the CODING.md scheme (FI / US / ANG / ELSE / NONE / WRONG; dominant place,
  second one in `code_note`; garbled or spam texts coded from the names they still contain,
  noted in `code_note`).
- **Output check (structure only, no counts looked at):** 423 rows in the blind sheet's order,
  every `code_place` one of the six codes; 235 rows carry a note.
