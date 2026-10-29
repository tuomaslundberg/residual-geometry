# EMBARGOED-claude-codes.tsv — how these codes were made

> **Blind coding:** do not summarise or count these codes, or join them to `coding_key.tsv`,
> until all coding is finished (see `CODING.md`).

- **Coder:** a fresh instance of Claude Opus 5.5 (general-purpose agent), run on 2026-10-21
  as a blind second coder.
- **What it saw:** only `CODING.md` and `coding_blind.tsv` (shuffled, text only; no end,
  score, encoder, site or key). It was told not to open any other file, run git, or search
  the web, and to judge each text only by its own content. It reported opening, in addition,
  only one general document unrelated to the study and its own scratch copies (deleted
  afterwards).
- **Why a separate coder:** the lists were built by a process that read the documents at both
  ends, so codes from it would not be blind.
- **Instructions:** the CODING.md scheme (FI / US / ANG / ELSE / NONE / WRONG; dominant place,
  second one in `code_note`; garbled or spam texts coded from the names they still contain,
  noted in `code_note`).
- **Output check (structure only, no counts looked at):** 423 rows in the blind sheet's order,
  every `code_place` one of the six codes; 235 rows carry a note.
