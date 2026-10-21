# Thesis analyses A–D on the HPLT 4 thesis sample — 2026-10-21

*Sample: `$WORK/data/thesis-sample-v4/sample-N18000-cap500-s20261020/` — Finnish and English
HPLT 4 web text at WDS 9–10; one main register per document (subregisters folded in, no
hybrids, MT + register counted as a hybrid); LY left out; at most 500 documents per site per
cell; 18,000 per register per language. Wrong-language pages flagged by fastText
(`lid-*.tsv`; 23 FI, 256 EN) are left out, so 143,977 FI and 143,744 EN documents remain
(7 registers + the MT-only class; register analyses use the 7 registers).
Code: `experiments/thesis_analyses.py` (A–D) and `experiments/topics.py` (topics); outputs
in `experiments/thesis/` (metrics, coding sheets, topic lists). One run per encoder.*

*Translation pairs: Europarl (20,000 sampled pairs, parliamentary debate) and infopankki
(14,419 unique pairs, information about living in Finland); 2,000 of each held out for
retrieval, the rest give the pure-language direction.*

## Removal conditions

| Condition | What it does |
|---|---|
| raw | nothing |
| global | subtracts the pooled mean: moves the origin, keeps the FI–EN difference |
| mean | subtracts each language's mean: removes the difference and moves the origin |
| leace | equalises the language means with the smallest change; origin stays |
| inlp | removes directions until a held-out language probe is at chance (LaBSE 10, XLM-R 2) |
| transdir_ep / _ip | projects out the Europarl / infopankki translation-pair direction |

INLP and LEACE are fitted on 20,000 + 20,000 random web documents.

## A. What the language direction is made of

The *web direction* runs from the average English web document to the average Finnish one.
Registers are balanced, so register mix does not enter it.

| | LaBSE | XLM-R |
|---|---|---|
| cosine with the Europarl translation direction (95 % bootstrap) | **0.359** [0.355, 0.362] | **0.926** [0.925, 0.926] |
| cosine with the infopankki translation direction | 0.204 [0.199, 0.209] | 0.907 [0.905, 0.908] |
| Europarl vs infopankki direction | 0.875 | 0.963 |
| share of the web direction inside the two translation directions (squared) | **18 %** | **86 %** |
| the remainder ("cultural direction") inside what INLP removes | 0.91 | 0.44 |

Per register (the FI–EN difference computed within each register separately), LaBSE gives
the same web direction in all seven (cosine 0.90–0.96 with the pooled one) and the same weak
link to translation (0.27–0.44); XLM-R 0.98–0.99 and 0.90–0.94.

**Reading.** In XLM-R the difference between Finnish and English web text is almost all
translation-level language. In LaBSE, only about a fifth of it is; the rest is shared by every
register, so it is not genre. INLP, fitted to separate the two languages, removes that
remainder almost entirely in LaBSE.

## B. What the cultural direction separates (counted, then to be read)

Each document is scored along the direction; deciles from the English-leaning end (1) to the
Finnish-leaning end (10). "Mentions" = a Finland or US place/institution word in the first
500 words (crude regex proxy, listed in the code).

**LaBSE, web direction**

| Decile | 1 | 2 | 5 | 9 | 10 |
|---|---|---|---|---|---|
| Finnish docs mentioning Finland | 32 % | 33 % | 42 % | 59 % | **70 %** |
| Finnish docs mentioning the US | **19 %** | 14 % | 10 % | 8 % | 7 % |
| English docs mentioning the US | **43 %** | 30 % | 20 % | 12 % | 10 % |

The cultural direction (translation part removed) is steeper: Finnish documents 21 % → 75 %
mentioning Finland. Topics follow: Finnish/European politics, education and travel at the
Finnish end; US politics and real estate at the English end; technology and film at the
English-leaning end of the *Finnish* documents. Reading the lists: the Finnish end is local
Finland (a Tornio football match, Viro-instituutti, Wikimedia Suomi's board minutes, a
Turku museum's Tom of Finland exhibition, orienteering), and the English end is the US public
sphere (Counterpunch, the Harvard Crimson, the New York Times on Democrats, US–UK policy).
English pages from the Baltics, Serbia, India and Lithuanian courts lean *Finnish*.

**XLM-R, web direction.** The place gradient is much weaker among Finnish documents (40 % →
57 % mentioning Finland) and runs the other way among English ones (US mentions 12 % → 27 %
→ 21 %). What orders documents is language form: the Finnish end is dense official Finnish
(government bills on Finlex, research reports, fire regulations, statements to ministries),
the English-leaning end of the Finnish documents is code-switching (English band names,
quoted English, an English abstract), and the English documents' "Finnish" end is
non-native or partly non-English commercial text.

**Coding sheets** (`experiments/thesis/<encoder>.coding_sheet.tsv`): 30 documents at each end
of the web and cultural directions, per language — 240 rows per encoder — with register,
site, topic, the automatic mention flags and two empty columns, `code_place` and
`code_note`, for systematic reading. Proposed codes for `code_place`: *about Finland* · *about
the US* · *about another Anglophone country* · *about elsewhere* · *no place* · *wrong
language* (a few pages fastText misses remain: Russian in look-alike Latin letters,
Chinese with English boilerplate).

## C. Retrieval

**Translations** (held-out pairs; P@1 = share whose nearest neighbour is the translation)

| | LaBSE Europarl FI→EN / EN→FI | LaBSE infopankki | XLM-R Europarl | XLM-R infopankki |
|---|---|---|---|---|
| raw | 0.99 / 0.99 | 0.94 / 0.95 | 0.30 / 0.21 | 0.25 / 0.30 |
| global | 0.99 / 0.99 | 0.94 / 0.95 | 0.55 / 0.09 | 0.26 / 0.18 |
| mean | 0.99 / 0.99 | 0.94 / 0.95 | **0.75 / 0.59** | **0.61 / 0.53** |
| leace | 0.99 / 0.99 | 0.94 / 0.95 | 0.56 / 0.63 | 0.51 / 0.52 |
| inlp | 0.99 / 0.99 | 0.94 / 0.95 | 0.58 / 0.66 | 0.53 / 0.54 |
| transdir_ep | 0.99 / 0.99 | 0.94 / 0.95 | **0.73 / 0.73** | 0.54 / 0.53 |
| transdir_ip | 0.99 / 0.99 | 0.94 / 0.95 | 0.66 / 0.73 | 0.58 / 0.58 |

**Web neighbours** (3,000 Finnish documents, their 10 nearest English documents)

| | LaBSE same register | LaBSE neighbour mentions Finland (query does / doesn't) | XLM-R same register | XLM-R neighbour mentions Finland (does / doesn't) |
|---|---|---|---|---|
| raw | 0.53 | 1.2 % / 0.6 % | 0.49 | **10.6 % / 5.3 %** |
| mean | 0.53 | 0.9 % / 0.4 % | **0.61** | 0.8 % / 0.5 % |
| leace | 0.53 | 0.8 % / 0.4 % | 0.56 | 0.6 % / 0.5 % |
| inlp | 0.52 | 0.8 % / 0.4 % | 0.49 | 0.6 % / 0.4 % |

Base rate: 0.4 % of English documents mention Finland. Raw XLM-R pulls English pages that
merely mention Finnish names to Finnish queries at 13–26 times the base rate, whatever the
query is about; removal ends that. LaBSE's neighbours already match on content, and keep
doing so.

## D. What organises what remains

Probes are standardised, with one penalty per encoder chosen by inner cross-validation
(LaBSE C = 0.1, XLM-R C = 0.01). 2,500 documents per register per language for the
within-language probes; transfer is tested on all documents of the other language.

| | LaBSE raw | LaBSE mean | LaBSE leace | LaBSE inlp | XLM-R raw | XLM-R mean | XLM-R leace | XLM-R inlp |
|---|---|---|---|---|---|---|---|---|
| language probe (guessing 0.50) | 0.997 | 0.500 | 0.500 | 0.517 | 1.000 | 0.500 | 0.500 | 0.500 |
| register, within FI / EN (chance 0.14) | 0.75 / 0.80 | 0.75 / 0.80 | 0.75 / 0.80 | 0.74 / 0.79 | **0.83 / 0.87** | 0.83 / 0.87 | 0.83 / 0.87 | 0.83 / 0.87 |
| topic, within FI / EN (largest topic 0.08 / 0.12) | 0.62 / 0.68 | 0.62 / 0.68 | 0.63 / 0.68 | 0.62 / 0.68 | **0.69 / 0.74** | 0.69 / 0.74 | 0.69 / 0.74 | 0.69 / 0.74 |
| register transfer FI→EN / EN→FI (macro-F1, chance 0.14) | 0.70 / 0.64 | 0.73 / 0.67 | 0.73 / 0.67 | 0.71 / 0.64 | 0.74 / 0.55 | **0.78 / 0.71** | 0.78 / 0.71 | 0.67 / 0.62 |
| register-centroid RSA (chance 95th pct ≈ 0.43) | 0.74 | 0.79 | 0.73 | 0.66 | 0.45 | 0.45 | 0.45 | **0.76** |

Global centring equals raw and the translation directions change nothing within language
(full table in the metrics files).

**Reading.**

- **Removal takes nothing measurable from within a language.** Register and topic are as
  decodable after every condition as before, in both encoders: at most ten directions out of
  768 are removed, and register and topic do not live in them.
- **What removal changes is how the two languages line up.** Transfer rises once the
  difference is removed (LEACE = mean centring), most in XLM-R English→Finnish (0.55 → 0.71).
- **Once probes are fair, XLM-R encodes register and topic at least as well as LaBSE**
  within a language. Unscaled probes had put XLM-R far below (Finnish register 0.53, topic
  0.16); those numbers, and Sprint 2's XLM-R probe results, were the probe underfitting an
  anisotropic space.
- **Centroid RSA is not reliable on XLM-R.** Raw, mean, LEACE and translation conditions all
  sit at the chance edge (0.45, p ≈ 0.04–0.05), but INLP's two directions lift it to 0.76.
  Cosine on XLM-R's anisotropic vectors is dominated by a few high-variance directions;
  INLP's second direction evidently removes one of them. Open: what that direction is, and
  whether RSA after standardising the features agrees.

## What this adds to the storyline

- **Confirmed on a clean sample** (balanced registers, capped sites, high quality, two
  translation corpora): LaBSE's Finnish–English difference is about 80 % something other
  than translation-level language, the same in every register, and its documents order by
  place — Finland at one end, the US at the other. XLM-R's difference is about 86 %
  translation-level language and orders documents by language form.
- **New:** removal leaves within-language register and topic untouched in both encoders. It
  changes cross-language alignment, and, in LaBSE, the cultural information that would let
  a Finnish page find English pages about Finland.

## Caveats

- One draw (seed 20261020); intervals only for A. The place-mention proxy is a crude regex
  over the first 500 words; the coding sheets exist to check it by hand.
- Encoders truncate (LaBSE 256 tokens, XLM-R 512) and Finnish needs more tokens per word, so
  they see fewer Finnish words than English.
- Finnish cells are a census of WDS 9–10; English cells a proportional sample. The site cap
  departs from proportionality in two Finnish cells (IN, ID).
- Topics are per language (25 each, NMF), so topic results are within-language only.
- Europarl is parliamentary speech, infopankki one government information site; their
  directions agree (0.88 / 0.96), which suggests subject matter does not move the
  pure-language direction much.
