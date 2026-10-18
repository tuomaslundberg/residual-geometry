# Looking round — what language removal takes away, and what remains — 2026-10-18

*LaBSE and XLM-R base on the same data: 83,468 Finnish + 94,403 English HPLT v3 web documents
(machine-translated dropped), and 20,000 Europarl translation pairs (2,000 held out for
retrieval). Erasers fitted on web text. Full reports with every document listed:
[`looking_round/LaBSE.md`](looking_round/LaBSE.md) ·
[`looking_round/xlm-roberta-base.md`](looking_round/xlm-roberta-base.md). Code:
`experiments/looking_round.py`, run with `experiments/slurm/sl-run-looking` (≈3 min per encoder
from the embedding cache). This is a first look, written to be read, not a result section.*

## The three lenses, in one paragraph each

1. **What the language direction separates.** Take the direction from the average English web
   document to the average Finnish one. Score every document along it and read the documents
   at both ends, within each language.
2. **Retrieval.** Does a Finnish sentence still find its English translation after removal
   (Europarl)? And which English web pages are a Finnish page's nearest neighbours, before
   and after removal?
3. **Language versus culture.** In a translation pair only the language differs, so the
   average Finnish-minus-English difference over translation pairs is a *pure-language*
   direction. The web direction also carries whatever differs between what Finnish and
   English pages are about. The part of the web direction the pure-language direction does
   not explain is a candidate *cultural* direction; read its ends too.

## What came out

| | LaBSE | XLM-R base |
|---|---|---|
| Web direction vs pure-language direction (cosine; 1 = same) | **0.37** | **0.92** |
| Directions INLP removes | 13 | 1 |
| Translation retrieval FI→EN before → after removal (mean centering / INLP) | 0.99 → 0.99 / 0.99 | **0.30 → 0.73 / 0.53** |
| English neighbours of Finnish pages that are English pages on `.fi` sites (base rate 0.05 %) | 0.2 % → 0.1 % | **9.6 % → 0.04 %** |
| English neighbours with the same register as the Finnish page, before → after INLP | 62 % → **52 %** | 56 % → **60 %** |
| Top-10 neighbours unchanged by INLP | 62 % | 15 % |

### XLM-R: removal washes away language *form*, and meaning comes through

- **The web difference between Finnish and English is almost all pure language** (cosine 0.92
  with the translation-pair direction). One direction carries it; INLP removes just that one.
- **Reading along it:** the most "Finnish" Finnish documents are dense official Finnish:
  municipal procurement guidelines (Tampere), service-voucher rulebooks, meeting notices,
  curricula, elderly-care reports: long compounds, capitalised headings. The most
  English-leaning Finnish documents mix in English: theses with English abstracts, Nokia
  manuals, metal magazines full of band names. Among the English documents, the
  "Finnish-leaning" end is simply *not English*: Persian, Chinese, Swedish, Bosnian, Spanish,
  Danish, Arabic pages in the English sample. The direction measures how Finnish or how
  English the *text* is, not what it is about.
- **Before removal, XLM-R matches by word form.** Finnish pages' nearest English pages are
  English pages on Finnish sites 190 times more often than chance. One finland.fi article
  about a researcher in Oulu, full of Finnish names, is a top-3 neighbour for 7 of the 14
  unrelated Finnish example pages (a cosmetics guide, a speech by the prime minister, …).
- **After removal, meaning comes through.** Translation retrieval goes from 0.30 to 0.73,
  register agreement among neighbours rises (56 % → 60 %), and 85 % of each page's
  neighbours change.

### LaBSE: language form is already gone, so "language" removal takes away *culture*

- **Translation retrieval is 0.99 before any removal**, as LaBSE's training intends. There is
  little pure language left to remove.
- **Yet Finnish and English web pages still differ, and mostly not by language:** the web
  direction is only 0.37 aligned with the pure-language direction. INLP needs 13 directions
  to make the languages inseparable, and the cultural direction lies almost entirely (0.93)
  inside what it removes.
- **Reading along it, this difference is about place and public sphere.** The most "Finnish"
  Finnish documents are everyday Finnish life: a ferry trip to Tallinn on Silja Europa,
  orienteering season in Varsinais-Suomi, a horse groom's career, moving away from North
  Karelia, dog-dancing championships in Norway, a hike on Patvinsuo, the kaamos photo festival
  in Inari, Hämeenkyrö council minutes. The most "English" English documents are the American
  public sphere: Congress, the CFPB, the Fed, US sanctions, the Washington Post, the NBA,
  Benghazi. In between sit the crossings: **Finnish pages about American things lean English**
  (Trump and climate policy, the MIT blackjack team, a Philadelphia hotel lawsuit, CIA
  documents), and **English pages from outside the Anglophone world lean Finnish**, mostly
  European (Latvian and Baltic art events, a Danish travel diary, a French web shop).
- **Register follows the same axis:** among Finnish documents, the Finnish end is 72 %
  narrative (NA) against under 17 % at the English end; English documents tilt the same way,
  more weakly (32 % against 19 %). So removing "language" from LaBSE also removes part of
  what distinguishes narrative from informational and persuasive text, which is why INLP
  lowered register agreement (62 % → 52 %) and the register RSA in the balanced runs.
- **Retrieval shows what remains:** LaBSE matches topic before and after removal (skin care to
  skin care, LED to LED, a Finnish court-procedure text to US state law). Finnish-specific
  content has no Finnish-specific English counterpart to find, so it lands on its closest
  American equivalent.

## What this suggests (for Tuomas to judge)

- **"What remains after language removal" has two different answers depending on whether the
  encoder was trained to suppress language.** In XLM-R, removal takes away language form and
  uncovers meaning. In LaBSE the form is already gone, and the same procedure takes away the
  cultural difference between the two webs (everyday Finland vs the American public sphere)
  plus the register mix that comes with it.
- **The parallel direction is what makes this visible.** Without it, LaBSE's 13 removed
  directions would read as "the language signal". Comparing against translation pairs shows
  most of them are not language in the translation sense.
- **Erasure is not neutral about culture.** An eraser fitted to separate Finnish from English
  web text removes whatever separates the two corpora. In a model that no longer encodes much
  language, that is mostly culture and genre.

## Caveats

- **The cultural direction is partly a corpus property.** It measures how the Finnish and
  English *samples* differ as the encoder sees them: Finnish web text is about Finland,
  English web text is US-centred. That is the phenomenon, but the sampling decides its
  strength. The proper sample (high WDS, prototypical documents) should be read the same way.
- **Europarl is one domain** (parliamentary speech). Pairing cancels the domain in the
  difference, but translationese stays in it.
- **The MT filter misses some machine translation:** several English-leaning "Finnish" pages
  in LaBSE's list read as machine-translated commercial text (gold-price API, casino guides,
  eTurboNews in Finnish).
- **Truncation differs:** LaBSE reads 256 tokens, XLM-R 512.
- **Retrieval counts for `.fi` sites are small** (about 50 English pages in 94k), and XLM-R's
  effect is driven by a few hub pages. The direction of the effect is clear; its size is not
  a stable estimate.
- **One first look, one corpus, one seed.** Nothing here has a chance level yet; that comes
  after deciding what is worth testing.
