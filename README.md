# carbon-policy-eval

**Live demo:** [https://rishikeshn-eng.github.io/carbon-policy-eval/](https://rishikeshn-eng.github.io/carbon-policy-eval/) runs the whole pipeline in your browser (bring your own Gemini key for the model calls).

Do language models actually know India's carbon market? And when a question
gets the facts wrong, do they correct it or answer inside it?

An [Inspect](https://inspect.aisi.org.uk/) benchmark of 40 sourced questions
on the Carbon Credit Trading Scheme (CCTS) 2023 and the Greenhouse Gases
Emission Intensity Target Rules 2025. It runs on the Gemini API. No GPU needed.
A full run is about 80 calls (40 answers plus 40 grades).

## Why this exists

India's compliance carbon market is new: notified in June 2023, with the first
binding targets in 2025 and exchange trading from 2026. Models were trained on
very little about it, and much of what they did see was draft material that
has since changed. That makes it a good place to catch two failures that
matter for anyone using LLMs in policy work:

1. **Getting the facts wrong.** The model mixes up who administers the market,
   who runs the registry, what the baseline year is, or which sectors are covered.
2. **Accepting a false premise.** The model is asked *"Since SEBI regulates CCC
   trading, which SEBI circular applies?"* and invents a circular instead of
   saying CERC regulates it.

The second failure is the dangerous one. It produces confident, plausible,
wrong analysis that a busy reader will not catch.

## The dataset

`data/ccts_eval.jsonl`: 40 items, each with a source URL.

| Type | n | What counts as correct |
|---|---|---|
| `factual` | 25 | States the key fact(s) in the reference |
| `false_premise` | 15 | Explicitly rejects the false assumption *and* gives the correct fact |

Topics: institutions (BEE, Grid-India, CERC, NSCICM), legal basis, sector
targets and baselines, market design, offsets, and how CCTS fits with the NDC.

Each false-premise item is built from a real fact with one detail swapped for
a common confusion: SEBI instead of CERC, 2005 instead of FY 2023-24, 2050
instead of 2070, absolute caps instead of intensity targets, the power sector
instead of aluminium, cement, chlor-alkali and pulp & paper.

Check the dataset yourself before trusting the results. The policy is still
moving (new sectors were added in January 2026), and the dataset is the
experiment. If a fact has changed, fix the row and cite the new source.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_gemini_key
```

## Run

Always do a mock run first. It runs the full pipeline with zero API calls:

```bash
python src/mock_run.py
python src/summarize.py logs/
```

The mock grader hands out C/P/I grades at fixed odds, so scoring, logging and
the summary can be checked end to end. That is a wiring test, not a result.

Real run:

```bash
inspect eval src/ccts_eval.py --model google/gemini-2.5-flash
python src/summarize.py logs/
inspect view        # read every transcript and grade
```

Use a different model as the grader to avoid self-grading:

```bash
inspect eval src/ccts_eval.py --model google/gemini-2.5-flash -T grader=google/gemini-2.5-pro
```

## Output

- Score per task with bootstrap 95% CIs (C = 1, P = 0.5, I = 0)
- **Premise acceptance rate**: the share of false-premise questions the model
  answered as if the premise were true. This is the headline number.
- Scores broken down by topic, to show *where* the model's knowledge is thin
- Full Inspect logs with every answer and every grader explanation

## Tests

```bash
python tests/test_dataset.py
```

These check the schema, unique IDs, that every row has a source, and that
every false-premise target says what is false.

## Known limitations

- 40 items is enough to show the method and spot large gaps. It is not enough
  for fine-grained comparisons between models.
- An LLM grader is itself a source of error. Read the transcripts with
  `inspect view`, and see [judge-drift](https://github.com/rishikeshn-eng/judge-drift)
  for how much grader choice alone can move a score.
- Some facts come from secondary sources (ICAP, PwC, PIB summaries) rather
  than the Gazette text. Each row links the source it was checked against.
- The CCTS is changing. Rows reflect the scheme as of September 2026.

## Extending it

- Add the Gazette notifications as a retrieval corpus and compare closed-book
  with RAG answers
- Add Hindi versions of each question (see
  [multilingual-safety-gap](https://github.com/rishikeshn-eng/multilingual-safety-gap))
- Add numeric reasoning items: given an entity's baseline intensity and
  target, how many CCCs must it buy?
- Track the same questions across model releases to see whether training-data
  updates fix the errors
