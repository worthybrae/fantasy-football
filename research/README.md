# Research lab

Experiments that test what this project believes about its own model, and
posts written from what they measured.

## Why it is built this way

Over one afternoon of ad-hoc analysis, three claims were made and then had
to be retracted: the schedule factor was called "essentially noise" (it is
not), the value of preseason betting lines was projected at 0.272 (measured:
0.337), and the O-line rating was declared to have no effect on production
(it has a real one on efficiency — the first test measured the wrong
outcome). Every one came from a number stated from memory or from a model
rather than read off a result.

So the rule is mechanical: **a post cannot restate a number, only reference
one.** Prose is a template; `{r2_gain}` resolves against the dict the
experiment returned, or `render.py` raises. Findings are stored with the git
revision that produced them.

## Layout

    lab.py                framework: Experiment, Result, resolve()
    data.py               shared inputs (the odds workbook, team codes)
    run.py                run experiments, persist findings
    render.py             findings + prose template -> HTML post
    experiments/          one module per experiment
    results/<id>.json     what it measured, and at which revision
    posts/<id>.md         the prose, with {placeholders}
    posts/<id>.html       the rendered post

## Running

    python research/run.py            # everything
    python research/run.py e001       # one
    python research/render.py e001    # rebuild a post from stored findings

`run.py` opens `data/nfl.duckdb` read-only, so stop the API server first —
DuckDB will not share the file with a writer.

## The odds workbook

`data.py` needs historical opening lines, which nflverse does not carry (it
stores closing lines only). They come from a spreadsheet at
aussportsbetting.com, which serves 403 to anything scripted — so it is
downloaded by hand and committed to `research/data/nfl.xlsx`, or pointed at with
`NFL_ODDS_XLSX`.

## Adding an experiment

Write the question first, before you know the answer, so the experiment
cannot be quietly reframed around whatever it happens to find. Return a flat
dict of measured values; anything a post will mention has to be in it. Set
`overturns` when the answer contradicts what the project currently does —
those are the ones worth reading.
