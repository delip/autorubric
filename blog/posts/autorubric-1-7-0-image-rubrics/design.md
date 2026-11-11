# Design: AutoRubric 1.7.0 release (image rubrics)

The presentation record for `post.md`. Together with `blog/RENDERING.md` and
`blog/DESIGN.md` it must be enough for a fresh agent to reproduce `index.html`.

## Intent

A release post that reads as part of autorubric.org: dark GitHub palette, one gold
accent, Inter and JetBrains Mono. A reader should get the release in the first
screen (the "In this release" box), then see at once why images matter: Alice's and
Bob's charts side by side with their verdicts, the `trend` row marked as the one the
text is about. The page is also built to be lifted by search engines and answer
engines: a marked summary, verbatim question headings, a FAQPage node, real tables.

The one memorable element is the Alice/Bob comparison. Everything around it stays
quiet: plain prose, base code blocks, base tables.

## Layout

- The article is `<article class="container post-layout">` (base `.container`:
  1200px max, 24px gutters). Inside it, in DOM order: `<header class="post-head">`,
  `<nav class="toc">`, `<div class="post-body prose">`.
- Below 1100px: one column, in DOM order. Post head padding `80px 0 40px`
  (`50px 0 32px` at ≤768px).
- From 1100px (`@media (min-width: 1100px)`): CSS grid,
  `grid-template-columns: 220px minmax(0, 1fr)`, `column-gap: 60px`,
  `align-items: start`. Head in column 2 row 1, body in column 2 row 2, the rail in
  column 1 spanning rows 1–2, `position: sticky; top: 24px; margin-top: 80px` (so its
  first line aligns with the eyebrow).
- Measures inside `.post-body` (`.prose` with `max-width: none; padding-bottom: 0`):
  every direct child is capped at `var(--measure)` (700px); these children are capped
  at 960px (the style notes' code-block measure): `.wide` (the Alice/Bob grid, the
  link-card list, the tag footer), `.code-block`, `.table-wrap`, `.cost`. All left
  aligned, none centered. `.post-layout` has `padding-bottom: 80px`.
- `h2` and `h3` in the body get `scroll-margin-top: 24px`.
- Only breakpoints: 1100px (rail) and the shared 768px.

## Components

**Post head.** Eyebrow `<p class="eyebrow">` lists all three tags, each an `<a>` to
`/blog/tags/<tag>/`, as a flex row with `gap: 8px 16px` (base eyebrow style:
0.7rem, 600, uppercase, gold). `<h1>`: `AutoRubric 1.7.0: <span class="accent">grade
images against a rubric</span>` — the span is `display: block; color: var(--accent)`,
the landing hero's two-line, gold-subject signature, at the base post-head size
(2.25rem; 1.75rem ≤768px), `max-width: 800px`. Then `<p class="release-line">`
(flex, `gap: 8px 16px`, 0.9rem, `--text-secondary`, `margin-bottom: 20px`) holding
`<span class="version">v1.7.0</span>` (JetBrains Mono 0.82rem, gold text, padding
`4px 10px`, radius 8px, border 1px `color-mix(in srgb, var(--accent) 55%,
transparent)`, background the accent at 10%) and `<span>Released <time
datetime="2026-10-31">October 31, 2026</time></span>`. Then the base `.dek` (the
front matter's description) and `.post-meta`: `Delip Rao &middot; <time
datetime="2026-11-10">November 10, 2026</time>`.

**Contents rail.** `<nav class="toc" aria-labelledby="toc-title">` holding
`<details open>` → `<summary>On this page</summary>`, `<p class="toc-title"
id="toc-title">On this page</p>`, `<ol>` of nine links, one per `h2` in order, link
text = the heading text verbatim ("In this release" first). Below 1100px the
`<details>` is a box: `--bg-secondary`, 1px border, radius 12px, padding `16px 20px`,
`margin-bottom: 40px`; summary 0.9rem/600 `--text-primary`; `.toc-title` hidden. From
1100px: box styles removed, summary `display: none`, `.toc-title` shown (0.9rem/600,
`margin-bottom: 16px`). The `<ol>`: no bullets, `border-left: 1px solid
var(--border)`, `margin-top: 16px` (0 at ≥1100px), items 8px apart. Links: block,
0.9rem/1.5, `--text-secondary`, no underline, `padding-left: 16px`,
`border-left: 2px solid transparent` with `margin-left: -1px`, transition
`color 0.2s, border-color 0.2s`; hover → `--text-primary`; current
(`[aria-current="location"]`) → `--text-primary` with a gold left border.

**"In this release" summary** (directive). `<section class="summary"
aria-labelledby="in-this-release">` before the first content heading, holding
`<h2 id="in-this-release">In this release</h2>`, the body's first paragraph and the
two-bullet `<ul>` (and nothing else; the "The release also tells…" paragraph and the
install command follow outside the box). Box: `--bg-secondary`, 1px border,
`border-left: 3px solid var(--accent)`, radius 12px, padding 24px, children 16px
apart, list `padding-left: 20px`. Its `h2` is set small: 1rem/600/lh 1.5, gold,
`margin-top: 0`.

**Alice and Bob** (directive). `<div class="compare wide">`: grid,
`repeat(auto-fit, minmax(320px, 1fr))`, `gap: 20px`, `align-items: start` (two
columns from a ~660px body width, stacked below). Each card `<div class="report">`:
`--bg-secondary`, 1px border, radius 12px, padding 20px (16px ≤768px),
`min-width: 0`. Inside, in order:
1. `<figure>`: `<a class="full-size" href="assets/<who>-chart.png">` (opens the
   full-size PNG, useful on phones) wrapping `<img width="1200" height="750"
   loading="lazy" decoding="async">` with the post's alt text. The image is shown as
   submitted: `width: 100%`, white background, base 1px `--border` frame, radius 8px,
   no filter or recolouring. `<figcaption>` (base style, `margin-top: 10px`) is the
   Markdown title verbatim, with its leading name wrapped as
   `<strong>Alice.</strong>` / `<strong>Bob.</strong>` (600, `--text-primary`).
2. `<table class="verdicts" role="table" aria-label="Verdicts for Alice|Bob">` with
   the post's four columns. `thead` is visually hidden (sr-only rules) but present.
   The table is restyled as stacked rows, so explicit ARIA roles (`rowgroup`, `row`,
   `columnheader`, `rowheader`, `cell`) keep its semantics: `table`/`tbody`
   `display: block`, `margin-top: 16px`, 0.9rem/1.55. Each `tr` is a grid,
   `grid-template-columns: minmax(0, 1fr) auto`, areas `"crit verdict" "req req"
   "why why"`, `gap: 8px 16px`, padding `16px 10px`, `border-top: 1px solid
   var(--border)`. `th scope="row"` (the criterion in `<code>`) → `crit`;
   `td.verdict` → `verdict`; `td.req` and `td.why` carry `data-label="Requirement"` /
   `"Why"`, shown by `::before` as a block label (0.75rem/600, `--text-secondary`).
   `td.why` text is `--text-secondary`.
3. The `trend` row is `tr.is-trend`: `--bg-tertiary` background, radius 8px,
   `box-shadow: inset 3px 0 0 var(--accent)`, `padding-left: 16px`, no top border,
   labels in `--text-primary`. Same treatment in both cards. No text is added to mark it.

Verdict chips `<span class="chip">`: 0.75rem/600/lh 1.5, padding `2px 10px`, radius
8px. MET (`.chip.met`): gold text, gold 55% border, gold 10% fill. UNMET: plain
`.chip`: `--text-secondary` on `--bg-tertiary`, 1px `--border` (4.95:1). The word is
always present.

**Code.** Six blocks, each `<div class="code-block hl"><span
class="code-label">…</span><pre><code>…</code></pre></div>` with Pygments spans
(`HtmlFormatter(nowrap=True)`, lexers `bash`, `python`, `json`; trailing newline
stripped). Labels: `Shell`, `Python`, `JSON`. Label `margin-bottom: 10px`. At ≤768px
the block padding is `20px 16px` and the copy button sits at `top/right: 12px`. Long
lines scroll inside the `<pre>` (base). Copied state: gold text and border.

**Inline code** stays base. Model ids written as plain words in `post.md`
(openai/gpt-5.6-luna, jev-1.14.0 in prose; judge names in the cost table) stay plain.

**Formats table.** `<div class="table-wrap"><table>`, `th scope="col"` headers, the
first cell of each row a `th scope="row"`. Body row headers: `--text-primary`, 500.
Code inside tables does not wrap. Tables in the body have `min-width: 600px`, so at
375px they scroll inside `.table-wrap` instead of squeezing.

**Memory callout** (directive). `<aside class="callout" role="note">` holding only
the "**Memory.**" paragraph (the run-in bold stays; no separate callout title). Base
callout plus `border-left: 3px solid var(--accent)`. The `evaluate()` code block
follows it outside the box.

**Cost comparison** (directive). `<div class="cost">` (`--bg-secondary`, 1px border,
radius 12px, padding 24px; 20px ≤768px) holding:
1. `<figure class="cost-bars" aria-hidden="true">` — a static CSS bar pair per judge
   (the table below carries the same numbers for assistive tech and agents). Legend
   `.cost-legend`: flex, gap 16px, 0.75rem/600 `--text-secondary`, two 10×10 swatches
   "Before" (`--syn-number` `#79c0ff`) and "After" (`--accent`), `margin-bottom: 16px`.
   Rows `.cost-row`: grid `170px minmax(0, 1fr)` (one column ≤768px), gap
   `8px 16px`, padding `10px 0`, 1px `--border` between rows. Judge name 0.9rem
   `--text-primary`. Each bar line: flex, gap 10px, a flexible track holding a 10px
   high square-ended bar of width `--w`, then the dollar value (JetBrains Mono
   0.75rem `--text-secondary`, 56px right-aligned column). Widths are normalized per
   judge to the larger of its two costs: gpt-5.6-luna before 100% / after 76.09%;
   deepseek-v4.1-flash 100% / 77.27%; gemini-3.8-flash 99.56% / 100%.
2. The cost table in `.table-wrap` (`margin-top: 24px`), numeric columns `th.num` /
   `td.num` (right-aligned, tabular), row-header judge names `nowrap`, first column
   flush left (`padding-left: 0`).
Dollar amounts, written `\$` in `post.md`, are plain `$` in the page.

**"What else" list.** `<ul class="more">`, items 16px apart, bold run-ins kept.

**Questions** (directive). `<h2 id="questions">`, then five
`<div class="faq-item">` each with `<h3 id>` (question verbatim) and `<p>` (answer,
always visible; no `<details>`). Item: `padding-top: 24px`, `border-top: 1px solid
var(--border)`, 24px apart; h3 1rem/600, `margin-top: 0`; answer `margin-top: 8px`,
`--text-primary`. The same questions and answers (plain text, code spans unwrapped)
form the FAQPage node.

**Where to go next.** `<ul class="next wide">`: grid `repeat(auto-fit,
minmax(280px, 1fr))`, gap 20px, no bullets. Each `<li>` is a card
(`--bg-secondary`, 1px border, radius 12px, padding `20px 24px`, 0.9rem/1.55
`--text-secondary`, hover border `--text-secondary` over 0.2s). The link text is 1rem/600
`--text-primary` (gold on hover), no underline, followed inline by the post's
trailing words (the comma kept). The link's `::after` covers the whole card
(`inset: 0`), so the card is the click target; its focus ring is drawn on the
`::after`. Links are the resolved `{{ site.url }}` URLs, same tab.

**Tags.** `<nav class="post-foot wide" aria-label="Tags">` with the base
`<ul class="tags">` of three `a.tag` chips; `margin-top: 60px`, `padding-top: 24px`,
1px top border.

**Head and structured data.** Head as the skeleton, filled from the front matter;
`keywords` = the three tags plus "AutoRubric 1.7.0, image inputs, LLM judges,
decision models, Jev, prompt cache, per-criterion grading, LiteLLM"; one
`article:tag` per tag; `article:section` = `release`; `og:image:alt` and
`twitter:image:alt` describe the card. JSON-LD `@graph`: `BlogPosting` (`@id` …`#article`,
`articleSection: "release"`, `wordCount: 2099`, `about` → the SoftwareApplication,
`mentions` → the SoftwareApplication, LiteLLM, Jev, TypeSafe), `SoftwareApplication`
(`@id` `https://autorubric.org/#software`, AutoRubric, `softwareVersion` 1.7.0, url
`https://autorubric.org/`, `applicationCategory` DeveloperApplication,
`releaseNotes` = this post), `BreadcrumbList`, `FAQPage` (`@id` …`#questions`). The
word count is the body's prose words (code blocks, alt text and directives excluded;
captions and table text included), counted on the resolved body.

**Math.** KaTeX is omitted (its link, two scripts and the auto-render call): the body
has no TeX, only `\$` dollar amounts, which are rendered as plain `$`. `math` is
left at its default in the front matter.

**Interface text added** (not in `post.md`): "On this page" (summary and rail
title); "v1.7.0"; "Released"; the release date "October 31, 2026" taken from the
`release_date` variable; code labels "Shell", "Python", "JSON"; button text "Copy",
"Copied", "Copy failed" and its `aria-label` "Copy code"; the bar legend "Before",
"After", and the bars' judge names and dollar values (repeated from the table,
`aria-hidden`); `aria-label`s "Verdicts for Alice", "Verdicts for Bob", "Tags"; the
`Requirement` / `Why` labels (the table's own headers, via CSS). The "In this
release" heading comes from the author's directive.

## Libraries

None. Pygments (the version in the project's `uv` environment) runs at render time
only. Fonts: Inter 400/500/600/700 and JetBrains Mono 400 from Google Fonts, as in
the skeleton.

## Interactive elements

One inline script at the end of `<body>` (plus the shared footer's year script):

- **Copy buttons.** When `navigator.clipboard` exists and the context is secure, each
  `.code-block` gets a `<button class="copy-code">Copy</button>` (base style, top right)
  and an sr-only `aria-live="polite"` status. Click copies the `<code>` text (final
  newline dropped), shows "Copied" for 2000ms, then "Copy". Without JavaScript or the
  Clipboard API there is no button; the code is selectable text.
- **Rail state.** `matchMedia('(min-width: 1100px)')`: the `<details>` is open when it
  matches and closed when it does not, re-synced on `change`. The HTML ships
  `<details open>`, so with JavaScript off the contents list is open at every width.
- **Current section.** On scroll (throttled with `requestAnimationFrame`), the last
  rail target whose top is ≤120px from the viewport top gets
  `aria-current="location"`; the first link is current at the top. No smooth
  scrolling and no animation; the 0.2s color/border transition is removed under
  `prefers-reduced-motion` by `base.css`. Without JavaScript no link is marked.

## Social card

`card.html` → `card.png`, 1200×630, body `#0d1117`, system font stack
(`'Inter', -apple-system, …`; JetBrains Mono → SF Mono/Menlo for mono), rendered
with headless Chrome at device scale 1 (about 50 KB).

- Top left at (72, 72): "AutoRubric" wordmark 34px/700 `#e6edf3`, then 20px gap and a
  "v1.7.0" badge (mono 22px, gold, padding `6px 14px`, radius 8px, 2px gold 55%
  border, gold 10% fill).
- Title at (72, 196), 600px wide, 60px/1.15/700, letter-spacing −0.5px:
  "AutoRubric 1.7.0:" in `#e6edf3`, then "grade images against a rubric" on its own
  lines in gold `#f0b429`.
- "autorubric.org/blog" at left 72, bottom 64, 22px `#8b949e`.
- Motif at (740, 72), 388px wide: a white 388×243 sheet (radius 12px, 1px `#30363d`)
  holding an inline-SVG line chart drawn after Bob's (rising, then flat; four light
  gridlines, dark axes, blue `#2b6cb0` 4px line with seven 6px dots, no labels); 20px
  below, three rows 56px tall, 12px apart (`#161b22`, 1px `#30363d`, radius 8px, 22px
  mono criterion name left, chip right): `axes` UNMET, `trend` UNMET (row on
  `#21262d` with a 4px gold inset left bar), `chart_type` MET. Chips 16px/700, padding
  `4px 12px`, radius 8px; MET gold on gold 10% with gold 55% border; UNMET `#8b949e`.
  These are Bob's verdicts from the post.

## Deviations

- The "In this release" box title is an `<h2>` (set at 1rem, gold), so the summary is
  a named section an answer engine can lift and the rail can link to.
- Layout: a 220px sticky contents rail from 1100px (a breakpoint of the post's own),
  and 960px blocks (comparison, code, tables, cost box, link cards) beside a 700px
  prose measure.
- Verdict tables are restyled as stacked rows with explicit ARIA table roles.
- Data colors: the cost bars use blue `#79c0ff` (before) and gold (after), per
  `blog/DESIGN.md`.
- The student charts are white-ground raster images, shown as submitted (white
  background inside the 1px frame), not on `--bg-secondary`.
- KaTeX omitted (no math in the body).

## Render log

- v1 (2026-11-10): first render.
- v1 (2026-11-10): text corrected after a fact check (summary sentence and the
  paragraph after it, the paragraph after Bob's table, few-shot images, registry,
  dataset paths, prompt order, few-shot caching, the "What else" list now five
  items, one FAQ answer and its FAQPage entry); the `evaluate()` block re-highlighted
  with its new import line; `wordCount` 1819 → 2012. Design unchanged.
- v1 (2026-11-10): the code block under "LLM judges" now mirrors the one under
  "Decision models" (imports, config, grader, grade); re-highlighted with Pygments.
- v1 (2026-11-10): "LLM judges" opening sentence names two example models; the unused
  `llm_judge` variable was removed from the front matter.
- v1 (2026-11-10): every model name in prose, tables and chart labels is set as inline
  `<code>` (JetBrains Mono, the `.prose :not(pre) > code` chip from base.css); the unused
  `decision_model` variable was removed from the front matter.
- v1 (2026-11-10): new section "Explanations" (`h2#explanations`) before the dataset section,
  built from existing pieces: two paragraphs, a Pygments Python block, a plain block labelled
  "Text" (interface text) holding the illustrative judge line, and the decision-model note as
  an `aside.callout`. Added to the "On this page" rail. `wordCount` 2012 → 2102.
  The "Text" block has class `wrap` (`.code-block.wrap pre { white-space: pre-wrap;
  overflow-wrap: anywhere }`) so the judge's sentence wraps instead of scrolling sideways.
- v1 (2026-11-10): the sentence introducing the two students now names Alice and Bob;
  `wordCount` 2102 → 2105.
- v1 (2026-11-10): the example explanation is now real output (Bob's report graded once with
  `openai/gpt-5.6-luna`; the `axes` line); the paragraph above it reworded; `wordCount` 2105 → 2099.
