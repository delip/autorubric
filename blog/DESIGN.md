# Blog design guide

The blog is part of autorubric.org and follows `WEBSITE-DESIGN-STYLE-NOTES.md`: the
eight color tokens, Inter and JetBrains Mono, the one gold accent, 1px `#30363d`
dividers, radii of 8px and 12px, restrained motion, and the precise, citation-grounded
voice. This guide says what the blog adds, and where a post may depart. A post's own
`design.md` records what that post does.

## What is fixed

- **Tokens and fonts.** Colors come from the `:root` tokens in `blog/_shared/base.css`.
  No new accent hue. Inter 400/500/600/700 and JetBrains Mono 400 are what is loaded;
  a post that needs another weight or face adds it to its own font link and says so
  in `design.md`.
- **Navbar and footer** come from `blog/_shared/header.html` and `footer.html`
  through the marked regions (`RENDERING.md` §4). "Blog" is marked current.
- **Dark only.** The site has no light theme.
- **Accessibility.** Skip link, one `<h1>`, visible `:focus-visible` ring, text
  contrast of at least 4.5:1, controls reachable by keyboard, `prefers-reduced-motion`
  honored in CSS and in any script that animates.

## What the blog adds to the site's system

| Addition | Rule |
|---|---|
| Shared stylesheet | Posts link `/blog/assets/base.css` and add an inline `<style>`. The landing and Research pages keep their inline CSS. |
| Long-form text | Body text is `--text-primary` at `1.05rem` / `1.7` on a 700px measure. The landing page's gray prose is for short blurbs; gray is tiring over a thousand words. Captions, meta lines and asides stay `--text-secondary`. |
| Images | Raster and SVG images are allowed inside posts and for social cards. Charts and diagrams use the site palette on `--bg-secondary` or transparent ground; photographs and screenshots get the 12px radius and a 1px border. A scrim (a dark gradient) over an image is allowed where text sits on it, to hold 4.5:1 contrast. |
| Breakpoints | A post may add breakpoints of its own (a rail that appears at 1100px, say). The shared header, footer and index keep the single 768px one. |
| Shapes | Radii stay 8px and 12px for boxes. Circles are fine for dots, markers and avatars. |
| Data colors | A chart may need more than one hue. Two series are gold `#f0b429` and blue `#79c0ff`; gold is the series the text is about. A third is salmon `#ff7b72`, a fourth purple `#d2a8ff`. Blue and purple are close for color-blind readers, so beyond two series never rely on color alone: label series directly or add a second channel (shape, dash, position). |
| Syntax highlighting | Pygments at render time, colored by `base.css` (`.hl`). |
| Motion | Scroll- or interaction-driven motion is allowed when it explains something. Keep it within 0.2–0.4s, ease-out, no ambient loops, and static under reduced motion. |
| Libraries | Any, pinned and recorded (`RENDERING.md` §5). |

## Building blocks in `base.css`

`.post-head` (eyebrow, `h1`, `.dek`, `.post-meta`) · `.prose` (headings, lists, links,
blockquote, inline code, tables, `hr`) · `.code-block` (+ `.code-label`, `.copy-code`,
`.hl`) · `figure` / `figcaption` · `.table-wrap` · `.callout` (+ `.callout-title`) ·
`.math-display` · `.footnotes` · `.tags` / `.tag` · `.sr-only`.

Start from these. A post may restyle or replace any of them in its inline `<style>`;
it should not copy them.

## Where a post may depart

Layout is the post's own: column count, measure, full-bleed bands, sticky side
elements, a custom post head. So are components the building blocks do not cover.
A post may also set `chrome: none` and bring its own header. Each departure goes in
the post's `design.md` under *Deviations*, with the reason.

Use the type sizes, spacing steps and radii catalogued in the style notes before
inventing a value. When a design needs one outside the catalog (a display size for a
magazine-style opener, say), that is a deviation to record, not a reason to hold back.

## The index and tag pages

`blog/_shared/index.template.html` and `index.entry.html` are the design; the build
script fills them. Their design record is `blog/_shared/index.design.md`.

## Voice

As in the style notes §10: name the technique, give the number, cite the source, no
superlatives. Write "Cohen's kappa", `CANNOT_ASSESS`, "LLM-as-a-Judge" exactly so.
Name both judge kinds (LLMs and decision models) or stay judge-neutral.
