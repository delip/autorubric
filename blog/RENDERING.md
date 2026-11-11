# Rendering a blog post

This is the contract for the design/rendering agent. It says how `post.md` and
`design.md` become `index.html`. There is no template engine: you write the page.

```
blog/posts/<slug>/post.md   +   blog/posts/<slug>/design.md   ->   blog/posts/<slug>/index.html
     content (the author's)          presentation (yours, steered by the author)       the served page
```

The transform must be repeatable. A fresh agent holding only `post.md`, `design.md`,
this file and `blog/DESIGN.md` should produce the same page, give or take details
`design.md` leaves open. When you decide something the author would want kept, write
it into `design.md`.

## 1. Read

1. `blog/DESIGN.md`, the blog-wide design guide (it extends `WEBSITE-DESIGN-STYLE-NOTES.md`).
2. `post.md`. The front matter is described in `blog/README.md`.
3. `design.md`. If it is still the scaffold, or the author gave you a brief instead,
   design the page and fill `design.md` in as you go.
4. `blog/_shared/post.skeleton.html`, the starting point for every page, and
   `blog/_shared/base.css`, which every page links.

## 2. Content rules

- **`post.md` is the source of truth for words.** Render all of it and add nothing:
  no new claims, numbers, headings or captions. Three things are presentation and
  are fine: interface text for something you build (a button, tab or axis label, a
  callout's one-word title, a widget's name), linking the author's existing words
  to a section or equation on the page, and text alternatives for a chart you draw
  (kept to what the chart shows). List the interface text you added in `design.md`.
  If the prose needs a change to suit the design, propose it to the author; do not
  make it.
- **Figures.** `![alt](assets/file.png "caption")` gives the alt text and, as the
  Markdown title, the caption. A purely decorative image (a hero behind the title)
  takes `alt=""`.
- **Money.** A dollar amount is written `\$5` in `post.md`. An unescaped pair of
  dollar signs in one paragraph reads as math, to the variable resolver and to KaTeX
  alike. In the page, write the plain `$` inside an element KaTeX skips (add its
  class to auto-render's `ignoredClasses`).
- **Variables.** `{{ name }}` in the body is replaced by the front matter's
  `variables` entry, or a built-in (`site.title`, `site.url`, `site.description`,
  `site.author`, `post.title`, `post.slug`, `post.description`, `post.date`,
  `post.updated`, `post.version`, `post.url`, `post.author`). Text inside code spans,
  fenced code and math is left alone. Get the resolved body from the script so that
  your page and the Markdown copy agree:

  ```bash
  python -c "import sys; sys.path.insert(0, 'scripts'); import blog; print(blog.load_post(blog.BLOG_DIR / 'posts' / 'SLUG', blog.load_site()).resolved_body())"
  ```
- **Title.** The body has no top-level heading. The page's one `<h1>` is the front
  matter's `title`. Body headings start at `##` → `<h2>`.
- **Directives.** An HTML comment in `post.md` that starts with `design:` is a note
  to you from the author (for example `<!-- design: interactive kappa table here -->`).
  Act on it, record what you built in `design.md`, and leave the comment out of the page.
  A directive applies to the block that follows it. `pull quote` means that the
  following sentence or paragraph is itself set as a pull quote, in place; do not
  duplicate text.
- **Markdown to HTML is done by hand.** There is no converter to run. Footnotes
  (`[^1]`) become a superscript link to an entry in `<section class="footnotes">`,
  with a link back.

## 3. The head starts from the front matter

Copy the skeleton's `<head>` and fill it from the front matter. These are defaults:
you may adapt the `<title>`, description and Open Graph text when the page reads
better for it. When you do, declare it under `## Deviations` in `design.md` with a
list item that starts with the key in backticks (`title`, `description`, `og:title`
or `og:description`) and gives the reason, such as ``- `title`: shortened to fit a
search result``. Otherwise `blog.py check` reports the difference. Everything else must match the front matter.

| Head value | From |
|---|---|
| `<title>` | `title` + ` — AutoRubric Blog` |
| `meta description`, `og:description`, `twitter:description` | `description` |
| `canonical`, `og:url`, JSON-LD `url` | `https://autorubric.org/blog/<slug>/` (or `canonical`) |
| `og:image`, `twitter:image`, JSON-LD `image` | `https://autorubric.org/blog/<slug>/card.png` (or `image`) |
| `article:published_time`, JSON-LD `datePublished` | `date` |
| `article:modified_time`, JSON-LD `dateModified` | `updated`, else `date` |
| `article:tag` (one per tag), `keywords` | `tags` |
| author fields | `author`, else `site.yml`'s |

Keep both alternates: `text/markdown` to `index.md` (the build writes that file) and
`application/atom+xml` to `/blog/feed.xml`. Keep the `BlogPosting` and
`BreadcrumbList` JSON-LD. Add further schema.org nodes when the content has them
(`FAQPage`, `HowTo`, `Dataset`), using only what `post.md` says.

## 4. The body

- **Semantic HTML.** One `<article>` holding one `<h1>`; headings in order with no
  skipped level; `<time datetime>` for dates; `<figure>`/`<figcaption>`; `<table>`
  with `<th scope>`; every `<img>` with `alt`, `width` and `height`; `id`s on `<h2>`
  and `<h3>` so sections can be linked. Search engines and AI agents read this structure.
- **Readable with JavaScript off.** All of the post's text is in the HTML as
  delivered. An interactive piece sits on top of a static version of the same
  content (a table, a static SVG, a default state), never in place of it.
- **Elements of your own.** `base.css` styles the site's header and footer only as
  direct children of `<body>`, so `<header>`, `<nav>` and `<footer>` inside the
  article are yours to style.
- **Shared navbar and footer.** Keep the skeleton's two marked regions and fill them with

  ```bash
  python scripts/blog.py sync-chrome
  ```

  Never edit inside the markers. A post whose design needs its own header sets
  `chrome: none` in the front matter and may drop the markers; it must still link
  back to `/blog/` and `/`.
- **Math.** Leave the author's TeX in the HTML between its delimiters (`$…$`, `$$…$$`).
  KaTeX, loaded by the skeleton, typesets it in the browser, and the TeX is the
  no-JavaScript fallback and what agents read. Escape `<`, `>` and `&` inside TeX
  as HTML entities. Wrap display math in `<div class="math-display">`. A post with
  `math: false` drops the KaTeX lines. `design.md` may choose another renderer.
- **Code.** Highlight at render time, not in the browser: run Pygments and paste
  its spans inside `<div class="code-block hl"><pre><code>…</code></pre></div>`.
  `base.css` colors Pygments' classes with the site's syntax palette. Run it from
  the repository root, where `uv` finds the project's environment.

  ```bash
  uv run python -c "import sys; from pygments import highlight; from pygments.lexers import get_lexer_by_name; from pygments.formatters import HtmlFormatter; print(highlight(sys.stdin.read(), get_lexer_by_name('python'), HtmlFormatter(nowrap=True)))" < snippet.py
  ```
- **Charts** need a static version in the HTML as delivered: an inline SVG (one way
  is to run the chart code in headless Chrome and paste the SVG it draws) or a data
  table. The script then replaces or enhances it.
- **Tags.** The skeleton's eyebrow shows one tag; list every tag when there are several.
- **Images** go in `assets/` and are referenced relatively (`assets/figure-1.svg`).
  Prefer SVG for diagrams and charts. Give raster images `width`/`height`, and
  `loading="lazy"` below the fold.

## 5. Libraries

Any JavaScript library is allowed. Record each in `design.md` with its exact version.

- Load from `cdn.jsdelivr.net` or `cdnjs.cloudflare.com` at a pinned version with an
  `integrity` hash and `crossorigin="anonymous"`, or copy the file into `assets/vendor/`.

  ```bash
  curl -fsSL URL | openssl dgst -sha384 -binary | openssl base64 -A
  ```
- TypeScript: keep sources and their tests in `src/` (not published), compile to a
  committed file, and record the command in `design.md`:

  ```bash
  npx --yes esbuild@0.25.0 blog/posts/SLUG/src/widget.ts --bundle --minify --format=iife --target=es2020 --outfile=blog/posts/SLUG/assets/widget.js
  ```
- Scripts use `defer` or sit at the end of `<body>`. A script that fails to load must
  leave the page readable.

## 6. Social card

Each post has a 1200×630 `card.png`. Write `card.html` (a self-contained page whose
`<body>` is exactly 1200×630, on `#0d1117`, with the post title, the "AutoRubric"
wordmark and the gold accent; system fonts or fonts embedded so it renders offline)
and screenshot it:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --hide-scrollbars --force-device-scale-factor=1 --window-size=1200,630 --screenshot=blog/posts/SLUG/card.png "file://$PWD/blog/posts/SLUG/card.html"
```

Design the card for the smallest place it appears. LinkedIn shows a link's image as
a thumbnail about 250px wide, so anything set below roughly 80px on the 1200px card
is unreadable there. Use few words (a short headline, the wordmark, at most a
version or date), very large type, and one bold shape. Leave out URLs, captions and
fine detail. Check it by scaling `card.png` down to 256px wide and reading it.

Set `og:image:alt` to a sentence describing the card. Keep `card.png` under about
300 KB (flat color compresses well; avoid noise and photographs).

## 7. Record, stamp, check, look

1. Update `design.md`: every section filled, and a new line in the render log
   (`- v<version> (<date>): what changed`). Write it for an agent that will never
   see your `index.html`. That means numbers, not adjectives: every size, spacing,
   color, breakpoint and selector choice that `base.css` does not already give;
   the element that carries each piece (which tag holds a label, what wraps what);
   the full list of interface text you added; the social card's layout with its
   sizes and positions; and each script's behavior, including its reduced-motion
   and no-JavaScript states. A note about working around a shared file goes stale
   when that file is fixed, so on a re-render check each such note against the
   current `base.css` and drop the ones that no longer apply.
2. Record the sources in the page, so a later edit to `post.md` or `design.md` shows
   up as a stale render. Do this last: any edit to either file after it makes the page stale.

   ```bash
   python scripts/blog.py stamp SLUG
   ```
3. Check. Fix everything it reports.

   ```bash
   python scripts/blog.py check
   ```
4. Look at the page. Build with drafts and open it in a browser at desktop width and
   at 375px; operate every interactive element; read the console for errors; and
   read it with JavaScript off.

   Two things to know when you only have headless Chrome. On macOS its window is
   never narrower than 500px, so a `--window-size=375,…` screenshot is laid out at
   500px and cropped, which looks like overflow that is not there. Check narrow
   widths with real device emulation, or by screenshotting a page that holds the
   post in a 375px-wide `<iframe>`. And it has no working switch for disabling
   JavaScript: screenshot a copy of the page with its `<script>` elements removed.

   ```bash
   python scripts/blog.py build --out /tmp/autorubric-site --drafts
   ```

   ```bash
   python3 -m http.server -d /tmp/autorubric-site 8001
   ```

## 8. Re-rendering

When `post.md` changes, the author bumps `version`. Re-render from `post.md` and
`design.md`, keeping the design unless asked otherwise. When only the design changes,
edit `design.md` first, then the page. In both cases finish with section 7.
