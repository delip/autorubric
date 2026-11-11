# The AutoRubric blog

Source for `autorubric.org/blog/`. Posts are written in Markdown with LaTeX. A
design/rendering agent turns each one into a hand-designed static page, and
`scripts/blog.py` generates the index, tag pages, feed, sitemaps, `robots.txt` and
`llms.txt` around them. No site generator is involved.

## Files

```
blog/
  README.md            this file
  RENDERING.md         how post.md + design.md become index.html (the agent's contract)
  DESIGN.md            blog-wide design guide
  site.yml             title, description, URL, page size, llms.txt entry points
  _shared/             navbar, footer, base.css, the post skeleton, the index template
  posts/<slug>/
    post.md            the content; you edit this
    design.md          the presentation record; the agent keeps it, you steer it
    index.html         the rendered page (committed)
    card.html          social card source
    card.png           social card, 1200x630
    assets/            images, data, scripts
    src/               TypeScript or other sources that compile into assets/ (not published)
```

A post is served at `/blog/<slug>/`, with a Markdown copy at `/blog/<slug>/index.md`.

## Writing a post

```bash
python scripts/blog.py new my-post-slug
```

This creates `post.md` and `design.md`. Write the post in `post.md`, then ask the
agent to render it ("render the blog post my-post-slug", or the `render-blog-post`
skill). Say what you want the page to look like; the agent writes that down in
`design.md` and builds `index.html` from the two files. To change the design later,
tell the agent or edit `design.md`, and have it render again.

### Front matter

```yaml
---
title: Why kappa and phi disagree          # the page's <h1>
slug: kappa-vs-phi                         # the directory name and the URL
date: 2026-11-10                           # first published
version: 1                                 # bump when the content changes
description: One or two sentences, at most 160 characters.   # search and social snippet
tags: [metrics, agreement]                 # lowercase-with-dashes; each gets /blog/tags/<tag>/
# optional
updated: 2026-11-20                        # last substantive change
author: Delip Rao                          # a name or a list; default from site.yml
draft: true                                # left out of the deployed site
variables:                                 # used in the body as {{ name }}
  judge_model: openai/gpt-5.6-luna
canonical: https://example.com/original    # when the post first appeared elsewhere
image: hero.png                            # social image instead of card.png
math: false                                # the post has no LaTeX
chrome: none                               # the post brings its own header and footer
---
```

Unknown keys are errors, so a typo does not pass silently.

### In the body

- Markdown, starting at `##` (the title comes from the front matter).
- LaTeX between `$…$` and `$$…$$`.
- `{{ name }}` for a front-matter variable or a built-in such as `{{ post.date }}` or
  `{{ site.url }}`. Not replaced inside code or math.
- `\$5` for a dollar amount. Two unescaped dollar signs in a paragraph read as math.
- `![alt text](assets/chart.png "Caption shown under the figure")` for a figure.
- `<!-- design: … -->` for a note to the rendering agent at that spot.

## Commands

| Command | What it does |
|---|---|
| `python scripts/blog.py new <slug>` | Scaffold a post. |
| `python scripts/blog.py stamp <slug>` | Record the current `post.md` and `design.md` in `index.html` (the agent runs this after rendering). |
| `python scripts/blog.py check` | Validate front matter, each page's SEO tags against it, stale renders, the shared navbar, social cards. Runs in CI before a deploy. |
| `python scripts/blog.py sync-chrome` | Re-stamp the navbar and footer from `_shared/` into every post and into the landing and Research pages. |
| `python scripts/blog.py build --out site [--drafts] [--page-size N]` | Assemble `/blog/` and the root `sitemap.xml`, `robots.txt` and `llms.txt` under `site/`. |

## Previewing

```bash
python scripts/blog.py build --out /tmp/autorubric-site --drafts
```

```bash
python3 -m http.server -d /tmp/autorubric-site 8001
```

Then open `http://localhost:8001/blog/`. Pages use root-relative links, so open them
through a server, not from disk. The `autorubric-site` preview configuration in
`.claude/launch.json` builds the whole site (docs included) the same way.

## Publishing

Remove `draft: true` (or set it to `false`), make sure `check` passes, and merge to
`main`. The docs workflow builds and deploys the site.

## Changing the navbar or footer

Edit `blog/_shared/header.html` or `footer.html`, run `sync-chrome`, and look at a
few posts. The marked regions in every page are rewritten; nothing else is touched.

## What a stale render is

Each `index.html` records the hash of the `post.md` and `design.md` it was rendered
from. If either changes afterwards, `check` reports the post as stale until it is
rendered and stamped again. This is what keeps the page and its sources from drifting.
