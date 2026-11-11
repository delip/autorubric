# Design: blog index and tag pages

The presentation record for `index.template.html` and `index.entry.html`.
`scripts/blog.py build` fills them; the HTML and CSS here are the design.

## Intent

A plain reverse-chronological list that reads like the Research page: the reader
scans titles and one-sentence descriptions and clicks through.

## Layout

- Page head as on `/research/`: `h1` at 2.25rem/700 and a 1.25rem gray tagline.
- Entries in one 800px column, left-aligned in the 1200px container, separated by
  1px `--border` rules.
- Each entry: date (0.82rem, gray) → linked title (`h2`, 1.25rem/600, gold on hover)
  → description (1rem, gray) → tag chips.
- Pagination under the list: "← Newer", "Page N of M", "Older →". Page 1 is `/blog/`,
  later pages `/blog/page/N/`. Ten posts per page (`site.yml` `page_size`).
- Tag pages (`/blog/tags/<tag>/`) use the same template with the heading
  "Posts tagged “<tag>”".
- Empty state: the sentence "No posts yet."

## Placeholders

Template: `language`, `title`, `site_title`, `description`, `robots`, `canonical`,
`rel_links`, `jsonld`, `header`, `footer`, `heading`, `tagline`, `entries`, `pagination`.
Entry: `url`, `title`, `date_iso`, `date_human`, `description`, `tags`.

## Deviations

None from `blog/DESIGN.md`.
