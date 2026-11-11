"""Build and check the AutoRubric blog (autorubric.org/blog/).

A post lives in ``blog/posts/<slug>/`` as ``post.md`` (content, with front matter),
``design.md`` (presentation) and ``index.html`` (the page a design/rendering agent
wrote from the two; see ``blog/RENDERING.md``). This script never writes a post's
page. It checks the pages, keeps the shared navbar and footer in step, and generates
everything that follows from front matter alone: the paginated index, tag pages,
the Atom feed, sitemaps, ``robots.txt``, ``llms.txt`` and a Markdown copy of each post.

Usage::

    python scripts/blog.py new <slug>
    python scripts/blog.py stamp <slug>
    python scripts/blog.py check
    python scripts/blog.py sync-chrome
    python scripts/blog.py build --out site [--drafts] [--page-size N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import yaml

REPO = Path(__file__).resolve().parents[1]
BLOG_DIR = REPO / "blog"

# Pages outside the blog whose navbar is stamped from blog/_shared/header.html.
SITE_PAGES = ("docs/landing/index.html", "docs/research/index.html")

REQUIRED_KEYS = ("title", "slug", "date", "version", "description", "tags")
OPTIONAL_KEYS = (
    "updated",
    "author",
    "variables",
    "draft",
    "canonical",
    "image",
    "math",
    "chrome",
)
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
MAX_DESCRIPTION = 160
CARD_SIZE = (1200, 630)

# Files in a post directory that are sources, not part of the served page.
UNPUBLISHED = ("post.md", "design.md", "card.html", "src")

SOURCE_META = "autorubric:source"
# Head values a post's design.md may declare it changed from the front-matter default.
DEVIATION_KEYS = ("title", "description", "og:title", "og:description")

AI_CRAWLERS = (
    "GPTBot",
    "OAI-SearchBot",
    "ChatGPT-User",
    "ClaudeBot",
    "Claude-User",
    "Claude-SearchBot",
    "PerplexityBot",
    "Perplexity-User",
    "Google-Extended",
    "Applebot-Extended",
    "CCBot",
)

EMPTY_STATE = '                <p class="empty">No posts yet.</p>'


class BlogError(Exception):
    """A problem with the blog's sources that the author has to fix."""


# --------------------------------------------------------------------------- posts


@dataclass
class Post:
    """One post: its front matter, its Markdown body and where it lives."""

    directory: Path
    meta: dict[str, Any]
    body: str
    site: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def slug(self) -> str:
        return self.meta["slug"]

    @property
    def title(self) -> str:
        return self.meta["title"]

    @property
    def description(self) -> str:
        return self.meta["description"]

    @property
    def date(self) -> dt.date:
        return self.meta["date"]

    @property
    def updated(self) -> dt.date:
        return self.meta.get("updated", self.date)

    @property
    def version(self) -> str:
        return str(self.meta["version"])

    @property
    def tags(self) -> list[str]:
        return self.meta["tags"]

    @property
    def draft(self) -> bool:
        return self.meta.get("draft", False)

    @property
    def authors(self) -> list[str]:
        author = self.meta.get("author", self.site.get("author", ""))
        return [author] if isinstance(author, str) else list(author)

    @property
    def path(self) -> str:
        return f"/blog/{self.slug}/"

    @property
    def url(self) -> str:
        return self.meta.get("canonical") or self.site["url"] + self.path

    @property
    def markdown_url(self) -> str:
        return self.site["url"] + self.path + "index.md"

    @property
    def image_url(self) -> str:
        image = self.meta.get("image", "card.png")
        if re.match(r"https?://", image):
            return image
        return self.site["url"] + self.path + image

    def context(self) -> dict[str, str]:
        """The names a post's body may use as ``{{ name }}``."""
        site = self.site
        names = {
            "site.title": site["title"],
            "site.url": site["url"],
            "site.description": site["description"],
            "site.author": site.get("author", ""),
            "post.title": self.title,
            "post.slug": self.slug,
            "post.description": self.description,
            "post.date": self.date.isoformat(),
            "post.updated": self.updated.isoformat(),
            "post.version": self.version,
            "post.url": self.url,
            "post.author": ", ".join(self.authors),
        }
        for name, value in self.meta.get("variables", {}).items():
            names[name] = str(value)
        return names

    def resolved_body(self) -> str:
        return resolve_variables(self.body, self.context())


def load_site(blog_dir: Path = BLOG_DIR) -> dict[str, Any]:
    site = yaml.safe_load((blog_dir / "site.yml").read_text(encoding="utf-8"))
    site["url"] = site["url"].rstrip("/")
    return site


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``post.md`` into its front matter and its Markdown body."""
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n?(.*)", text, re.DOTALL)
    if match is None:
        raise BlogError("no front matter: the file must start with a '---' block")
    meta = yaml.safe_load(match.group(1))
    if not isinstance(meta, dict):
        raise BlogError("front matter is not a mapping")
    return meta, match.group(2)


def front_matter_problems(meta: dict[str, Any], directory_name: str) -> list[str]:
    """What is wrong with a post's front matter, as messages; empty when it is valid."""
    problems = [f"missing required key '{key}'" for key in REQUIRED_KEYS if key not in meta]
    known = set(REQUIRED_KEYS) | set(OPTIONAL_KEYS)
    problems += [f"unknown key '{key}'" for key in meta if key not in known]

    def check(key: str, ok: bool, expected: str) -> None:
        if key in meta and not ok:
            problems.append(f"'{key}' must be {expected} (got {meta[key]!r})")

    def is_text(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def is_date(value: Any) -> bool:
        return isinstance(value, dt.date) and not isinstance(value, dt.datetime)

    slug = meta.get("slug")
    check("title", is_text(meta.get("title")), "a non-empty string")
    check(
        "slug", isinstance(slug, str) and SLUG.fullmatch(slug) is not None, "lowercase-with-dashes"
    )
    if isinstance(slug, str) and slug != directory_name:
        problems.append(f"'slug' is '{slug}' but the post directory is '{directory_name}'")
    check("date", is_date(meta.get("date")), "a date written YYYY-MM-DD")
    check("updated", is_date(meta.get("updated")), "a date written YYYY-MM-DD")
    if (
        is_date(meta.get("date"))
        and is_date(meta.get("updated"))
        and meta["updated"] < meta["date"]
    ):
        problems.append("'updated' is earlier than 'date'")
    version = meta.get("version")
    check(
        "version",
        isinstance(version, int | float | str)
        and not isinstance(version, bool)
        and str(version) != "",
        "a number or a string",
    )
    description = meta.get("description")
    check("description", is_text(description), "a non-empty string")
    if isinstance(description, str) and len(description) > MAX_DESCRIPTION:
        problems.append(
            f"'description' is {len(description)} characters; search results cut it "
            f"after about {MAX_DESCRIPTION}"
        )
    tags = meta.get("tags")
    check(
        "tags",
        isinstance(tags, list)
        and bool(tags)
        and all(isinstance(tag, str) and SLUG.fullmatch(tag) for tag in tags)
        and len(set(tags)) == len(tags),
        "a non-empty list of distinct lowercase-with-dashes tags",
    )
    author = meta.get("author")
    check(
        "author",
        is_text(author)
        or (isinstance(author, list) and bool(author) and all(map(is_text, author))),
        "a name or a list of names",
    )
    variables = meta.get("variables")
    check(
        "variables",
        isinstance(variables, dict)
        and all(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z_]\w*", name)
            and isinstance(value, str | int | float)
            for name, value in variables.items()
        ),
        "a mapping from names (letters, digits, underscores) to strings or numbers",
    )
    check("draft", isinstance(meta.get("draft"), bool), "true or false")
    check("math", isinstance(meta.get("math"), bool), "true or false")
    canonical = meta.get("canonical")
    check(
        "canonical",
        isinstance(canonical, str) and re.match(r"https?://", canonical) is not None,
        "an absolute http(s) URL",
    )
    check("image", is_text(meta.get("image")), "a file name in the post directory or a URL")
    check("chrome", meta.get("chrome") in ("shared", "none"), "'shared' or 'none'")
    return problems


def load_post(directory: Path, site: dict[str, Any]) -> Post:
    """Read and validate one post directory."""
    source = directory / "post.md"
    try:
        meta, body = split_front_matter(source.read_text(encoding="utf-8"))
    except BlogError as error:
        raise BlogError(f"{source}: {error}") from error
    problems = front_matter_problems(meta, directory.name)
    if problems:
        raise BlogError(f"{source}: " + "; ".join(problems))
    post = Post(directory=directory, meta=meta, body=body, site=site)
    try:
        post.resolved_body()
    except BlogError as error:
        raise BlogError(f"{source}: {error}") from error
    return post


def load_posts(blog_dir: Path = BLOG_DIR, *, drafts: bool = True) -> list[Post]:
    """Every post, newest first; drafts only when asked for."""
    site = load_site(blog_dir)
    posts_dir = blog_dir / "posts"
    directories = sorted(d for d in posts_dir.iterdir() if d.is_dir()) if posts_dir.is_dir() else []
    posts = [load_post(d, site) for d in directories if (d / "post.md").is_file()]
    if not drafts:
        posts = [post for post in posts if not post.draft]
    return sorted(posts, key=lambda post: (-post.date.toordinal(), post.slug))


# ----------------------------------------------------------------------- variables

# Code and math are skipped: LaTeX such as \frac{{a}}{b} is not a variable.
_VARIABLE_OR_VERBATIM = re.compile(
    r"(?P<fence>^(?P<ticks>`{3,}|~{3,})[^\n]*\n.*?^(?P=ticks)[ \t]*$)"
    r"|(?P<code>`+[^`\n]+?`+)"
    r"|(?P<display>\$\$.*?\$\$)"
    r"|(?P<inline>(?<![\\$])\$(?!\s)[^$\n]+?(?<![\\\s])\$)"
    r"|\{\{\s*(?P<name>[A-Za-z_][\w.]*)\s*\}\}",
    re.DOTALL | re.MULTILINE,
)


def resolve_variables(text: str, names: dict[str, str]) -> str:
    """Replace each ``{{ name }}`` outside code and math with its value."""

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name is None:
            return match.group(0)
        if name not in names:
            raise BlogError(f"undefined variable '{{{{ {name} }}}}'")
        return names[name]

    return _VARIABLE_OR_VERBATIM.sub(replace, text)


def fill(template: str, values: dict[str, str]) -> str:
    """Fill a shared template's ``{{ name }}`` placeholders."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise BlogError(f"template uses unknown placeholder '{{{{ {name} }}}}'")
        return values[name]

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", replace, template)


# -------------------------------------------------------------------------- chrome

_REGION = re.compile(
    r"(?P<indent>[ \t]*)<!-- blog:(?P<part>header|footer):start(?: current=(?P<current>\S+))? -->\n"
    r".*?"
    r"(?P=indent)<!-- blog:(?P=part):end -->",
    re.DOTALL,
)


def chrome_html(part: str, blog_dir: Path = BLOG_DIR, *, current: str | None = None) -> str:
    """The shared header or footer, with ``aria-current`` on the link to ``current``."""
    text = (blog_dir / "_shared" / f"{part}.html").read_text(encoding="utf-8").rstrip("\n")
    if current is not None:
        link = f'<a href="{current}">'
        if link not in text:
            raise BlogError(f"no link to {current} in blog/_shared/{part}.html")
        text = text.replace(link, f'<a href="{current}" aria-current="page">', 1)
    return text


def indented(text: str, indent: str) -> str:
    return "\n".join(indent + line if line else line for line in text.split("\n"))


def stamp_chrome(page: str, blog_dir: Path = BLOG_DIR) -> str:
    """Rewrite each marked header or footer region of a page from the shared partials."""

    def replace(match: re.Match[str]) -> str:
        indent, part, current = match.group("indent", "part", "current")
        opening = f"<!-- blog:{part}:start" + (f" current={current}" if current else "") + " -->"
        body = indented(chrome_html(part, blog_dir, current=current), indent)
        return f"{indent}{opening}\n{body}\n{indent}<!-- blog:{part}:end -->"

    return _REGION.sub(replace, page)


def chrome_targets(blog_dir: Path = BLOG_DIR, repo: Path = REPO) -> list[Path]:
    posts_dir = blog_dir / "posts"
    pages = sorted(posts_dir.glob("*/index.html")) if posts_dir.is_dir() else []
    return pages + [repo / page for page in SITE_PAGES if (repo / page).is_file()]


def sync_chrome(blog_dir: Path = BLOG_DIR, repo: Path = REPO, *, write: bool = True) -> list[Path]:
    """Bring every marked region up to date; returns the pages that were out of date."""
    stale = []
    for page in chrome_targets(blog_dir, repo):
        before = page.read_text(encoding="utf-8")
        after = stamp_chrome(before, blog_dir)
        if after != before:
            stale.append(page)
            if write:
                page.write_text(after, encoding="utf-8")
    return stale


# ---------------------------------------------------------------------- provenance


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_stamp(post: Post) -> str:
    """What a rendered page records about the sources it was rendered from."""
    return (
        f"version={post.version}; "
        f"post=sha256:{sha256(post.directory / 'post.md')}; "
        f"design=sha256:{sha256(post.directory / 'design.md')}"
    )


_SOURCE_META = re.compile(rf'<meta name="{SOURCE_META}" content="[^"]*">')


def stamp_source(post: Post) -> None:
    """Record the current sources in the post's rendered page."""
    page = post.directory / "index.html"
    text = page.read_text(encoding="utf-8")
    tag = f'<meta name="{SOURCE_META}" content="{source_stamp(post)}">'
    if _SOURCE_META.search(text):
        text = _SOURCE_META.sub(tag, text, count=1)
    elif "</title>" in text:
        text = text.replace("</title>", f"</title>\n    {tag}", 1)
    else:
        raise BlogError(f"{page}: no <title> to place the source record after")
    page.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- check


class _Page(HTMLParser):
    """The parts of a rendered page that ``check`` reads."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang: str | None = None
        self.title = ""
        self.meta: dict[str, str] = {}
        self.links: list[dict[str, str | None]] = []
        self.json_ld: list[str] = []
        self.h1 = 0
        self.images_without_alt: list[str] = []
        self.images_without_size: list[str] = []
        self.hrefs: set[str] = set()
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "html":
            self.lang = attributes.get("lang")
        elif tag == "title" and not self.title:
            self._capture = "title"
        elif tag == "meta":
            key = attributes.get("name") or attributes.get("property")
            if key:
                self.meta[key] = attributes.get("content") or ""
        elif tag == "link":
            self.links.append(attributes)
        elif tag == "script" and attributes.get("type") == "application/ld+json":
            self._capture = "json_ld"
            self.json_ld.append("")
        elif tag == "h1":
            self.h1 += 1
        elif tag == "img":
            src = attributes.get("src") or "?"
            if "alt" not in attributes:
                self.images_without_alt.append(src)
            if "width" not in attributes or "height" not in attributes:
                self.images_without_size.append(src)
        elif tag == "a" and attributes.get("href"):
            self.hrefs.add(attributes["href"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("title", "script"):
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture == "title":
            self.title += data
        elif self._capture == "json_ld":
            self.json_ld[-1] += data


def declared_deviations(design: str) -> set[str]:
    """Head values design.md's "Deviations" section says differ from the front matter.

    A declaration is a list item that starts with the key in backticks, such as
    ``- `title`: shortened``; a key mentioned in running text declares nothing.
    """
    section = re.search(r"^## Deviations[^\n]*\n(.*?)(?=^## |\Z)", design, re.DOTALL | re.MULTILINE)
    if section is None:
        return set()
    declared = re.findall(r"^[-*] `([^`]+)`", section.group(1), re.MULTILINE)
    return {key for key in declared if key in DEVIATION_KEYS}


def png_size(path: Path) -> tuple[int, int] | None:
    head = path.read_bytes()[:24]
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def _json_ld_nodes(blocks: list[str], problems: list[str]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError as error:
            problems.append(f"a JSON-LD block is not valid JSON ({error})")
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        nodes += [item for item in items if isinstance(item, dict)]
    return nodes


def post_problems(post: Post, blog_dir: Path = BLOG_DIR) -> list[str]:
    """What is wrong with a post's rendered page, as messages; empty when it is sound."""
    directory = post.directory
    design_path = directory / "design.md"
    page_path = directory / "index.html"
    if not design_path.is_file():
        return ["design.md is missing"]
    if not page_path.is_file():
        return ["index.html is missing: the post has not been rendered"]

    text = page_path.read_text(encoding="utf-8")
    page = _Page()
    page.feed(text)
    problems: list[str] = []
    deviations = declared_deviations(design_path.read_text(encoding="utf-8"))

    def expect(label: str, actual: str | None, expected: str) -> None:
        if actual != expected:
            problems.append(f"{label} is {actual!r}; expected {expected!r}")

    def expect_default(
        key: str, actual: str | None, expected: str, *, contains: bool = False
    ) -> None:
        if key in deviations:
            if not actual:
                problems.append(f"{key} is empty")
            return
        ok = expected in (actual or "") if contains else actual == expected
        if not ok:
            problems.append(
                f"{key} is {actual!r}; the front matter gives {expected!r} "
                f"(to keep it, add the list item '- `{key}`: reason' under '## Deviations' in design.md)"
            )

    # Provenance: was the page rendered from the sources as they are now?
    recorded = page.meta.get(SOURCE_META)
    if recorded is None:
        problems.append(
            f"no <meta name=\"{SOURCE_META}\">: run 'blog.py stamp {post.slug}' after rendering"
        )
    elif recorded != source_stamp(post):
        problems.append(
            "stale render: post.md or design.md changed since index.html was rendered; "
            f"render it again, then run 'blog.py stamp {post.slug}'"
        )

    # Head.
    if not page.lang:
        problems.append("<html> has no lang attribute")
    expect_default("title", page.title.strip(), post.title, contains=True)
    expect_default("description", page.meta.get("description"), post.description)
    canonical = [link.get("href") for link in page.links if link.get("rel") == "canonical"]
    expect("canonical link", canonical[0] if canonical else None, post.url)
    expect("og:type", page.meta.get("og:type"), "article")
    expect("og:url", page.meta.get("og:url"), post.url)
    expect_default("og:title", page.meta.get("og:title"), post.title)
    expect_default("og:description", page.meta.get("og:description"), post.description)
    expect("og:image", page.meta.get("og:image"), post.image_url)
    expect("twitter:card", page.meta.get("twitter:card"), "summary_large_image")
    expect("twitter:image", page.meta.get("twitter:image"), post.image_url)
    expect("article:published_time", page.meta.get("article:published_time"), post.date.isoformat())
    for key in ("og:site_name", "og:image:alt", "twitter:title", "twitter:description"):
        if not page.meta.get(key):
            problems.append(f"{key} is missing")

    alternates = {
        link.get("type"): link.get("href") for link in page.links if link.get("rel") == "alternate"
    }
    if alternates.get("text/markdown") not in (
        post.markdown_url,
        post.path + "index.md",
        "index.md",
    ):
        problems.append(f'no <link rel="alternate" type="text/markdown"> to {post.path}index.md')
    if alternates.get("application/atom+xml") not in (
        "/blog/feed.xml",
        post.site["url"] + "/blog/feed.xml",
    ):
        problems.append('no <link rel="alternate" type="application/atom+xml"> to /blog/feed.xml')

    # Structured data.
    nodes = _json_ld_nodes(page.json_ld, problems)
    postings = [node for node in nodes if node.get("@type") == "BlogPosting"]
    if len(postings) != 1:
        problems.append(f"expected one BlogPosting JSON-LD node, found {len(postings)}")
    else:
        posting = postings[0]
        for key in ("headline", "description", "author", "image", "mainEntityOfPage", "publisher"):
            if not posting.get(key):
                problems.append(f"BlogPosting JSON-LD has no '{key}'")
        expect("BlogPosting datePublished", posting.get("datePublished"), post.date.isoformat())
        expect("BlogPosting dateModified", posting.get("dateModified"), post.updated.isoformat())
        expect("BlogPosting url", posting.get("url"), post.url)
    if not any(node.get("@type") == "BreadcrumbList" for node in nodes):
        problems.append("no BreadcrumbList JSON-LD node")

    # Body.
    if page.h1 != 1:
        problems.append(f"expected exactly one <h1>, found {page.h1}")
    for src in page.images_without_alt:
        problems.append(f'<img src="{src}"> has no alt attribute')
    for src in page.images_without_size:
        problems.append(f'<img src="{src}"> has no width and height (the page shifts as it loads)')
    for href, name in (("/", "the home page"), ("/blog/", "the blog index")):
        if href not in page.hrefs:
            problems.append(f'no link to {name} (<a href="{href}">)')

    # Shared navbar and footer.
    regions = {match.group("part") for match in _REGION.finditer(text)}
    if post.meta.get("chrome", "shared") == "shared":
        for part in ("header", "footer"):
            if part not in regions:
                problems.append(
                    f"no '<!-- blog:{part}:start -->' region (set 'chrome: none' in the "
                    "front matter if the post has its own)"
                )
        if stamp_chrome(text, blog_dir) != text:
            problems.append("navbar or footer differs from blog/_shared: run 'blog.py sync-chrome'")

    # Social card.
    if "image" not in post.meta:
        card = directory / "card.png"
        if not card.is_file():
            problems.append("card.png is missing")
        elif png_size(card) != CARD_SIZE:
            problems.append(
                f"card.png is {png_size(card)}; a social card is {CARD_SIZE[0]}x{CARD_SIZE[1]}"
            )
    return problems


def check(blog_dir: Path = BLOG_DIR, repo: Path = REPO) -> list[str]:
    """Every problem in the blog, each prefixed with where it is."""
    problems: list[str] = []
    site = load_site(blog_dir)
    posts_dir = blog_dir / "posts"
    directories = sorted(d for d in posts_dir.iterdir() if d.is_dir()) if posts_dir.is_dir() else []
    for directory in directories:
        try:
            post = load_post(directory, site)
        except (BlogError, OSError, yaml.YAMLError) as error:
            problems.append(str(error))
            continue
        problems += [f"{directory.name}: {problem}" for problem in post_problems(post, blog_dir)]
    for page in SITE_PAGES:
        path = repo / page
        if path.is_file() and stamp_chrome(
            path.read_text(encoding="utf-8"), blog_dir
        ) != path.read_text(encoding="utf-8"):
            problems.append(f"{page}: navbar differs from blog/_shared: run 'blog.py sync-chrome'")
    return problems


# --------------------------------------------------------------------------- build


def human_date(date: dt.date) -> str:
    return f"{date:%B} {date.day}, {date.year}"


def pages_of(posts: list[Post], page_size: int) -> list[list[Post]]:
    """Posts split into listing pages; an empty blog still has one (empty) page."""
    return [posts[i : i + page_size] for i in range(0, len(posts), page_size)] or [[]]


def listing_path(base: str, number: int) -> str:
    """The URL path of a listing's page ``number`` (1-based)."""
    return base if number == 1 else f"{base}page/{number}/"


def entry_html(post: Post, template: str) -> str:
    tags = "\n".join(
        f'                        <li><a class="tag" href="/blog/tags/{tag}/">{html.escape(tag)}</a></li>'
        for tag in post.tags
    )
    title = html.escape(post.title) + (' <span class="tag">Draft</span>' if post.draft else "")
    return fill(
        template,
        {
            "url": post.path,
            "title": title,
            "date_iso": post.date.isoformat(),
            "date_human": human_date(post.date),
            "description": html.escape(post.description),
            "tags": tags,
        },
    ).rstrip("\n")


def pagination_html(base: str, number: int, total: int) -> str:
    if total == 1:
        return ""
    newer = (
        f'<a rel="prev" href="{listing_path(base, number - 1)}">&larr; Newer</a>'
        if number > 1
        else '<a class="spacer" aria-hidden="true" tabindex="-1">&larr; Newer</a>'
    )
    older = (
        f'<a rel="next" href="{listing_path(base, number + 1)}">Older &rarr;</a>'
        if number < total
        else '<a class="spacer" aria-hidden="true" tabindex="-1">Older &rarr;</a>'
    )
    return (
        '                <nav class="pagination" aria-label="Pagination">\n'
        f"                    {newer}\n"
        f"                    <span>Page {number} of {total}</span>\n"
        f"                    {older}\n"
        "                </nav>"
    )


def write_listing(
    out: Path,
    blog_dir: Path,
    site: dict[str, Any],
    posts: list[Post],
    *,
    base: str,
    heading: str,
    tagline: str,
    description: str,
    page_size: int,
) -> list[str]:
    """Write one paginated listing (the index, or one tag); returns its URL paths."""
    shared = blog_dir / "_shared"
    template = (shared / "index.template.html").read_text(encoding="utf-8")
    entry = (shared / "index.entry.html").read_text(encoding="utf-8")
    pages = pages_of(posts, page_size)
    paths = []
    for number, page_posts in enumerate(pages, start=1):
        path = listing_path(base, number)
        canonical = site["url"] + path
        title = site["title"] if base == "/blog/" else f"{heading} — {site['title']}"
        if number > 1:
            title = f"{title} (page {number})"
        rel_links = []
        if number > 1:
            rel_links.append(
                f'    <link rel="prev" href="{site["url"] + listing_path(base, number - 1)}">'
            )
        if number < len(pages):
            rel_links.append(
                f'    <link rel="next" href="{site["url"] + listing_path(base, number + 1)}">'
            )
        json_ld = {
            "@context": "https://schema.org",
            "@type": "Blog" if base == "/blog/" else "CollectionPage",
            "name": title,
            "description": description,
            "url": canonical,
            "inLanguage": site.get("language", "en"),
            "publisher": {"@type": "Organization", "name": "AutoRubric", "url": site["url"] + "/"},
            "blogPost" if base == "/blog/" else "hasPart": [
                {
                    "@type": "BlogPosting",
                    "headline": post.title,
                    "description": post.description,
                    "url": post.url,
                    "datePublished": post.date.isoformat(),
                    "dateModified": post.updated.isoformat(),
                    "author": [{"@type": "Person", "name": name} for name in post.authors],
                    "keywords": post.tags,
                }
                for post in page_posts
            ],
        }
        page = fill(
            template,
            {
                "language": site.get("language", "en"),
                "title": html.escape(title),
                "site_title": html.escape(site["title"]),
                "description": html.escape(description),
                "robots": "index, follow, max-image-preview:large",
                "canonical": canonical,
                "rel_links": "\n".join(rel_links),
                "jsonld": json.dumps(json_ld, indent=2, ensure_ascii=False).replace("</", "<\\/"),
                "header": indented(
                    chrome_html("header", blog_dir, current="/blog/"), "    "
                ).lstrip(),
                "footer": indented(chrome_html("footer", blog_dir), "    ").lstrip(),
                "heading": html.escape(heading),
                "tagline": html.escape(tagline),
                "entries": "\n".join(entry_html(post, entry) for post in page_posts) or EMPTY_STATE,
                "pagination": pagination_html(base, number, len(pages)),
            },
        )
        target = out / path.strip("/") / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(page, encoding="utf-8")
        paths.append(path)
    return paths


def markdown_copy(post: Post) -> str:
    """The post as plain Markdown, for readers and agents that prefer it to HTML."""
    lines = [
        f"# {post.title}",
        "",
        f"By {', '.join(post.authors)}. Published {post.date.isoformat()}"
        + (f", updated {post.updated.isoformat()}" if post.updated != post.date else "")
        + f". Tags: {', '.join(post.tags)}.",
        f"Canonical URL: {post.url}",
        "",
        f"> {post.description}",
        "",
        "",
    ]
    body = re.sub(r"[ \t]*<!--\s*design:.*?-->[ \t]*\n?", "", post.resolved_body(), flags=re.DOTALL)
    return "\n".join(lines) + body.lstrip("\n")


def atom_feed(site: dict[str, Any], posts: list[Post]) -> str:
    atom = "http://www.w3.org/2005/Atom"
    ET.register_namespace("", atom)

    def element(
        parent: ET.Element, tag: str, text: str | None = None, **attributes: str
    ) -> ET.Element:
        child = ET.SubElement(parent, f"{{{atom}}}{tag}", attributes)
        child.text = text
        return child

    def timestamp(date: dt.date) -> str:
        return f"{date.isoformat()}T00:00:00Z"

    feed = ET.Element(f"{{{atom}}}feed")
    element(feed, "title", site["title"])
    element(feed, "subtitle", site["description"])
    element(feed, "id", site["url"] + "/blog/")
    element(feed, "link", href=site["url"] + "/blog/", rel="alternate", type="text/html")
    element(
        feed, "link", href=site["url"] + "/blog/feed.xml", rel="self", type="application/atom+xml"
    )
    newest = max((post.updated for post in posts), default=dt.date(1970, 1, 1))
    element(feed, "updated", timestamp(newest))
    element(element(feed, "author"), "name", site.get("author", "AutoRubric"))
    for post in posts:
        entry = element(feed, "entry")
        element(entry, "title", post.title)
        element(entry, "id", post.url)
        element(entry, "link", href=post.url, rel="alternate", type="text/html")
        element(entry, "link", href=post.markdown_url, rel="alternate", type="text/markdown")
        element(entry, "published", timestamp(post.date))
        element(entry, "updated", timestamp(post.updated))
        for name in post.authors:
            element(element(entry, "author"), "name", name)
        for tag in post.tags:
            element(entry, "category", term=tag)
        element(entry, "summary", post.description)
    ET.indent(feed)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(feed, encoding="unicode") + "\n"


def sitemap(urls: list[tuple[str, dt.date | None]], *, index: bool = False) -> str:
    """A sitemap of pages, or with ``index`` a sitemap of sitemaps."""
    namespace = "http://www.sitemaps.org/schemas/sitemap/0.9"
    ET.register_namespace("", namespace)
    root = ET.Element(f"{{{namespace}}}{'sitemapindex' if index else 'urlset'}")
    for url, modified in urls:
        item = ET.SubElement(root, f"{{{namespace}}}{'sitemap' if index else 'url'}")
        ET.SubElement(item, f"{{{namespace}}}loc").text = url
        if modified is not None:
            ET.SubElement(item, f"{{{namespace}}}lastmod").text = modified.isoformat()
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def robots_txt(site: dict[str, Any]) -> str:
    lines = [
        "# Everyone may read everything, AI crawlers and agents included.",
        "User-agent: *",
        "Allow: /",
        "",
    ]
    for crawler in AI_CRAWLERS:
        lines += [f"User-agent: {crawler}", "Allow: /", ""]
    lines += [f"Sitemap: {site['url']}/sitemap.xml", ""]
    return "\n".join(lines)


def llms_txt(site: dict[str, Any], posts: list[Post]) -> str:
    """An llms.txt (https://llmstxt.org): what the site is and where its text lives."""
    llms = site.get("llms", {})
    lines = ["# AutoRubric", "", f"> {llms.get('summary', site['description'])}", "", "## Docs", ""]
    lines += [f"- [{doc['title']}]({doc['url']}): {doc['note']}" for doc in llms.get("docs", [])]
    lines += ["", "## Blog", ""]
    lines += [f"- [{post.title}]({post.markdown_url}): {post.description}" for post in posts]
    if not posts:
        lines.append(f"- [{site['title']}]({site['url']}/blog/): {site['description']}")
    lines += [
        "",
        "## Optional",
        "",
        f"- [Blog feed (Atom)]({site['url']}/blog/feed.xml): every post, newest first",
        "",
    ]
    return "\n".join(lines)


def build(
    out: Path,
    blog_dir: Path = BLOG_DIR,
    *,
    drafts: bool = False,
    page_size: int | None = None,
) -> list[Post]:
    """Assemble the blog under ``out`` (the site root); returns the posts published."""
    site = load_site(blog_dir)
    posts = []
    for post in load_posts(blog_dir, drafts=drafts):
        if (post.directory / "index.html").is_file():
            posts.append(post)
        elif post.draft:
            print(f"skipped draft '{post.slug}': it has not been rendered", file=sys.stderr)
        else:
            raise BlogError(f"{post.slug}: index.html is missing: the post has not been rendered")
    size = page_size or site.get("page_size", 10)
    if size < 1:
        raise BlogError("page size must be at least 1")
    blog_out = out / "blog"
    if blog_out.exists():
        shutil.rmtree(blog_out)
    (blog_out / "assets").mkdir(parents=True)
    shutil.copy2(blog_dir / "_shared" / "base.css", blog_out / "assets" / "base.css")

    for post in posts:
        target = blog_out / post.slug
        shutil.copytree(post.directory, target, ignore=shutil.ignore_patterns(*UNPUBLISHED))
        (target / "index.md").write_text(markdown_copy(post), encoding="utf-8")

    listing = {"page_size": size}
    paths = write_listing(
        out,
        blog_dir,
        site,
        posts,
        base="/blog/",
        heading="Blog",
        tagline=site.get("tagline", site["description"]),
        description=site["description"],
        **listing,
    )
    tags = sorted({tag for post in posts for tag in post.tags})
    for tag in tags:
        tagged = [post for post in posts if tag in post.tags]
        paths += write_listing(
            out,
            blog_dir,
            site,
            tagged,
            base=f"/blog/tags/{tag}/",
            heading=f"Posts tagged “{tag}”",
            tagline=f"{len(tagged)} post{'s' if len(tagged) != 1 else ''}. See all posts on the blog index.",
            description=f"AutoRubric blog posts tagged {tag}.",
            **listing,
        )

    (blog_out / "feed.xml").write_text(atom_feed(site, posts), encoding="utf-8")
    newest = max((post.updated for post in posts), default=None)
    blog_urls: list[tuple[str, dt.date | None]] = [(site["url"] + path, newest) for path in paths]
    blog_urls += [(post.url, post.updated) for post in posts]
    (blog_out / "sitemap.xml").write_text(sitemap(blog_urls), encoding="utf-8")
    (out / "sitemap-pages.xml").write_text(
        sitemap([(site["url"] + "/", None), (site["url"] + "/research/", None)]), encoding="utf-8"
    )
    (out / "sitemap.xml").write_text(
        sitemap(
            [
                (site["url"] + "/sitemap-pages.xml", None),
                (site["url"] + "/blog/sitemap.xml", newest),
                (site["url"] + "/docs/sitemap.xml", None),
            ],
            index=True,
        ),
        encoding="utf-8",
    )
    (out / "robots.txt").write_text(robots_txt(site), encoding="utf-8")
    (out / "llms.txt").write_text(llms_txt(site, posts), encoding="utf-8")
    return posts


# ----------------------------------------------------------------------------- new

POST_SKELETON = """---
title: {title}
slug: {slug}
date: {date}
version: 1
description: One or two sentences, at most 160 characters, that say what the post shows.
tags: [evaluation]
author: {author}
draft: true
variables:
  example_model: openai/gpt-5.6-luna
---

Write the post here in Markdown. Do not repeat the title as a heading: the page's
`<h1>` comes from the front matter.

Inline math is written $\\kappa = \\frac{{p_o - p_e}}{{1 - p_e}}$ and display math

$$
\\kappa = \\frac{{p_o - p_e}}{{1 - p_e}}
$$

A front-matter variable: {{{{ example_model }}}}. Variables are not replaced inside code or math.
"""

DESIGN_SKELETON = """# Design: {title}

The presentation record for `post.md`. Together with `blog/RENDERING.md` and
`blog/DESIGN.md` it must be enough for a fresh agent to reproduce `index.html`.

## Intent

What the page should make a reader feel or do, in two or three sentences.

## Layout

Page structure, measures, breakpoints.

## Components

How each construct in `post.md` is presented (headings, code, figures, tables, callouts).

## Libraries

Name, exact version, where it is loaded from, and what it is used for.

## Interactive elements

What each one does, and what a reader sees with JavaScript off.

## Deviations

None.

(List departures from `blog/DESIGN.md` here. A head value that differs from the front
matter needs a list item that starts with its key in backticks and gives the reason;
the keys are title, description, og:title and og:description.)

## Render log

- v1 ({date}): first render.
"""


def new_post(slug: str, blog_dir: Path = BLOG_DIR) -> Path:
    if SLUG.fullmatch(slug) is None:
        raise BlogError(f"'{slug}' is not a slug: use lowercase letters, digits and dashes")
    directory = blog_dir / "posts" / slug
    if directory.exists():
        raise BlogError(f"{directory} already exists")
    site = load_site(blog_dir)
    values = {
        "title": slug.replace("-", " ").capitalize(),
        "slug": slug,
        "date": dt.date.today().isoformat(),
        "author": site.get("author", ""),
    }
    (directory / "assets").mkdir(parents=True)
    (directory / "post.md").write_text(POST_SKELETON.format(**values), encoding="utf-8")
    (directory / "design.md").write_text(DESIGN_SKELETON.format(**values), encoding="utf-8")
    return directory


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and check the AutoRubric blog.")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="scaffold blog/posts/<slug>/ with post.md and design.md")
    new.add_argument("slug")
    stamp = commands.add_parser(
        "stamp", help="record the current post.md and design.md in index.html"
    )
    stamp.add_argument("slug")
    commands.add_parser("check", help="validate front matter, rendered pages and shared chrome")
    sync = commands.add_parser(
        "sync-chrome", help="re-stamp the shared navbar and footer everywhere"
    )
    sync.add_argument(
        "--check", action="store_true", help="report pages out of date; change nothing"
    )
    builder = commands.add_parser("build", help="assemble the blog into a site directory")
    builder.add_argument("--out", type=Path, required=True, help="the site root, e.g. site")
    builder.add_argument("--drafts", action="store_true", help="include posts marked draft: true")
    builder.add_argument("--page-size", type=int, help="posts per index page (default: site.yml)")
    args = parser.parse_args(argv)

    try:
        if args.command == "new":
            print(f"Created {new_post(args.slug).relative_to(REPO)}")
        elif args.command == "stamp":
            post = load_post(BLOG_DIR / "posts" / args.slug, load_site())
            stamp_source(post)
            print(f"Recorded sources in {post.slug}/index.html")
        elif args.command == "check":
            problems = check()
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            if problems:
                print(
                    f"{len(problems)} problem{'s' if len(problems) != 1 else ''}", file=sys.stderr
                )
                return 1
            print("Blog OK")
        elif args.command == "sync-chrome":
            stale = sync_chrome(write=not args.check)
            for page in stale:
                print(
                    ("out of date: " if args.check else "updated: ") + str(page.relative_to(REPO))
                )
            if args.check and stale:
                return 1
            if not stale:
                print("Navbar and footer already in step")
        elif args.command == "build":
            posts = build(args.out, drafts=args.drafts, page_size=args.page_size)
            drafts = sum(post.draft for post in posts)
            print(
                f"Built {len(posts)} post{'s' if len(posts) != 1 else ''} ({drafts} draft) into {args.out}/blog"
            )
    except (BlogError, OSError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
