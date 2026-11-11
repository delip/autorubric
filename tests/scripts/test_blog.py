"""Tests for ``scripts/blog.py``, which checks and assembles the autorubric.org blog.

The script is not part of the package, so it is loaded from its file. Every test builds
a small blog under ``tmp_path`` from the real shared templates; the repository's own
posts are not read.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "blog.py"


def _load_script() -> ModuleType:
    name = "autorubric_blog_script"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


blog = _load_script()

ATOM = "{http://www.w3.org/2005/Atom}"
SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def _png(path: Path, width: int, height: int) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    pixels = zlib.compress(b"".join(b"\x00" + b"\x00" * width for _ in range(height)))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def _front_matter(slug: str, /, **overrides: object) -> str:
    meta: dict[str, object] = {
        "title": f"Post {slug}",
        "slug": slug,
        "date": "2026-01-01",
        "version": 1,
        "description": f"What {slug} shows.",
        "tags": ["metrics"],
    }
    meta.update(overrides)
    lines = [
        f"{key}: {json.dumps(value) if not isinstance(value, str) else value}"
        for key, value in meta.items()
    ]
    return "---\n" + "\n".join(lines) + "\n---\n"


def _rendered_page(module: ModuleType, slug: str, title: str, description: str, date: str) -> str:
    """A page filled from the real skeleton the way the rendering contract asks."""
    page = (REPO / "blog" / "_shared" / "post.skeleton.html").read_text(encoding="utf-8")
    for old, new in (
        ("POST TITLE", title),
        ("POST DESCRIPTION", description),
        ("SLUG", slug),
        ("YYYY-MM-DD", date),
        ("WHAT THE CARD SHOWS", "A card."),
        ("AUTHOR", "Ada Author"),
    ):
        page = page.replace(old, new)
    return page


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """An empty blog with the real shared files, plus stand-ins for the two site pages."""
    blog_dir = tmp_path / "blog"
    shutil.copytree(REPO / "blog" / "_shared", blog_dir / "_shared")
    shutil.copy2(REPO / "blog" / "site.yml", blog_dir / "site.yml")
    (blog_dir / "posts").mkdir()
    return blog_dir


def _add_post(
    blog_dir: Path,
    slug: str,
    *,
    body: str = "Some text.\n",
    rendered: bool = True,
    **front_matter: object,
) -> Path:
    directory = blog_dir / "posts" / slug
    directory.mkdir()
    (directory / "post.md").write_text(_front_matter(slug, **front_matter) + body, encoding="utf-8")
    (directory / "design.md").write_text("# Design\n\n## Deviations\n\nNone.\n", encoding="utf-8")
    if rendered:
        post = blog.load_post(directory, blog.load_site(blog_dir))
        page = _rendered_page(blog, slug, post.title, post.description, post.date.isoformat())
        (directory / "index.html").write_text(blog.stamp_chrome(page, blog_dir), encoding="utf-8")
        _png(directory / "card.png", 1200, 630)
        blog.stamp_source(post)
    return directory


def _problems(blog_dir: Path, slug: str) -> list[str]:
    post = blog.load_post(blog_dir / "posts" / slug, blog.load_site(blog_dir))
    return blog.post_problems(post, blog_dir)


class TestFrontMatter:
    def test_a_complete_post_loads(self, site: Path) -> None:
        directory = _add_post(site, "kappa", tags=["metrics", "agreement"], updated="2026-02-01")
        post = blog.load_post(directory, blog.load_site(site))
        assert post.date == dt.date(2026, 1, 1)
        assert post.updated == dt.date(2026, 2, 1)
        assert post.url == "https://autorubric.org/blog/kappa/"
        assert post.image_url == "https://autorubric.org/blog/kappa/card.png"
        assert post.authors == ["Delip Rao"]

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"slug": "Not A Slug"}, "'slug' must be lowercase-with-dashes"),
            ({"slug": "other"}, "post directory is 'kappa'"),
            ({"date": "soon"}, "'date' must be a date"),
            ({"tags": []}, "'tags' must be"),
            ({"tags": ["Metrics"]}, "'tags' must be"),
            ({"description": "x" * 161}, "161 characters"),
            ({"updated": "2025-12-31"}, "'updated' is earlier than 'date'"),
            ({"chrome": "custom"}, "'chrome' must be"),
            ({"draft": "maybe"}, "'draft' must be"),
            ({"tittle": "typo"}, "unknown key 'tittle'"),
        ],
    )
    def test_a_bad_value_is_named(self, site: Path, overrides: dict, message: str) -> None:
        directory = site / "posts" / "kappa"
        directory.mkdir()
        (directory / "post.md").write_text(_front_matter("kappa", **overrides), encoding="utf-8")
        with pytest.raises(blog.BlogError, match=message):
            blog.load_post(directory, blog.load_site(site))

    def test_a_missing_key_is_named(self, site: Path) -> None:
        directory = site / "posts" / "kappa"
        directory.mkdir()
        (directory / "post.md").write_text("---\ntitle: T\nslug: kappa\n---\n", encoding="utf-8")
        with pytest.raises(blog.BlogError, match="missing required key 'date'"):
            blog.load_post(directory, blog.load_site(site))

    def test_a_file_without_front_matter_is_refused(self, site: Path) -> None:
        directory = site / "posts" / "kappa"
        directory.mkdir()
        (directory / "post.md").write_text("Just text.\n", encoding="utf-8")
        with pytest.raises(blog.BlogError, match="no front matter"):
            blog.load_post(directory, blog.load_site(site))


class TestVariables:
    names = {"model": "luna", "post.title": "T"}

    def test_names_are_replaced(self) -> None:
        text = "Use {{ model }} for {{post.title}}."
        assert blog.resolve_variables(text, self.names) == "Use luna for T."

    def test_an_undefined_name_is_an_error(self) -> None:
        with pytest.raises(blog.BlogError, match="undefined variable '{{ nope }}'"):
            blog.resolve_variables("{{ nope }}", self.names)

    @pytest.mark.parametrize(
        "verbatim",
        [
            "`{{ model }}`",
            "```python\nx = '{{ model }}'\n```",
            "$\\frac{{model}}{b}$",
            "$$\n\\frac{{model}}{b}\n$$",
        ],
    )
    def test_code_and_math_are_left_alone(self, verbatim: str) -> None:
        text = f"{verbatim}\n\nthen {{{{ model }}}}"
        assert blog.resolve_variables(text, self.names) == f"{verbatim}\n\nthen luna"

    def test_a_dollar_amount_does_not_start_math(self) -> None:
        text = "It cost $5 with {{ model }} and $6 without."
        assert blog.resolve_variables(text, self.names) == "It cost $5 with luna and $6 without."

    def test_an_undefined_variable_fails_the_post(self, site: Path) -> None:
        directory = _add_post(site, "kappa", rendered=False, body="{{ missing }}\n")
        with pytest.raises(blog.BlogError, match="undefined variable"):
            blog.load_post(directory, blog.load_site(site))

    def test_the_markdown_copy_has_variables_resolved(self, site: Path) -> None:
        directory = _add_post(
            site, "kappa", body="Judge: {{ judge }}.\n", variables={"judge": "luna"}
        )
        copy = blog.markdown_copy(blog.load_post(directory, blog.load_site(site)))
        assert copy.startswith("# Post kappa\n")
        assert "\n> What kappa shows.\n\nJudge: luna.\n" in copy
        assert "Judge: luna." in copy
        assert "Canonical URL: https://autorubric.org/blog/kappa/" in copy


class TestChrome:
    def test_stamping_fills_the_regions_and_marks_the_current_link(self, site: Path) -> None:
        region = "  <!-- blog:header:start current=/blog/ -->\n  <!-- blog:header:end -->\n"
        page = f"<body>\n{region}</body>\n"
        stamped = blog.stamp_chrome(page, site)
        assert '\n  <header>\n      <div class="container">' in stamped
        assert '<a href="/blog/" aria-current="page">Blog</a>' in stamped
        assert stamped.index("Blog</a>") < stamped.index("Docs</a>")
        assert stamped.endswith("  <!-- blog:header:end -->\n</body>\n")

    def test_stamping_twice_changes_nothing(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        page = (directory / "index.html").read_text(encoding="utf-8")
        assert blog.stamp_chrome(page, site) == page

    def test_a_changed_partial_reaches_only_the_marked_regions(
        self, site: Path, tmp_path: Path
    ) -> None:
        directory = _add_post(site, "kappa")
        before = (directory / "index.html").read_text(encoding="utf-8")
        header = site / "_shared" / "header.html"
        header.write_text(
            header.read_text(encoding="utf-8").replace(">PyPI<", ">Package<"), encoding="utf-8"
        )

        assert blog.sync_chrome(site, tmp_path, write=False) == [directory / "index.html"]
        assert (directory / "index.html").read_text(encoding="utf-8") == before
        assert blog.sync_chrome(site, tmp_path) == [directory / "index.html"]
        after = (directory / "index.html").read_text(encoding="utf-8")
        assert after == before.replace(">PyPI<", ">Package<", 1)
        assert blog.sync_chrome(site, tmp_path) == []

    def test_an_unknown_current_link_is_an_error(self, site: Path) -> None:
        page = "<!-- blog:header:start current=/nowhere/ -->\n<!-- blog:header:end -->"
        with pytest.raises(blog.BlogError, match="no link to /nowhere/"):
            blog.stamp_chrome(page, site)


class TestCheck:
    def test_a_page_filled_from_the_skeleton_is_sound(self, site: Path) -> None:
        _add_post(site, "kappa")
        assert _problems(site, "kappa") == []
        assert blog.check(site, site.parent) == []

    def test_an_unrendered_post_is_reported(self, site: Path) -> None:
        _add_post(site, "kappa", rendered=False)
        assert _problems(site, "kappa") == ["index.html is missing: the post has not been rendered"]

    @pytest.mark.parametrize("source", ["post.md", "design.md"])
    def test_a_source_edited_after_rendering_makes_the_page_stale(
        self, site: Path, source: str
    ) -> None:
        directory = _add_post(site, "kappa")
        with (directory / source).open("a", encoding="utf-8") as file:
            file.write("\nOne more sentence.\n")
        assert any("stale render" in problem for problem in _problems(site, "kappa"))
        blog.stamp_source(blog.load_post(directory, blog.load_site(site)))
        assert _problems(site, "kappa") == []

    def test_a_head_value_that_differs_from_the_front_matter_is_reported(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        page = directory / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                '<meta name="description" content="What kappa shows.">',
                '<meta name="description" content="Something else.">',
            ),
            encoding="utf-8",
        )
        (problem,) = _problems(site, "kappa")
        assert "description is 'Something else.'" in problem and "Deviations" in problem

    def test_a_declared_deviation_is_accepted(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        page = directory / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                '<meta name="description" content="What kappa shows.">',
                '<meta name="description" content="Something else.">',
            ),
            encoding="utf-8",
        )
        (directory / "design.md").write_text(
            "# Design\n\n## Deviations\n\n- `description`: reworded.\n\n## Render log\n",
            encoding="utf-8",
        )
        blog.stamp_source(blog.load_post(directory, blog.load_site(site)))
        assert _problems(site, "kappa") == []

    def test_a_key_mentioned_in_running_text_declares_nothing(self) -> None:
        design = "## Deviations\n\nNone. A changed `title` would be listed here.\n\n## Render log\n"
        assert blog.declared_deviations(design) == set()
        assert blog.declared_deviations(
            "## Deviations\n\n- `title`: shorter.\n* `og:title`: same.\n"
        ) == {
            "title",
            "og:title",
        }

    @pytest.mark.parametrize(
        ("old", "new", "message"),
        [
            ('content="article"', 'content="website"', "og:type"),
            (
                "https://autorubric.org/blog/kappa/card.png",
                "https://autorubric.org/x.png",
                "og:image",
            ),
            ('"datePublished": "2026-01-01"', '"datePublished": "2025-01-01"', "datePublished"),
            ('"@type": "BreadcrumbList"', '"@type": "Thing"', "BreadcrumbList"),
            ('type="text/markdown"', 'type="text/plain"', "text/markdown"),
            ("<h1>", "<h1>Again</h1><h1>", "exactly one <h1>"),
            ('<html lang="en">', "<html>", "lang"),
            ("summary_large_image", "summary", "twitter:card"),
        ],
    )
    def test_a_missing_or_wrong_seo_tag_is_reported(
        self, site: Path, old: str, new: str, message: str
    ) -> None:
        directory = _add_post(site, "kappa")
        page = directory / "index.html"
        text = page.read_text(encoding="utf-8")
        assert old in text
        page.write_text(text.replace(old, new), encoding="utf-8")
        assert any(message in problem for problem in _problems(site, "kappa"))

    def test_an_image_without_alt_is_reported(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        page = directory / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                "<!-- The post body. -->", '<img src="a.png">'
            ),
            encoding="utf-8",
        )
        assert any(
            '<img src="a.png"> has no alt' in problem for problem in _problems(site, "kappa")
        )

    def test_an_image_without_dimensions_is_reported(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        page = directory / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(
                "<!-- The post body. -->", '<img src="a.png" alt="A chart.">'
            ),
            encoding="utf-8",
        )
        assert any("has no width and height" in problem for problem in _problems(site, "kappa"))

    def test_a_card_of_the_wrong_size_is_reported(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        _png(directory / "card.png", 800, 600)
        assert any("card.png is (800, 600)" in problem for problem in _problems(site, "kappa"))
        (directory / "card.png").unlink()
        assert "card.png is missing" in _problems(site, "kappa")

    def test_an_edited_navbar_region_is_reported(self, site: Path) -> None:
        directory = _add_post(site, "kappa")
        page = directory / "index.html"
        page.write_text(
            page.read_text(encoding="utf-8").replace(">Cookbook<", ">Recipes<"), encoding="utf-8"
        )
        assert any("sync-chrome" in problem for problem in _problems(site, "kappa"))

    def test_a_post_with_its_own_chrome_needs_no_regions(self, site: Path) -> None:
        directory = _add_post(site, "kappa", chrome="none")
        page = directory / "index.html"
        text = page.read_text(encoding="utf-8")
        start, end = text.index("<!-- blog:header:start"), text.index("<!-- blog:header:end -->")
        rest = text[end + len("<!-- blog:header:end -->") :]
        own = '<a href="/">AutoRubric</a> <a href="/blog/">Blog</a>'
        page.write_text(text[:start] + own + rest, encoding="utf-8")
        assert _problems(site, "kappa") == []

        page.write_text(text[:start] + rest, encoding="utf-8")
        # The footer region is still there, and it links to the blog index.
        assert _problems(site, "kappa") == ['no link to the home page (<a href="/">)']

    def test_check_reports_a_site_page_whose_navbar_is_out_of_date(self, site: Path) -> None:
        landing = site.parent / "docs" / "landing" / "index.html"
        landing.parent.mkdir(parents=True)
        landing.write_text(
            "<!-- blog:header:start -->\n<header>old</header>\n<!-- blog:header:end -->\n",
            encoding="utf-8",
        )
        assert blog.check(site, site.parent) == [
            "docs/landing/index.html: navbar differs from blog/_shared: run 'blog.py sync-chrome'"
        ]
        assert blog.sync_chrome(site, site.parent) == [landing]
        assert blog.check(site, site.parent) == []


class TestBuild:
    def _many(self, site: Path, count: int) -> None:
        for i in range(count):
            _add_post(
                site,
                f"post-{i:02d}",
                date=(dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat(),
                tags=["metrics"] if i % 2 else ["metrics", "cascade"],
            )

    def test_an_empty_blog_builds_an_index_that_says_so(self, site: Path, tmp_path: Path) -> None:
        out = tmp_path / "site"
        assert blog.build(out, site) == []
        index = (out / "blog" / "index.html").read_text(encoding="utf-8")
        assert "No posts yet." in index
        assert 'class="pagination"' not in index
        assert '<a href="/blog/" aria-current="page">Blog</a>' in index
        assert "{{" not in index
        ET.fromstring((out / "blog" / "feed.xml").read_text(encoding="utf-8"))

    def test_23_posts_make_three_pages_newest_first(self, site: Path, tmp_path: Path) -> None:
        self._many(site, 23)
        out = tmp_path / "site"
        blog.build(out, site)
        pages = [
            (out / "blog" / "index.html").read_text(encoding="utf-8"),
            (out / "blog" / "page" / "2" / "index.html").read_text(encoding="utf-8"),
            (out / "blog" / "page" / "3" / "index.html").read_text(encoding="utf-8"),
        ]
        assert not (out / "blog" / "page" / "4").exists()
        assert [page.count('<article class="post-entry">') for page in pages] == [10, 10, 3]
        assert pages[0].index("Post post-22") < pages[0].index("Post post-13")
        assert "Post post-12" in pages[1] and "Post post-02" in pages[2]

        assert 'rel="next" href="/blog/page/2/"' in pages[0] and 'rel="prev" href=' not in pages[0]
        assert (
            'rel="prev" href="/blog/"' in pages[1] and 'rel="next" href="/blog/page/3/"' in pages[1]
        )
        assert 'rel="prev" href="/blog/page/2/"' in pages[2] and 'rel="next" href=' not in pages[2]
        assert '<link rel="canonical" href="https://autorubric.org/blog/page/2/">' in pages[1]
        assert "Page 3 of 3" in pages[2]

    def test_an_exact_multiple_has_no_empty_last_page(self, site: Path, tmp_path: Path) -> None:
        self._many(site, 4)
        out = tmp_path / "site"
        blog.build(out, site, page_size=2)
        assert (out / "blog" / "page" / "2" / "index.html").is_file()
        assert not (out / "blog" / "page" / "3").exists()

    def test_tag_pages_list_only_their_posts(self, site: Path, tmp_path: Path) -> None:
        self._many(site, 5)
        out = tmp_path / "site"
        blog.build(out, site)
        cascade = (out / "blog" / "tags" / "cascade" / "index.html").read_text(encoding="utf-8")
        assert cascade.count('<article class="post-entry">') == 3
        assert "Posts tagged “cascade”" in cascade
        assert (out / "blog" / "tags" / "metrics" / "index.html").is_file()

    def test_drafts_are_published_only_when_asked(self, site: Path, tmp_path: Path) -> None:
        _add_post(site, "ready")
        _add_post(site, "wip", draft=True)
        out = tmp_path / "site"
        assert [post.slug for post in blog.build(out, site)] == ["ready"]
        assert not (out / "blog" / "wip").exists()
        for name in ("blog/index.html", "blog/feed.xml", "blog/sitemap.xml", "llms.txt"):
            assert "wip" not in (out / name).read_text(encoding="utf-8")

        assert {post.slug for post in blog.build(out, site, drafts=True)} == {"ready", "wip"}
        assert (out / "blog" / "wip" / "index.html").is_file()

    def test_sources_are_not_published_and_the_markdown_copy_is(
        self, site: Path, tmp_path: Path
    ) -> None:
        directory = _add_post(site, "kappa")
        (directory / "card.html").write_text("<html></html>", encoding="utf-8")
        (directory / "src").mkdir()
        (directory / "src" / "widget.ts").write_text("export {}", encoding="utf-8")
        (directory / "assets").mkdir()
        (directory / "assets" / "figure.svg").write_text("<svg/>", encoding="utf-8")
        out = tmp_path / "site"
        blog.build(out, site)
        published = sorted(
            p.relative_to(out / "blog" / "kappa").as_posix()
            for p in (out / "blog" / "kappa").rglob("*")
        )
        assert published == ["assets", "assets/figure.svg", "card.png", "index.html", "index.md"]
        assert (out / "blog" / "assets" / "base.css").is_file()

    def test_feed_sitemaps_robots_and_llms(self, site: Path, tmp_path: Path) -> None:
        _add_post(site, "older", date="2026-01-01")
        _add_post(site, "newer", date="2026-03-01", updated="2026-03-05", title="Q&A <on> kappa")
        out = tmp_path / "site"
        blog.build(out, site)

        feed = ET.fromstring((out / "blog" / "feed.xml").read_text(encoding="utf-8"))
        entries = feed.findall(f"{ATOM}entry")
        assert [entry.findtext(f"{ATOM}title") for entry in entries] == [
            "Q&A <on> kappa",
            "Post older",
        ]
        assert feed.findtext(f"{ATOM}updated") == "2026-03-05T00:00:00Z"
        assert entries[0].findtext(f"{ATOM}id") == "https://autorubric.org/blog/newer/"

        urls = ET.fromstring((out / "blog" / "sitemap.xml").read_text(encoding="utf-8"))
        locations = [url.findtext(f"{SITEMAP}loc") for url in urls]
        assert locations == [
            "https://autorubric.org/blog/",
            "https://autorubric.org/blog/tags/metrics/",
            "https://autorubric.org/blog/newer/",
            "https://autorubric.org/blog/older/",
        ]
        root = ET.fromstring((out / "sitemap.xml").read_text(encoding="utf-8"))
        assert root.tag == f"{SITEMAP}sitemapindex"
        assert [item.findtext(f"{SITEMAP}loc") for item in root] == [
            "https://autorubric.org/sitemap-pages.xml",
            "https://autorubric.org/blog/sitemap.xml",
            "https://autorubric.org/docs/sitemap.xml",
        ]

        robots = (out / "robots.txt").read_text(encoding="utf-8")
        assert "User-agent: ClaudeBot\nAllow: /" in robots
        assert robots.rstrip().endswith("Sitemap: https://autorubric.org/sitemap.xml")

        llms = (out / "llms.txt").read_text(encoding="utf-8")
        assert llms.startswith("# AutoRubric\n\n> ")
        assert (
            "- [Q&A <on> kappa](https://autorubric.org/blog/newer/index.md): What newer shows."
            in llms
        )

        index = (out / "blog" / "index.html").read_text(encoding="utf-8")
        assert "Q&amp;A &lt;on&gt; kappa" in index
        block = index.split('<script type="application/ld+json">')[1].split("</script>")[0]
        data = json.loads(block)
        assert data["@type"] == "Blog"
        assert [item["headline"] for item in data["blogPost"]] == ["Q&A <on> kappa", "Post older"]

    def test_an_unrendered_post_stops_the_build(self, site: Path, tmp_path: Path) -> None:
        _add_post(site, "kappa", rendered=False)
        with pytest.raises(blog.BlogError, match="kappa: index.html is missing"):
            blog.build(tmp_path / "site", site)

    def test_an_unrendered_draft_is_skipped(
        self, site: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _add_post(site, "ready")
        _add_post(site, "wip", rendered=False, draft=True)
        posts = blog.build(tmp_path / "site", site, drafts=True)
        assert [post.slug for post in posts] == ["ready"]
        assert "skipped draft 'wip'" in capsys.readouterr().err

    def test_the_markdown_copy_leaves_out_notes_to_the_renderer(self, site: Path) -> None:
        body = (
            "Before.\n\n<!-- design: a chart\n  goes here -->\n\nAfter. <!-- a kept comment -->\n"
        )
        directory = _add_post(site, "kappa", body=body)
        copy = blog.markdown_copy(blog.load_post(directory, blog.load_site(site)))
        assert copy.endswith("Before.\n\n\nAfter. <!-- a kept comment -->\n")


class TestNew:
    def test_the_scaffold_is_a_valid_draft(self, site: Path) -> None:
        directory = blog.new_post("my-first-post", site)
        post = blog.load_post(directory, blog.load_site(site))
        assert post.draft and post.title == "My first post"
        assert "openai/gpt-5.6-luna" in post.resolved_body()
        assert "## Deviations" in (directory / "design.md").read_text(encoding="utf-8")

    def test_an_existing_post_is_not_overwritten(self, site: Path) -> None:
        blog.new_post("my-first-post", site)
        with pytest.raises(blog.BlogError, match="already exists"):
            blog.new_post("my-first-post", site)

    def test_a_bad_slug_is_refused(self, site: Path) -> None:
        with pytest.raises(blog.BlogError, match="not a slug"):
            blog.new_post("My Post", site)
