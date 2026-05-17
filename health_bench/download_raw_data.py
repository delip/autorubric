#!/usr/bin/env python3
"""Download HealthBench raw JSONL files into ``raw_data/``.

The four source files are hosted by OpenAI on their public blob store and are
the same URLs that ``simple-evals/healthbench_eval.py`` and
``simple-evals/healthbench_meta_eval.py`` read from. This script exists so
the conversion pipeline does not depend on a checked-out ``simple-evals``
copy (which is gitignored at the project root).

Source URLs are renamed on disk to the short names that
``convert_to_rubric_dataset.py`` expects:

    {OpenAI URL basename}                         -> {local name}
    2025-05-07-06-14-12_oss_eval.jsonl            -> healthbench_main.jsonl
    hard_2025-05-08-21-00-10.jsonl                -> healthbench_hard.jsonl
    consensus_2025-05-09-20-00-46.jsonl           -> healthbench_consensus.jsonl
    2025-05-07-06-14-12_oss_meta_eval.jsonl       -> healthbench_meta.jsonl

Usage:
    cd health_bench
    uv run python download_raw_data.py            # skips already-downloaded files
    uv run python download_raw_data.py --force    # re-downloads everything

Total payload is ~240 MB. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench"

FILES: dict[str, str] = {
    # local filename -> source URL
    "healthbench_main.jsonl": f"{BASE_URL}/2025-05-07-06-14-12_oss_eval.jsonl",
    "healthbench_hard.jsonl": f"{BASE_URL}/hard_2025-05-08-21-00-10.jsonl",
    "healthbench_consensus.jsonl": f"{BASE_URL}/consensus_2025-05-09-20-00-46.jsonl",
    "healthbench_meta.jsonl": f"{BASE_URL}/2025-05-07-06-14-12_oss_meta_eval.jsonl",
}


def _format_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` into ``dest`` atomically with a single-line progress meter."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    started = time.monotonic()
    last_log = 0.0

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_log >= 0.5 or downloaded == total:
                    elapsed = max(now - started, 1e-6)
                    rate = downloaded / elapsed
                    if total:
                        pct = 100 * downloaded / total
                        msg = (
                            f"  {dest.name}: {_format_size(downloaded)}/"
                            f"{_format_size(total)} ({pct:5.1f}%) "
                            f"@ {_format_size(int(rate))}/s"
                        )
                    else:
                        msg = (
                            f"  {dest.name}: {_format_size(downloaded)} "
                            f"@ {_format_size(int(rate))}/s"
                        )
                    sys.stderr.write("\r" + msg.ljust(80))
                    sys.stderr.flush()
                    last_log = now
    sys.stderr.write("\n")
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the local file already exists.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "raw_data",
        help="Output directory (default: ./raw_data next to this script).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(FILES)} HealthBench JSONL files to {args.out}/")

    failures: list[str] = []
    for name, url in FILES.items():
        dest = args.out / name
        if dest.exists() and not args.force:
            print(f"  {name}: already present ({_format_size(dest.stat().st_size)}), skipping")
            continue
        print(f"  {name}: {url}")
        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001  -- surface any download failure
            print(f"  {name}: FAILED ({exc})", file=sys.stderr)
            failures.append(name)
            # Clean up any partial file so the next run starts fresh.
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()

    if failures:
        print(f"\n{len(failures)} download(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    # Free-disk sanity check so the next step (conversion) has room to write.
    free = shutil.disk_usage(args.out).free
    print(f"\nAll files present in {args.out}/ ({_format_size(free)} free).")
    print("Next: uv run python convert_to_rubric_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
