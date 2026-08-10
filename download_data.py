"""Fetch the replication extracts used by empirical.py.

The three datasets are redistributed inside public R packages. Rather than
vendoring them, this script downloads them from the packages' source
repositories and verifies each file against a recorded SHA-256 digest, so a
silent upstream change cannot pass unnoticed.

Usage
-----
    python download_data.py            # download anything missing
    python download_data.py --force    # re-download and re-verify everything
    python download_data.py --check    # verify only, exit 1 on any mismatch

Sources
-------
mpdta.rda    did 2.5.1 (GPL-3), Callaway and Sant'Anna
             https://github.com/bcallaway11/did
castle.rda   bacondecomp 0.1.3 (MIT), Flack and Edwards; the underlying data
divorce.rda  are from Cheng and Hoekstra (2013) and Stevenson and Wolfers
             (2006) respectively
             https://github.com/evanjflack/bacondecomp

Both licenses permit redistribution, so the .rda files may instead be shipped
with the replication archive; if they are, keep the attributions above and
include the upstream license texts.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# name -> (url, sha256, expected size in bytes)
FILES = {
    "mpdta.rda": (
        "https://raw.githubusercontent.com/bcallaway11/did/master/data/mpdta.rda",
        "0be3bc5eaaedaa50e047deaf45ad0e193e7210df1644103c63a6c1baa75965d4",
        28345,
    ),
    "castle.rda": (
        "https://raw.githubusercontent.com/evanjflack/bacondecomp/master/data/castle.rda",
        "3e102de8f64e5fbc602c165144f4f7ed453f0c2a8e106ea5add156df3f88f708",
        81184,
    ),
    "divorce.rda": (
        "https://raw.githubusercontent.com/evanjflack/bacondecomp/master/data/divorce.rda",
        "59ab3530a8229905c9f737c6b9bfbf1d78f56afc6c1a8123d74a39ac490f98b1",
        692064,
    ),
}

TIMEOUT = 60
RETRIES = 3


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, sha256: str) -> bool:
    return path.is_file() and digest(path) == sha256


def fetch(name: str, url: str, sha256: str, size: int, dest: Path) -> None:
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                payload = response.read()
            break
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
            last = exc
            print(f"  attempt {attempt}/{RETRIES} failed: {exc}")
    else:  # pragma: no cover
        raise SystemExit(
            f"could not download {name} from {url}\n"
            f"last error: {last}\n"
            "If the machine has no outbound network access, obtain the file "
            "manually and place it in ./data/."
        )

    got = hashlib.sha256(payload).hexdigest()
    if got != sha256:
        raise SystemExit(
            f"checksum mismatch for {name}\n"
            f"  expected {sha256}\n  received {got}\n"
            "The upstream file has changed. Do not use it silently: the "
            "numbers in Table 1 were computed from the recorded version."
        )
    if len(payload) != size:  # pragma: no cover
        raise SystemExit(f"size mismatch for {name}: {len(payload)} != {size}")

    dest.write_bytes(payload)


def ensure(force: bool = False, check_only: bool = False, quiet: bool = False) -> Path:
    """Make sure ./data holds verified copies of all three files."""
    DATA_DIR.mkdir(exist_ok=True)
    failures = []

    for name, (url, sha256, size) in FILES.items():
        dest = DATA_DIR / name
        if not force and verify(dest, sha256):
            if not quiet:
                print(f"ok       {name}")
            continue
        if check_only:
            failures.append(name)
            print(f"{'CORRUPT' if dest.is_file() else 'MISSING':<8} {name}")
            continue
        print(f"fetching {name} ...")
        fetch(name, url, sha256, size, dest)
        print(f"ok       {name}  ({size:,} bytes, sha256 verified)")

    if check_only and failures:
        raise SystemExit(f"{len(failures)} file(s) missing or corrupt")
    return DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a valid copy exists")
    parser.add_argument("--check", action="store_true",
                        help="verify existing files without downloading")
    args = parser.parse_args()
    path = ensure(force=args.force, check_only=args.check)
    print(f"\ndata directory: {path}")


if __name__ == "__main__":
    sys.exit(main())
