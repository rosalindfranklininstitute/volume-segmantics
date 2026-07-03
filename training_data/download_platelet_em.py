#!/usr/bin/env python
"""Download the bio3d-vision platelet-EM dataset into this directory.

Fetches the RGBA "images and labels" archive for the platelet-EM project and
unpacks it next to this script, producing::

    training_data/images_and_labels_rgba/platelet-em/
        images/            24-images.tif, 50-images.tif
        labels-semantic/   24-semantic.tif, 50-semantic.tif
        labels-instance/   24-instance-{cell,organelle}.tif, 50-instance-*.tif

Dependency-free (standard library only). Run it directly::

    python training_data/download_platelet_em.py             # download + extract
    python training_data/download_platelet_em.py --force     # re-download even if present
    python training_data/download_platelet_em.py --keep-zip  # keep the .zip after extracting
    python training_data/download_platelet_em.py --dest DIR  # download somewhere else

The download resumes automatically (HTTP range requests) if the connection
drops part-way, and the archive is verified before extraction.

The dataset and its terms of use are described at
https://bio3d-vision.github.io/platelet-description.html - please cite the
platelet-EM authors if you use it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Direct-download (``dl=1``) Dropbox link for the RGBA images+labels archive,
# taken from the platelet-EM project page.
DATASET_URL = (
    "https://www.dropbox.com/s/pvrfnurjq11k0l3/images_and_labels_rgba.zip?dl=1"
)
# The zip unpacks a top-level ``platelet-em/`` directory; we extract it under
# this subdirectory so the dataset lives at <dest>/images_and_labels_rgba/platelet-em.
EXTRACT_SUBDIR = "images_and_labels_rgba"
# A file that only exists once extraction has completed - used for idempotency.
MARKER = Path(EXTRACT_SUBDIR) / "platelet-em" / "images" / "50-images.tif"

_USER_AGENT = "volume-segmantics-dataset-downloader/1.0"
_CHUNK = 1 << 20  # 1 MiB
_MAX_RETRIES = 6
_TIMEOUT = 60  # seconds, per network read


def _human(num_bytes: float) -> str:
    """Format a byte count for humans."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GiB"  # unreachable; keeps type checkers happy


def _progress(downloaded: int, total: int) -> None:
    if total:
        pct = 100 * downloaded / total
        sys.stdout.write(
            f"\r  downloading... {_human(downloaded)} / {_human(total)} ({pct:4.1f}%)"
        )
    else:
        sys.stdout.write(f"\r  downloading... {_human(downloaded)}")
    sys.stdout.flush()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, resuming with HTTP range on dropped connections.

    ``dest`` is appended to across retries, so an interrupted transfer picks up
    where it left off rather than starting over. Raises ``RuntimeError`` if the
    download is still incomplete after ``_MAX_RETRIES`` attempts.
    """
    downloaded = dest.stat().st_size if dest.exists() else 0
    total = 0
    attempt = 0

    while True:
        headers = {"User-Agent": _USER_AGENT}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        request = urllib.request.Request(url, headers=headers)

        network_error: Exception | None = None
        try:
            # nosec B310: the URL is hard-coded and HTTPS.
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
                status = getattr(response, "status", None) or response.getcode()
                if downloaded and status != 206:
                    # Server ignored our Range request; restart from scratch.
                    downloaded = 0
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    total = (
                        int(content_range.rsplit("/", 1)[-1])
                        if "/" in content_range
                        else 0
                    )
                else:
                    total = int(response.headers.get("Content-Length", 0))

                with dest.open("ab" if downloaded else "wb") as handle:
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        _progress(downloaded, total)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            network_error = exc

        sys.stdout.write("\n")

        if total and downloaded >= total:
            return  # complete
        if total == 0 and network_error is None:
            return  # size unknown and the stream ended cleanly; nothing more to do

        attempt += 1
        if attempt > _MAX_RETRIES:
            detail = f" ({network_error})" if network_error else ""
            raise RuntimeError(
                f"Download incomplete after {_MAX_RETRIES} retries: got "
                f"{downloaded} of {total or '?'} bytes{detail}."
            )
        reason = network_error if network_error else "connection closed early"
        print(
            f"  interrupted at {_human(downloaded)}"
            f"{' / ' + _human(total) if total else ''} ({reason}); "
            f"resuming (attempt {attempt}/{_MAX_RETRIES})...",
            file=sys.stderr,
        )


def _safe_extract(zip_path: Path, target: Path) -> None:
    """Extract ``zip_path`` into ``target``, rejecting path-traversal entries."""
    target = target.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            resolved = (target / member).resolve()
            if resolved != target and not str(resolved).startswith(str(target) + os.sep):
                raise RuntimeError(f"Refusing unsafe path in archive: {member!r}")
        archive.extractall(target)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=here,
        help="Directory to download into (default: this script's directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the dataset is already present.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded .zip after extraction (default: delete it).",
    )
    args = parser.parse_args()

    dest: Path = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    extract_root = dest / EXTRACT_SUBDIR
    marker = dest / MARKER

    if marker.exists() and not args.force:
        print(
            f"Dataset already present at {extract_root}\n"
            "Nothing to do (pass --force to re-download)."
        )
        return 0

    zip_path = dest / "images_and_labels_rgba.zip"
    part_path = dest / "images_and_labels_rgba.zip.part"
    print("Downloading platelet-EM dataset")
    print(f"  source: {DATASET_URL}")
    print(f"  into:   {zip_path}")

    # Download to a ``.part`` file so an interrupted run can resume and never
    # leaves a truncated archive that looks complete.
    _download(DATASET_URL, part_path)

    if not zipfile.is_zipfile(part_path):
        part_path.unlink(missing_ok=True)
        print(
            "ERROR: the downloaded file is not a valid zip archive; it was "
            "removed. Please re-run to try again.",
            file=sys.stderr,
        )
        return 1
    part_path.replace(zip_path)

    print(f"Extracting to {extract_root}")
    if args.force and extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    _safe_extract(zip_path, extract_root)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("Removed archive (pass --keep-zip to retain it).")

    print(f"Done. Dataset is at {extract_root / 'platelet-em'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
