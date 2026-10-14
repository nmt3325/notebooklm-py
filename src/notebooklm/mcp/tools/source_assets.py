"""Original source downloads and PDF page images for the existing NotebookLM MCP server.

All Google requests run on the NotebookLM host through the library's own
trusted-host, redirect-guarded AssetDownloadService. No arbitrary URL inputs,
raw cookies, or signed URLs are returned to MCP clients.
"""
from __future__ import annotations

import base64
import json
import subprocess
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

from fastmcp import Context
from fastmcp.tools.tool import ToolResult
from mcp.types import ImageContent, TextContent

from ..._artifact.downloads import AssetDownloadService
from ...exceptions import SourceNotFoundError, ValidationError
from ...paths import get_storage_path
from .._confirm import READ_ONLY
from .._context import get_client, get_profile_state
from .._errors import mcp_errors
from .._resolve import resolve_notebook, resolve_source

_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MCP_IMAGE_BYTES = 1024 * 1024
_PAGE_URL_RE = re.compile(r"https://\S+", re.IGNORECASE)
_ALLOWED_IMAGE_HOSTS = ("googleusercontent.com", "google.com")


def _page_urls(content: str) -> list[str]:
    """Only NotebookLM-hosted page images, in original text order."""
    found: list[str] = []
    seen: set[str] = set()
    for candidate in _PAGE_URL_RE.findall(content):
        url = candidate.rstrip(".,;)]}>\\\"\u0027")
        try:
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()
            unsafe_authority = bool(parts.username or parts.password or parts.port)
        except ValueError:
            continue
        if parts.scheme != "https" or unsafe_authority:
            continue
        if not any(host == base or host.endswith("." + base) for base in _ALLOWED_IMAGE_HOSTS):
            continue
        if not parts.path.startswith("/notebooklm/"):
            continue
        if url not in seen:
            found.append(url)
            seen.add(url)
    return found


def _content_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None



def _prepare_for_mcp(data: bytes, mime: str) -> tuple[bytes, str, bool]:
    """Keep already-small Google images byte-identical; recompress only if needed."""
    if len(data) <= _MCP_IMAGE_BYTES:
        return data, mime, False

    # ImageMagick is installed on Oracle Ubuntu. Conversion remains server-local.
    suffix = {"image/jpeg": ".jpg", "image/png": ".png",
              "image/webp": ".webp", "image/gif": ".gif"}[mime]
    with tempfile.TemporaryDirectory(prefix="notebooklm-mcp-reencode-") as directory:
        folder = Path(directory)
        input_image = folder / ("input" + suffix)
        result_image = folder / "result.jpg"
        input_image.write_bytes(data)
        # Iterate down in size only until the encoded *single* image fits.
        for edge, quality in ((2200, 90), (1800, 87), (1500, 85),
                              (1200, 82), (1000, 78), (800, 74)):
            command = [
                "convert", "-limit", "memory", "128MiB",
                "-limit", "map", "256MiB", "-limit", "time", "15",
                str(input_image), "-auto-orient",
                "-resize", f"{edge}x{edge}>",
                "-background", "white", "-alpha", "remove",
                "-alpha", "off", "-strip",
                "-sampling-factor", "4:4:4",
                "-quality", str(quality), str(result_image),
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, timeout=25, check=False
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ValidationError("Image re-encoding failed") from exc
            if completed.returncode != 0:
                raise ValidationError("Image re-encoding failed")
            if 0 < result_image.stat().st_size <= _MCP_IMAGE_BYTES:
                return result_image.read_bytes(), "image/jpeg", True
    raise ValidationError("Image could not be reduced to the MCP size limit")


async def _download(ctx: Context, url: str, *, max_size: int) -> bytes:
    """Download to a short-lived server-side temp file, then load bounded bytes."""
    state = get_profile_state(ctx)
    downloader = AssetDownloadService(storage_path=get_storage_path(state.profile))
    with tempfile.TemporaryDirectory(prefix="notebooklm-mcp-source-") as directory:
        destination = Path(directory) / "source.bin"
        await downloader.download_url(url, str(destination))
        length = destination.stat().st_size
        if not 0 < length <= max_size:
            raise ValidationError(f"Downloaded file size {length} exceeds 0-{max_size} byte limit")
        return destination.read_bytes()


async def _source(ctx: Context, notebook: str, source: str) -> tuple[Any, str, str, Any]:
    client = await get_client(ctx)
    notebook_id = await resolve_notebook(client, notebook)
    source_id = await resolve_source(client, notebook_id, source)
    record = await client.sources.get_or_none(notebook_id, source_id)
    if record is None:
        raise SourceNotFoundError(source_id)
    return client, notebook_id, source_id, record


def register(mcp: Any) -> None:
    """Register native binary-source tools in the running NotebookLM MCP."""

    @mcp.tool(annotations=READ_ONLY)
    async def source_get_page_image(
        ctx: Context,
        notebook: str,
        source: str,
        page: int = 1,
    ) -> ToolResult:
        """Return exactly one actual image from a PDF page or image source.

        The response contains an MCP image content block (not an image URL).
        Pages are numbered from 1. Existing small images are not re-encoded;
        large images are reduced to at most 1 MiB for MCP transport.
        """
        with mcp_errors():
            if page < 1:
                raise ValidationError("page must be >= 1")
            client, nb_id, source_id, record = await _source(ctx, notebook, source)
            # Uploaded image sources may not have an indexed text body. Prefer
            # the original image URL rather than requiring get_fulltext to succeed.
            is_image = "image" in str(getattr(record, "kind", "")).lower()
            original = getattr(record, "download_url", None)
            if is_image and original:
                urls = [original]
            else:
                full = await client.sources.get_fulltext(nb_id, source_id, output_format="text")
                urls = _page_urls(full.content)
            if not urls:
                raise ValidationError("No image references found for this source")
            if page > len(urls):
                raise ValidationError(f"page {page} out of range (1-{len(urls)})")
            data = await _download(ctx, urls[page - 1], max_size=_MAX_IMAGE_BYTES)
            mime = _content_type(data)
            if mime is None:
                raise ValidationError("Google returned a non-image response")
            original_bytes = len(data)
            data, mime, reencoded = _prepare_for_mcp(data, mime)
            meta = {"notebook_id": nb_id, "source_id": source_id, "title": record.title,
                    "page": page, "page_count": len(urls), "mime_type": mime,
                    "size_bytes": len(data), "original_bytes": original_bytes,
                    "reencoded": reencoded, "status": "image_ready"}
            return ToolResult(
                content=[
                    TextContent(type="text", text=json.dumps(meta, ensure_ascii=False)),
                    ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mimeType=mime),
                ],
                structured_content=meta,
            )
