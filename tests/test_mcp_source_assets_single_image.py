"""Regression tests for the single-image NotebookLM MCP source tool.

Run with: PYTHONPATH=src <notebooklm-python> tests/test_mcp_source_assets_single_image.py
"""

import asyncio
import subprocess
import tempfile
import unittest
from pathlib import Path

from fastmcp import FastMCP

from notebooklm.mcp.server import register_all
from notebooklm.mcp.tools.source_assets import (
    _content_type,
    _page_urls,
    _prepare_for_mcp,
)


class SourceImageTests(unittest.TestCase):
    def test_page_url_parsing_and_deduplication(self):
        first = "https://lh3.googleusercontent.com/notebooklm/page1?key=signed%2Fpart"
        second = "https://lh3.googleusercontent.com/notebooklm/page2"
        self.assertEqual(_page_urls(f"{first}, {second} ({first})."), [first, second])

    def test_rejects_untrusted_hosts_and_redirect_like_authorities(self):
        candidates = [
            "http://lh3.googleusercontent.com/notebooklm/page",
            "https://lh3.googleusercontent.com.evil.example/notebooklm/page",
            "https://googleusercontent.com.evil.example/notebooklm/page",
            "https://evil.example/notebooklm/page",
            "https://user@lh3.googleusercontent.com/notebooklm/page",
            "https://lh3.googleusercontent.com:443/notebooklm/page",
            "https://lh3.googleusercontent.com/other/page",
            "https://lh3.googleusercontent.com/notebooklm",
        ]
        for url in candidates:
            with self.subTest(url=url):
                self.assertEqual(_page_urls(url), [])

    def test_image_magic_bytes(self):
        self.assertEqual(_content_type(b"\x89PNG\r\n\x1a\nrest"), "image/png")
        self.assertEqual(_content_type(b"\xff\xd8\xffrest"), "image/jpeg")
        self.assertEqual(_content_type(b"RIFFxxxxWEBPrest"), "image/webp")
        self.assertEqual(_content_type(b"GIF89arest"), "image/gif")
        self.assertIsNone(_content_type(b"<html>Google sign-in</html>"))

    def test_small_image_not_reencoded(self):
        data = b"\xff\xd8\xff" + b"x" * 100
        actual, mime, changed = _prepare_for_mcp(data, "image/jpeg")
        self.assertIs(actual, data)
        self.assertEqual(mime, "image/jpeg")
        self.assertFalse(changed)

    def test_large_image_reencoded_with_size_limit(self):
        with tempfile.TemporaryDirectory() as dirname:
            image_path = Path(dirname) / "large.png"
            subprocess.run(
                ["convert", "-size", "1200x1200", "xc:white", "+noise", "Random", str(image_path)],
                check=True, capture_output=True, timeout=35,
            )
            original = image_path.read_bytes()
            self.assertGreater(len(original), 1024 * 1024)
            processed, mime, changed = _prepare_for_mcp(original, "image/png")
            self.assertTrue(changed)
            self.assertEqual(mime, "image/jpeg")
            self.assertLessEqual(len(processed), 1024 * 1024)
            self.assertEqual(_content_type(processed), "image/jpeg")

    def test_only_single_page_image_tool_registered(self):
        mcp = FastMCP("single-image-test")
        register_all(mcp)
        self.assertIsNotNone(asyncio.run(mcp.get_tool("source_get_page_image")))
        self.assertIsNone(asyncio.run(mcp.get_tool("source_download_original")))


if __name__ == "__main__":
    unittest.main()
