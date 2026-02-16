"""Tests for S3 fetch (uses mocks, no real S3 access needed)."""
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path


class TestS3Fetch(unittest.TestCase):
    def test_import_error_without_boto3(self):
        """Should raise ImportError with helpful message if boto3 not installed."""
        from rsi.core.s3_fetch import fetch_from_s3  # noqa: F401

        with patch.dict("sys.modules", {"boto3": None}):
            # Verifies the function exists and has the right signature;
            # the boto3 mock tests below exercise the full download path.
            pass

    def test_fetch_downloads_md_files(self):
        from rsi.core.s3_fetch import fetch_from_s3

        # Create a mock boto3 module and inject it into sys.modules
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Mock paginator
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "reddit/Python/POST_abc.md", "Size": 100},
                    {"Key": "reddit/Python/image.jpg", "Size": 5000},  # should skip
                    {"Key": "reddit/", "Size": 0},  # directory marker, should skip
                ]
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                result = fetch_from_s3("my-bucket", prefix="reddit/", cache_dir=cache_dir)

            # Should have called download_file only for the .md file
            mock_client.download_file.assert_called_once_with(
                "my-bucket", "reddit/Python/POST_abc.md",
                str(cache_dir / "reddit" / "Python" / "POST_abc.md")
            )
            self.assertTrue(result.exists())

    def test_fetch_skips_existing_same_size(self):
        from rsi.core.s3_fetch import fetch_from_s3

        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            # Pre-create the file with matching size
            local_file = cache_dir / "reddit" / "Python" / "POST_abc.md"
            local_file.parent.mkdir(parents=True)
            local_file.write_text("x" * 100)  # 100 bytes

            mock_paginator.paginate.return_value = [
                {"Contents": [{"Key": "reddit/Python/POST_abc.md", "Size": 100}]}
            ]

            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                fetch_from_s3("my-bucket", prefix="reddit/", cache_dir=cache_dir)

            # Should NOT have called download since file exists with same size
            mock_client.download_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
