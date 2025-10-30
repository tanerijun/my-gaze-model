"""
R2 (Cloudflare) Upload Manager

Handles uploading collected session data to Cloudflare R2 using S3-compatible API.
Supports multipart uploads for large files with progress tracking.
"""

from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError


class R2UploadManager:
    """
    Manages uploads to Cloudflare R2 using S3-compatible API.
    """

    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        endpoint_url: str,
        bucket_name: str,
    ):
        """
        Initialize the R2 upload manager.

        Args:
            access_key_id: R2 access key ID
            secret_access_key: R2 secret access key
            endpoint_url: R2 endpoint URL (from Cloudflare dashboard)
            bucket_name: Name of the R2 bucket
        """
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.s3_client = None

    def authenticate(self) -> bool:
        """
        Create S3 client for R2.

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name="auto",  # R2 uses 'auto' for region
            )

            # Test connection by listing buckets
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✓ Authenticated with R2 bucket: {self.bucket_name}")
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            print(f"Authentication failed: {error_code} - {e}")
            return False
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False

    def upload_file(self, file_path: str, progress_callback=None) -> Optional[str]:
        """
        Upload a file to R2 with progress tracking.

        Args:
            file_path: Path to the file to upload
            progress_callback: Optional callback function(current, total) for progress

        Returns:
            Public URL if upload successful, None otherwise
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"File not found: {file_path}")
                return None

            # Authenticate if not already
            if not self.s3_client:
                if not self.authenticate():
                    return None

            file_name = file_path_obj.name
            file_size = file_path_obj.stat().st_size
            print(f"Uploading: {file_name} ({file_size / (1024 * 1024):.2f} MB)")

            # Upload with progress callback
            class ProgressCallback:
                def __init__(self, filename, filesize, callback=None):
                    self._filename = filename
                    self._size = filesize
                    self._seen_so_far = 0
                    self._callback = callback

                def __call__(self, bytes_amount):
                    self._seen_so_far += bytes_amount
                    percentage = (self._seen_so_far / self._size) * 100
                    print(f"Upload progress: {percentage:.1f}%")
                    if self._callback:
                        self._callback(self._seen_so_far, self._size)

            callback = ProgressCallback(file_name, file_size, progress_callback)

            # Upload to R2
            self.s3_client.upload_file(
                str(file_path_obj),
                self.bucket_name,
                file_name,
                Callback=callback,
            )

            print("✓ Upload complete!")

            # Return the file URL (R2 public URL format)
            # Note: This assumes public access is enabled on the bucket
            url = f"{self.endpoint_url}/{file_name}"
            return url

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            print(f"Upload failed: {error_code} - {e}")
            return None
        except Exception as e:
            print(f"Upload failed: {e}")
            return None

    def get_file_url(self, file_name: str) -> str:
        """
        Get the URL for an uploaded file.

        Args:
            file_name: Name of the file in R2

        Returns:
            File URL
        """
        return f"{self.endpoint_url}/{file_name}"
