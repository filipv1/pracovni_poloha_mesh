#!/usr/bin/env python3
"""
Google Drive OAuth Client with Persistent Storage for RunPod
Supports device flow for headless environments, token persistence, and auto-refresh.
"""

import os
import json
import pickle
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Google Drive API scopes - need full access for our operations
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file'
]

class GoogleDriveOAuthClient:
    """
    Google Drive client using OAuth2 with persistent token storage.
    Designed for RunPod headless environment with device flow authentication.
    """

    def __init__(self, credentials_path: str = None, token_dir: str = None):
        """
        Initialize OAuth Google Drive client

        Args:
            credentials_path: Path to oauth_credentials.json file
            token_dir: Directory for persistent token storage (survives pod restarts)
        """
        self.credentials_path = credentials_path or os.getenv(
            'OAUTH_CREDENTIALS_PATH',
            '/workspace/persistent/oauth/oauth_credentials.json'
        )
        self.token_dir = token_dir or os.getenv(
            'OAUTH_TOKEN_DIR',
            '/workspace/persistent/oauth'
        )

        # Ensure token directory exists
        Path(self.token_dir).mkdir(parents=True, exist_ok=True)

        self.token_path = os.path.join(self.token_dir, 'token.pickle')
        self.credentials = None
        self.service = None

        logger.info(f"OAuth client initialized - credentials: {self.credentials_path}, tokens: {self.token_dir}")

        self._initialize_service()

    def _initialize_service(self):
        """Initialize Google Drive service with OAuth2 credentials"""
        try:
            self.credentials = self._load_or_create_credentials()

            # Build Drive service
            self.service = build('drive', 'v3', credentials=self.credentials)

            # Test connection
            if self._test_connection():
                logger.info("Google Drive OAuth service initialized successfully")
            else:
                raise Exception("Connection test failed")

        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise

    def _load_or_create_credentials(self) -> Credentials:
        """Load existing credentials or create new ones via device flow"""
        credentials = None

        # Try to load existing token
        if os.path.exists(self.token_path):
            logger.info("Loading existing OAuth token...")
            try:
                with open(self.token_path, 'rb') as token_file:
                    credentials = pickle.load(token_file)
                logger.info("Successfully loaded existing token")
            except Exception as e:
                logger.warning(f"Failed to load existing token: {e}")
                credentials = None

        # Check if credentials are valid and refresh if needed
        if credentials:
            if credentials.expired and credentials.refresh_token:
                logger.info("Token expired, attempting refresh...")
                try:
                    credentials.refresh(Request())
                    logger.info("Token refreshed successfully")
                    self._save_credentials(credentials)
                except Exception as e:
                    logger.warning(f"Failed to refresh token: {e}")
                    credentials = None
            elif not credentials.valid:
                logger.warning("Credentials are invalid")
                credentials = None

        # If no valid credentials, start device flow
        if not credentials:
            logger.info("No valid credentials found, starting device flow authorization...")
            credentials = self._device_flow_authorization()

        return credentials

    def _device_flow_authorization(self) -> Credentials:
        """
        Perform device flow authorization for headless environments
        User authorizes on a different device (PC/phone) using provided URL and code
        """
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"OAuth credentials not found: {self.credentials_path}")

        try:
            # Create flow from client secrets
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_path,
                SCOPES
            )

            # Run device flow (headless-friendly)
            print("\n" + "="*60)
            print("GOOGLE DRIVE AUTHORIZATION REQUIRED")
            print("="*60)
            print("Please complete authorization on another device:")
            print("1. Open the URL below in a web browser (PC/phone)")
            print("2. Enter the device code when prompted")
            print("3. Sign in and authorize access")
            print("="*60)

            # Check if headless environment (RunPod/Jupyter)
            is_headless = (
                os.getenv('RUNPOD_POD_ID') or
                os.getenv('JUPYTER_SERVER_ROOT') or
                not os.environ.get('DISPLAY')
            )

            if is_headless:
                logger.info("Headless environment detected - using manual authorization flow")
                auth_url, _ = flow.authorization_url(prompt='consent')
                print("\n" + "="*80)
                print("MANUAL AUTHORIZATION REQUIRED")
                print("="*80)
                print("1. Copy this URL and open it in your browser (PC/phone):")
                print()
                print(f"   {auth_url}")
                print()
                print("2. Sign in to Google and authorize the application")
                print("3. Copy the authorization code from the browser")
                print("4. Paste it below and press Enter")
                print("="*80)
                auth_code = input("Enter authorization code: ").strip()

                if not auth_code:
                    raise Exception("No authorization code provided")

                credentials = flow.fetch_token(code=auth_code)
            else:
                # Try local server for desktop environments
                try:
                    credentials = flow.run_local_server(port=0)
                except Exception as e:
                    logger.warning(f"Local server auth failed: {e}")
                    # Fallback to manual flow
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    print(f"Please visit this URL to authorize: {auth_url}")
                    auth_code = input("Enter authorization code: ").strip()
                    credentials = flow.fetch_token(code=auth_code)

            print("[ ] Authorization successful!")
            print("="*60 + "\n")

            # Save credentials for future use
            self._save_credentials(credentials)

            logger.info("Device flow authorization completed successfully")
            return credentials

        except Exception as e:
            logger.error(f"Device flow authorization failed: {e}")
            raise

    def _save_credentials(self, credentials: Credentials):
        """Save credentials to persistent storage"""
        try:
            with open(self.token_path, 'wb') as token_file:
                pickle.dump(credentials, token_file)

            # Also save auth state for debugging
            auth_state = {
                'token_saved_at': datetime.now().isoformat(),
                'token_expiry': credentials.expiry.isoformat() if credentials.expiry else None,
                'has_refresh_token': bool(credentials.refresh_token),
                'scopes': list(credentials.scopes) if credentials.scopes else []
            }

            state_path = os.path.join(self.token_dir, 'auth_state.json')
            with open(state_path, 'w') as f:
                json.dump(auth_state, f, indent=2)

            logger.info(f"Credentials saved to {self.token_path}")

        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
            raise

    def _test_connection(self) -> bool:
        """Test Google Drive connection and log user info"""
        try:
            about = self.service.about().get(fields='user,storageQuota').execute()
            user = about.get('user', {})
            storage = about.get('storageQuota', {})

            user_email = user.get('emailAddress', 'Unknown')
            user_name = user.get('displayName', 'Unknown')

            # Storage quota info
            quota_total = int(storage.get('limit', 0))
            quota_used = int(storage.get('usage', 0))

            quota_total_gb = quota_total / (1024**3) if quota_total > 0 else 0
            quota_used_gb = quota_used / (1024**3)

            logger.info(f"Connected as: {user_name} ({user_email})")
            logger.info(f"Storage: {quota_used_gb:.2f} GB used / {quota_total_gb:.2f} GB total")

            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def list_files(self, folder_id: str = None, name_contains: str = None) -> List[Dict[str, Any]]:
        """
        List files in Google Drive folder

        Args:
            folder_id: Google Drive folder ID (None for root)
            name_contains: Filter files by name substring

        Returns:
            List of file dictionaries with id, name, size, mimeType, etc.
        """
        try:
            query_parts = []

            # Folder filter
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")

            # Name filter
            if name_contains:
                query_parts.append(f"name contains '{name_contains}'")

            # Only non-trashed files
            query_parts.append("trashed=false")

            query = " and ".join(query_parts)

            # Execute query
            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="files(id,name,size,mimeType,createdTime,modifiedTime,parents)"
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder {folder_id or 'root'}")

            return files

        except HttpError as e:
            logger.error(f"Error listing files in folder {folder_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing files: {e}")
            return []

    def upload_file(self, local_path: str, folder_id: str,
                   drive_filename: str = None, progress_callback=None) -> Optional[str]:
        """
        Upload file to Google Drive

        Args:
            local_path: Local file path to upload
            folder_id: Target Google Drive folder ID
            drive_filename: Filename in Drive (defaults to local filename)
            progress_callback: Optional callback for upload progress

        Returns:
            Google Drive file ID if successful, None otherwise
        """
        try:
            if not os.path.exists(local_path):
                logger.error(f"Local file not found: {local_path}")
                return None

            filename = drive_filename or os.path.basename(local_path)
            file_size = os.path.getsize(local_path)

            logger.info(f"Uploading {filename} ({file_size:,} bytes) to folder {folder_id}")

            # File metadata
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }

            # Upload media
            media = MediaFileUpload(local_path, resumable=True)

            # Create upload request
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, size'
            )

            # Execute resumable upload
            file_info = None
            while file_info is None:
                try:
                    status, file_info = request.next_chunk()
                    if status and progress_callback:
                        progress_callback(status.progress())
                    elif status:
                        logger.debug(f"Upload progress: {int(status.progress() * 100)}%")
                except HttpError as chunk_error:
                    logger.error(f"Upload chunk failed: {chunk_error}")
                    return None

            file_id = file_info.get('id')
            uploaded_size = file_info.get('size', 'unknown')

            logger.info(f"[OK] Upload successful: {filename} (ID: {file_id}, Size: {uploaded_size} bytes)")

            return file_id

        except HttpError as e:
            logger.error(f"HTTP error uploading {local_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path}: {e}")
            return None

    def download_file(self, file_id: str, local_path: str) -> bool:
        """
        Download file from Google Drive

        Args:
            file_id: Google Drive file ID
            local_path: Local destination path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get file info first
            file_info = self.service.files().get(fileId=file_id, fields='name,size').execute()
            filename = file_info.get('name', 'unknown')
            filesize = int(file_info.get('size', 0))

            logger.info(f"Downloading {filename} ({filesize:,} bytes) to {local_path}")

            # Create download request
            request = self.service.files().get_media(fileId=file_id)

            # Download with progress tracking
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(f"Download progress: {int(status.progress() * 100)}%")

            # Verify downloaded file size
            actual_size = os.path.getsize(local_path)
            if actual_size == filesize:
                logger.info(f"[OK] Download successful: {filename}")
                return True
            else:
                logger.warning(f"Size mismatch: expected {filesize}, got {actual_size}")
                return False

        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return False

    def create_folder(self, folder_name: str, parent_folder_id: str) -> Optional[str]:
        """
        Create folder in Google Drive

        Args:
            folder_name: Name of new folder
            parent_folder_id: Parent folder ID

        Returns:
            New folder ID if successful, None otherwise
        """
        try:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }

            folder = self.service.files().create(
                body=folder_metadata,
                fields='id, name'
            ).execute()

            folder_id = folder.get('id')
            logger.info(f"Created folder '{folder_name}' (ID: {folder_id})")

            return folder_id

        except Exception as e:
            logger.error(f"Error creating folder '{folder_name}': {e}")
            return None

    def move_file(self, file_id: str, new_parent_id: str) -> bool:
        """
        Move file to different folder

        Args:
            file_id: Google Drive file ID
            new_parent_id: Target folder ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current parents
            file_info = self.service.files().get(fileId=file_id, fields='name,parents').execute()
            filename = file_info.get('name', 'unknown')
            previous_parents = ','.join(file_info.get('parents', []))

            # Move file
            self.service.files().update(
                fileId=file_id,
                addParents=new_parent_id,
                removeParents=previous_parents,
                fields='id, parents'
            ).execute()

            logger.info(f"Moved file '{filename}' (ID: {file_id}) to folder {new_parent_id}")
            return True

        except Exception as e:
            logger.error(f"Error moving file {file_id}: {e}")
            return False

    def delete_file(self, file_id: str) -> bool:
        """
        Delete file from Google Drive

        Args:
            file_id: Google Drive file ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get filename for logging
            try:
                file_info = self.service.files().get(fileId=file_id, fields='name').execute()
                filename = file_info.get('name', 'unknown')
            except:
                filename = 'unknown'

            # Delete file
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file '{filename}' (ID: {file_id})")

            return True

        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False

    def get_auth_info(self) -> Dict[str, Any]:
        """Get authentication status information for debugging"""
        auth_info = {
            'credentials_path': self.credentials_path,
            'token_dir': self.token_dir,
            'token_exists': os.path.exists(self.token_path),
            'credentials_valid': self.credentials.valid if self.credentials else False,
            'credentials_expired': self.credentials.expired if self.credentials else None,
            'has_refresh_token': bool(self.credentials.refresh_token) if self.credentials else False,
        }

        if self.credentials and self.credentials.expiry:
            auth_info['token_expiry'] = self.credentials.expiry.isoformat()
            auth_info['expires_in_minutes'] = int((self.credentials.expiry - datetime.utcnow()).total_seconds() / 60)

        return auth_info


def test_oauth_client(credentials_path: str = None, token_dir: str = None) -> bool:
    """
    Test OAuth client functionality

    Args:
        credentials_path: Path to oauth_credentials.json
        token_dir: Directory for token storage

    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "="*50)
    print("TESTING GOOGLE DRIVE OAUTH CLIENT")
    print("="*50)

    try:
        # Initialize client
        print("1. Initializing OAuth client...")
        client = GoogleDriveOAuthClient(credentials_path, token_dir)

        # Test connection
        print("2. Testing connection...")
        if not client._test_connection():
            print("[ERROR] Connection test failed")
            return False
        print("[OK] Connection test passed")

        # Test list files in root
        print("3. Testing file listing...")
        files = client.list_files()
        print(f"[OK] Found {len(files)} files in root folder")

        # Show auth info
        print("4. Authentication info:")
        auth_info = client.get_auth_info()
        for key, value in auth_info.items():
            print(f"   {key}: {value}")

        print("\n[OK] All tests passed!")
        print("="*50 + "\n")

        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        print("="*50 + "\n")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test with custom paths if provided
            creds_path = sys.argv[2] if len(sys.argv) > 2 else None
            token_dir = sys.argv[3] if len(sys.argv) > 3 else None
            test_oauth_client(creds_path, token_dir)
        elif sys.argv[1] == "--authorize":
            # Force re-authorization
            token_dir = sys.argv[2] if len(sys.argv) > 2 else "/workspace/persistent/oauth"
            token_path = os.path.join(token_dir, 'token.pickle')
            if os.path.exists(token_path):
                os.remove(token_path)
                print(f"Removed existing token: {token_path}")
            client = GoogleDriveOAuthClient(token_dir=token_dir)
        else:
            print("Usage: python google_drive_oauth_client.py [--test|--authorize] [credentials_path] [token_dir]")
    else:
        # Default test
        test_oauth_client()