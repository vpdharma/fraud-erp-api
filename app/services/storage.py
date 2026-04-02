import os
from pathlib import Path

from azure.core.exceptions import AzureError, ResourceExistsError
from azure.storage.blob import BlobServiceClient


DEFAULT_OUTPUT_CONTAINER = "outputs"


def get_storage_connection_string() -> str | None:
    """
    Read the Azure Storage connection string from an environment variable.

    Environment variable = a setting stored outside the code.
    This is safer than writing secrets directly in Python files.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if connection_string:
        connection_string = connection_string.strip()

    return connection_string or None


def is_azure_storage_configured() -> bool:
    """
    Return True only when the required connection string exists.
    """
    return get_storage_connection_string() is not None


def upload_file_to_blob(
    local_file_path: str | Path,
    container_name: str = DEFAULT_OUTPUT_CONTAINER,
    blob_name: str | None = None,
) -> dict:
    """
    Upload a local file to Azure Blob Storage.

    This function never crashes the whole app on configuration problems.
    Instead, it returns a simple status dictionary that the API can show.
    """
    local_file_path = Path(local_file_path)

    if not local_file_path.exists():
        return {
            "uploaded": False,
            "message": f"Local file not found: {local_file_path}",
            "blob_url": None,
        }

    connection_string = get_storage_connection_string()
    if not connection_string:
        return {
            "uploaded": False,
            "message": "Azure Blob Storage is not configured locally.",
            "blob_url": None,
        }

    if not blob_name:
        blob_name = local_file_path.name

    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        container_client = blob_service_client.get_container_client(container_name)
        container_client.create_container()
    except ResourceExistsError:
        # The container may already exist. That is fine, so we continue.
        pass
    except AzureError as error:
        return {
            "uploaded": False,
            "message": f"Could not prepare Azure container: {error}",
            "blob_url": None,
        }
    except Exception as error:
        return {
            "uploaded": False,
            "message": f"Could not prepare Azure container: {error}",
            "blob_url": None,
        }

    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name,
        )

        with local_file_path.open("rb") as file_handle:
            blob_client.upload_blob(file_handle, overwrite=True)

        return {
            "uploaded": True,
            "message": "File uploaded to Azure Blob Storage successfully.",
            "blob_url": blob_client.url,
        }
    except Exception as error:
        return {
            "uploaded": False,
            "message": f"Azure upload skipped or failed: {error}",
            "blob_url": None,
        }
