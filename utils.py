import logging
import uuid
from pathlib import Path


def generate_unique_filename() -> str:
    """
    Generate a unique filename using a 6-character hexadecimal string based on UUID version 4.
    Returns:
        A unique filename as a 6-character hexadecimal string.
    """
    return uuid.uuid4().hex[:6]


def file_ext(filename: str) -> str:
    """
    Extract the file extension from a given filename.
    Args:
        filename (str): The name of the file.
    Returns:
        str: The file extension, including the leading dot.
    """
    try:
        return Path(filename).suffix
    except Exception as e:
        raise ValueError(f"Invalid filename: {filename}") from e


def check_file_exists(file_path: str, file_type: str) -> None:
    if not Path(file_path).exists():
        logging.error(f"{file_type} file {file_path} does not exist.")
        raise FileNotFoundError(f"{file_type} file {file_path} does not exist.")


def delete_file(file_path: str) -> None:
    try:
        if Path(file_path).exists():
            Path(file_path).unlink()
            logging.info(f"Deleted {file_path}")
    except Exception as e:
        logging.error(f"Failed to delete {file_path}: {e}")
