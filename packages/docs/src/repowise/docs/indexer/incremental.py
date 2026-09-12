"""Hash-based incremental change detection for doc indexing."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from repowise.docs.library.models import LibraryFile

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file's content.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA256 hash

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks for memory efficiency
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


@dataclass
class FileInfo:
    """Information about a file for change detection."""

    path: str  # Relative path within repo
    content_hash: str


@dataclass
class ChangeSet:
    """Result of change detection between current and stored files."""

    added: set[str] = field(default_factory=set)
    modified: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    unchanged: set[str] = field(default_factory=set)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added or self.modified or self.deleted)

    @property
    def total_to_process(self) -> int:
        """Total number of files that need processing (added + modified)."""
        return len(self.added) + len(self.modified)

    def __str__(self) -> str:
        return (
            f"ChangeSet(added={len(self.added)}, modified={len(self.modified)}, "
            f"deleted={len(self.deleted)}, unchanged={len(self.unchanged)})"
        )


def compute_changes(
    current_files: list[FileInfo],
    stored_files: list[LibraryFile],
) -> ChangeSet:
    """
    Compute changes between current filesystem state and stored database records.

    Args:
        current_files: List of current files with their hashes
        stored_files: List of previously indexed files from database

    Returns:
        ChangeSet indicating which files were added, modified, deleted, or unchanged
    """
    # Build lookup maps
    current_map: dict[str, str] = {f.path: f.content_hash for f in current_files}
    stored_map: dict[str, str] = {f.file_path: f.content_hash for f in stored_files}

    changes = ChangeSet()

    # Find added and modified files
    for path, current_hash in current_map.items():
        if path not in stored_map:
            changes.added.add(path)
        elif stored_map[path] != current_hash:
            changes.modified.add(path)
        else:
            changes.unchanged.add(path)

    # Find deleted files
    for path in stored_map:
        if path not in current_map:
            changes.deleted.add(path)

    logger.debug(f"Change detection: {changes}")

    return changes


def compute_files_info(repo_dir: Path, file_paths: list[Path]) -> list[FileInfo]:
    """
    Compute hashes for a list of files.

    Args:
        repo_dir: Root directory of the repository
        file_paths: List of relative file paths

    Returns:
        List of FileInfo with paths and hashes
    """
    result = []

    for rel_path in file_paths:
        abs_path = repo_dir / rel_path
        if abs_path.exists():
            try:
                content_hash = compute_file_hash(abs_path)
                result.append(FileInfo(path=str(rel_path), content_hash=content_hash))
            except OSError as e:
                logger.warning(f"Failed to hash {rel_path}: {e}")

    return result
