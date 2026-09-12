"""Canonical import barrel for doc-search git management."""

from __future__ import annotations

from .discovery import (
    COMMON_DOCS_PATHS,
    discover_docs_path,
    discover_docs_path_from_git,
    get_related_docs_repos,
    list_doc_files,
)
from .git_exec import GitError, _is_permanent_error, _is_transient_error
from .remote import get_remote_head_sha, resolve_default_branch
from .repository import (
    RepoInfo,
    clone_repo,
    delete_repo_cache,
    fetch_latest,
    get_head_sha,
    get_library_root_path,
    get_library_root_path_from_repo_dir,
    get_repo_path,
    get_source_ref,
    prepare_repo,
    resolve_repo_docs_path,
)

__all__ = [
    "COMMON_DOCS_PATHS",
    "GitError",
    "RepoInfo",
    "_is_permanent_error",
    "_is_transient_error",
    "clone_repo",
    "delete_repo_cache",
    "discover_docs_path",
    "discover_docs_path_from_git",
    "fetch_latest",
    "get_head_sha",
    "get_library_root_path",
    "get_library_root_path_from_repo_dir",
    "get_related_docs_repos",
    "get_remote_head_sha",
    "get_repo_path",
    "get_source_ref",
    "list_doc_files",
    "prepare_repo",
    "resolve_default_branch",
    "resolve_repo_docs_path",
]
