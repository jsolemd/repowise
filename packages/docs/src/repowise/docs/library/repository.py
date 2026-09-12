"""Repository cache lifecycle helpers for doc-search git management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from repowise.docs.config import get_settings

from .discovery import discover_docs_path, discover_docs_path_from_git, list_doc_files
from .git_exec import GitError, run_git

logger = logging.getLogger(__name__)


def get_repo_path(repo: str) -> Path:
    """Get the local cache path for a repository."""
    settings = get_settings()
    return settings.cache_root / "repos" / repo


def get_library_root_path(repo: str, source_subpath: str = "") -> Path:
    """Get the logical library root inside the cached repository clone."""
    repo_path = get_repo_path(repo)
    if not source_subpath:
        return repo_path
    return repo_path / source_subpath


def delete_repo_cache(repo: str) -> bool:
    """Delete the cached git clone for a repository."""
    import shutil

    repo_path = get_repo_path(repo)
    if repo_path.exists():
        shutil.rmtree(repo_path)
        logger.info("Deleted git cache for %s at %s", repo, repo_path)
        return True
    logger.info("No git cache found for %s at %s", repo, repo_path)
    return False


def _get_repo_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


async def clone_repo(
    repo: str,
    branch: str = "main",
    docs_path: str = "",
    source_subpath: str = "",
    prefer_full_checkout: bool = False,
    force: bool = False,
) -> Path:
    """Clone a repository with shallow clone and optional sparse checkout."""
    repo_dir = get_repo_path(repo)

    if repo_dir.exists() and (repo_dir / ".git").exists() and not force:
        logger.info("Repo %s exists, fetching latest", repo)
        if source_subpath or prefer_full_checkout:
            await _disable_sparse_checkout(repo_dir)
        await fetch_latest(repo_dir, branch)
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if force and repo_dir.exists():
        import shutil

        shutil.rmtree(repo_dir)

    url = _get_repo_url(repo)
    use_sparse_checkout = bool(docs_path) and not source_subpath and not prefer_full_checkout
    logger.info(
        "Cloning %s (branch=%s, sparse=%s, source_subpath=%s)",
        repo,
        branch,
        use_sparse_checkout,
        source_subpath or ".",
    )

    if use_sparse_checkout:
        await run_git(
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--no-checkout",
            "--branch",
            branch,
            "--single-branch",
            url,
            str(repo_dir),
        )
        logger.info("Setting up sparse checkout for %s", docs_path)
        await run_git("sparse-checkout", "init", "--cone", cwd=repo_dir)
        await run_git("sparse-checkout", "set", docs_path, cwd=repo_dir)
        await run_git("checkout", branch, cwd=repo_dir)
    else:
        await run_git(
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            "--single-branch",
            url,
            str(repo_dir),
        )

    return repo_dir


async def resolve_repo_docs_path(
    repo_dir: Path,
    branch: str = "main",
    docs_path: str = "",
    source_subpath: str = "",
) -> str:
    """Resolve and, when needed, repair the checked-out docs path for a repo."""
    normalized_docs_path = docs_path.strip()
    library_root = get_library_root_path_from_repo_dir(repo_dir, source_subpath)
    if source_subpath and not library_root.exists():
        raise GitError(
            command="resolve-docs-path",
            return_code=-1,
            stderr=f"Source subpath not found in repository: {source_subpath}",
        )
    if not normalized_docs_path:
        return normalized_docs_path

    preferred_full = library_root / normalized_docs_path
    if preferred_full.exists():
        return normalized_docs_path

    if source_subpath:
        discovered = discover_docs_path(library_root, normalized_docs_path)
        if discovered is None:
            logger.warning(
                "Configured docs path '%s' missing in %s and no alternative was discovered",
                normalized_docs_path,
                library_root,
            )
            return normalized_docs_path
        logger.info(
            "Adjusted docs path for %s from '%s' to '%s'",
            library_root,
            normalized_docs_path,
            discovered or ".",
        )
        return discovered

    try:
        discovered = await discover_docs_path_from_git(repo_dir, normalized_docs_path)
    except GitError as exc:
        logger.warning(
            "Could not inspect git tree for docs path recovery in %s: %s",
            repo_dir,
            exc,
        )
        return normalized_docs_path
    if discovered is None:
        logger.warning(
            "Configured docs path '%s' missing in %s and no alternative was discovered",
            normalized_docs_path,
            repo_dir,
        )
        return normalized_docs_path

    logger.info(
        "Adjusted docs path for %s from '%s' to '%s'",
        repo_dir,
        normalized_docs_path,
        discovered or ".",
    )
    if discovered:
        await run_git("sparse-checkout", "set", discovered, cwd=repo_dir)
    else:
        await run_git("sparse-checkout", "disable", cwd=repo_dir)
    await run_git("checkout", branch, cwd=repo_dir)
    return discovered


async def fetch_latest(repo_dir: Path, branch: str = "main") -> None:
    """Fetch latest changes for a repository."""
    if not (repo_dir / ".git").exists():
        raise GitError(command="fetch", return_code=-1, stderr=f"Not a git repository: {repo_dir}")

    logger.info("Fetching latest for %s", repo_dir)
    # Explicit refspec: a --single-branch clone only tracks the branch it was
    # created with, so a bare ``fetch origin <branch>`` after a branch change
    # lands in FETCH_HEAD and the ``origin/<branch>`` reset below fails.
    await run_git(
        "fetch",
        "--depth",
        "1",
        "origin",
        f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
        cwd=repo_dir,
    )
    await run_git("reset", "--hard", f"origin/{branch}", cwd=repo_dir)


async def get_head_sha(repo_dir: Path) -> str:
    """Get the current HEAD commit SHA."""
    if not (repo_dir / ".git").exists():
        raise GitError(
            command="rev-parse", return_code=-1, stderr=f"Not a git repository: {repo_dir}"
        )

    return (await run_git("rev-parse", "HEAD", cwd=repo_dir)).strip()


async def get_source_ref(
    repo_dir: Path,
    source_subpath: str = "",
    head_sha: str | None = None,
) -> str:
    """Get a stable source ref for the indexed scope.

    Whole-repo libraries use the HEAD commit SHA. Subpath libraries use the git
    object SHA for that subtree so unrelated commits elsewhere do not trigger
    reindexing.
    """
    if not source_subpath:
        return head_sha or await get_head_sha(repo_dir)
    return (await run_git("rev-parse", f"HEAD:{source_subpath}", cwd=repo_dir)).strip()


def get_library_root_path_from_repo_dir(repo_dir: Path, source_subpath: str = "") -> Path:
    """Get the logical library root from an existing cached repository path."""
    if not source_subpath:
        return repo_dir
    return repo_dir / source_subpath


async def _disable_sparse_checkout(repo_dir: Path) -> None:
    try:
        await run_git("sparse-checkout", "disable", cwd=repo_dir)
    except GitError:
        return


@dataclass
class RepoInfo:
    """Information about a cloned repository."""

    repo: str
    path: Path
    head_sha: str
    source_ref: str
    file_count: int


async def prepare_repo(
    repo: str,
    branch: str = "main",
    docs_path: str = "",
    source_subpath: str = "",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> RepoInfo:
    """Prepare a repository for indexing: clone/fetch and list files."""
    repo_dir = await clone_repo(repo, branch, docs_path, source_subpath)
    resolved_docs_path = await resolve_repo_docs_path(
        repo_dir,
        branch,
        docs_path,
        source_subpath,
    )
    library_root = get_library_root_path_from_repo_dir(repo_dir, source_subpath)
    head_sha = await get_head_sha(repo_dir)
    source_ref = await get_source_ref(repo_dir, source_subpath)
    files = list_doc_files(library_root, resolved_docs_path, include_patterns, exclude_patterns)

    logger.info("Prepared %s: %s files at %s", repo, len(files), source_ref[:8])
    return RepoInfo(
        repo=repo,
        path=repo_dir,
        head_sha=head_sha,
        source_ref=source_ref,
        file_count=len(files),
    )
