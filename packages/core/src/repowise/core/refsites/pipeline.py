"""Repository-level reference-site extraction, with measured coverage.

Coverage is *measured here*, not asserted from a table. ``files_seen`` counts
what the traverser handed us; ``sites`` counts what came back. A language with
files and no sites therefore leaves a row that says so, which is the whole
mechanism by which "we cannot read YAML" stops being indistinguishable from
"this symbol is unreferenced".
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from repowise.core.ingestion.models import ParsedFile
from repowise.core.ingestion.parser import ASTParser
from repowise.core.ingestion.traverser import FileTraverser
from repowise.core.refsites.coverage import CoverageStatus, coverage_for
from repowise.core.refsites.extraction import extract_file
from repowise.core.refsites.records import CoverageRecord, ExtractionResult, ReferenceSiteRecord
from repowise.core.refsites.universe import Universe, build_universe


def _decode(source: bytes) -> str:
    """Decode for the lexer, never failing on a stray byte.

    Positions are character offsets throughout this package. That is a
    deliberate divergence from tree-sitter's byte columns: the consumers are
    editors and rename previews, which count characters, and we never mix the
    two because no tree-sitter column ever enters this package.
    """
    return source.decode("utf-8", errors="replace")


def parse_repository(repo_path: Path) -> tuple[list[ParsedFile], dict[str, str]]:
    """Parse every traversable file, returning parsed files and decoded sources.

    Uses the parse layer's public entry point (``ASTParser.parse_file``) so
    grammar selection, the adaptive TSX fallback and single-file-component
    source projection all stay owned by that layer.
    """
    traverser = FileTraverser(repo_path)
    parser = ASTParser()
    parsed: list[ParsedFile] = []
    sources: dict[str, str] = {}
    for file_info in traverser.traverse():
        raw = Path(file_info.abs_path).read_bytes()
        parsed.append(parser.parse_file(file_info, raw))
        sources[file_info.path] = _decode(raw)
    return parsed, sources


def extract_sites(
    parsed: list[ParsedFile],
    sources: dict[str, str],
    universe: Universe,
    *,
    sweep_textual: bool = True,
) -> tuple[ReferenceSiteRecord, ...]:
    """Extract sites for every parsed file against a prebuilt universe."""
    sites: list[ReferenceSiteRecord] = []
    for parsed_file in parsed:
        sites.extend(
            extract_file(
                parsed_file,
                sources.get(parsed_file.file_info.path, ""),
                universe,
                sweep_textual=sweep_textual,
            )
        )
    return tuple(sites)


def measure_coverage(
    parsed: list[ParsedFile], sites: tuple[ReferenceSiteRecord, ...]
) -> tuple[CoverageRecord, ...]:
    """Build one coverage record per language actually present in the repository."""
    files_seen = Counter(parsed_file.file_info.language for parsed_file in parsed)
    site_counts = Counter(site.language for site in sites)
    files_with_sites = Counter(
        {
            language: len({site.file_path for site in sites if site.language == language})
            for language in site_counts
        }
    )

    records: list[CoverageRecord] = []
    for language in sorted(files_seen):
        claim = coverage_for(language)
        produced = site_counts.get(language, 0)
        reason = claim.reason
        if claim.status is not CoverageStatus.NONE and produced == 0:
            # Declared as covered but produced nothing here. Say that plainly
            # rather than letting the declaration imply a measurement.
            reason = f"{claim.reason}; no site found in this repository"
        records.append(
            CoverageRecord(
                language=language,
                status=claim.status,
                files_seen=files_seen[language],
                files_extracted=files_with_sites.get(language, 0),
                sites=produced,
                reason=reason,
                limitations=claim.limitations,
            )
        )
    return tuple(records)


def extract_repository(repo_path: Path | str, *, sweep_textual: bool = True) -> ExtractionResult:
    """Parse *repo_path* and extract every reference site, with coverage.

    ``sweep_textual=False`` restricts the result to AST-attested sites; see
    :func:`repowise.core.refsites.extraction.extract_file` for what that costs.
    """
    root = Path(repo_path)
    parsed, sources = parse_repository(root)
    universe = build_universe(root, parsed)
    sites = extract_sites(parsed, sources, universe, sweep_textual=sweep_textual)
    return ExtractionResult(sites=sites, coverage=measure_coverage(parsed, sites))


__all__ = [
    "extract_repository",
    "extract_sites",
    "measure_coverage",
    "parse_repository",
]
