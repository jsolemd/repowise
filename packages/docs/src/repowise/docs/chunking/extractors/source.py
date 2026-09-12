"""Structure-aware chunking for source and configuration files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from repowise.docs.chunking.extractors.ast_symbols import extract_ast_declarations
from repowise.docs.chunking.extractors.text_split import split_large_section_with_line_spans
from repowise.docs.chunking.models import ChunkType, DocChunk, build_source_url, slugify
from repowise.docs.config import get_settings
from repowise.docs.formats import detect_config_language, detect_source_language

JAVA_TYPE_DECL_PATTERN = re.compile(
    r"^\s*(?:public|protected|private|abstract|final|sealed|non-sealed|static|\s)*"
    r"\b(class|interface|enum|record)\s+([A-Za-z_]\w*)\b"
)

JAVA_METHOD_DECL_PATTERN = re.compile(
    r"^\s*(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default|\s)+"
    r"[\w<>\[\], ?.@]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:\{|throws\b)"
)

JAVA_CONSTRUCTOR_DECL_PATTERN = re.compile(
    r"^\s*(?:public|protected|private)\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:\{|throws\b)"
)

JAVA_CONTROL_PATTERN = re.compile(r"^\s*(?:if|for|while|switch|catch|do|try|else)\b")

GRADLE_BLOCK_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*(?:\([^\)]*\))?)\s*\{\s*$")

GENERIC_CONTROL_PATTERN = re.compile(
    r"^\s*(?:if|for|while|switch|catch|do|try|else|return|break|continue)\b"
)

PY_CLASS_PATTERN = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\b")
PY_DEF_PATTERN = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")

JS_CLASS_PATTERN = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)\b")
JS_FUNCTION_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("
)
JS_ARROW_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^\)]*\)|[A-Za-z_$][\w$]*)\s*=>"
)

TS_INTERFACE_PATTERN = re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)\b")
TS_TYPE_PATTERN = re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\b")

GO_FUNC_PATTERN = re.compile(r"^\s*func\s+(?:\([^\)]*\)\s*)?([A-Za-z_]\w*)\s*\(")
GO_TYPE_PATTERN = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\b")

RUST_FN_PATTERN = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_]\w*)\b")
RUST_TYPE_PATTERN = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait|impl)\s+([A-Za-z_]\w*)\b")

C_STYLE_PATTERN = re.compile(
    r"^\s*(?:[A-Za-z_][\w:<>,\s\*\[\]]+\s+)?([A-Za-z_][\w:]*)\s*\([^;]*\)\s*\{\s*$"
)

INI_SECTION_PATTERN = re.compile(r"^\s*\[([^\]]+)\]\s*$")
YAML_TOP_KEY_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)\s*:\s*(?:#.*)?$")
JSON_TOP_KEY_PATTERN = re.compile(r'^\s*"([^\"]+)"\s*:\s*')
HCL_BLOCK_PATTERN = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*(?:"[^\"]+"\s*)*(?:\{|=)')
KEY_VALUE_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*")
XML_TAG_PATTERN = re.compile(r"^\s*<([A-Za-z][A-Za-z0-9:._-]*)\b")

LANGUAGE_PATTERNS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "python": [(PY_CLASS_PATTERN, "class"), (PY_DEF_PATTERN, "function")],
    "javascript": [
        (JS_CLASS_PATTERN, "class"),
        (JS_FUNCTION_PATTERN, "function"),
        (JS_ARROW_PATTERN, "function"),
    ],
    "jsx": [
        (JS_CLASS_PATTERN, "class"),
        (JS_FUNCTION_PATTERN, "function"),
        (JS_ARROW_PATTERN, "function"),
    ],
    "typescript": [
        (TS_INTERFACE_PATTERN, "interface"),
        (TS_TYPE_PATTERN, "type"),
        (JS_CLASS_PATTERN, "class"),
        (JS_FUNCTION_PATTERN, "function"),
        (JS_ARROW_PATTERN, "function"),
    ],
    "tsx": [
        (TS_INTERFACE_PATTERN, "interface"),
        (TS_TYPE_PATTERN, "type"),
        (JS_CLASS_PATTERN, "class"),
        (JS_FUNCTION_PATTERN, "function"),
        (JS_ARROW_PATTERN, "function"),
    ],
    "go": [(GO_TYPE_PATTERN, "type"), (GO_FUNC_PATTERN, "function")],
    "rust": [(RUST_TYPE_PATTERN, "type"), (RUST_FN_PATTERN, "function")],
    "c": [(C_STYLE_PATTERN, "function")],
    "cpp": [(C_STYLE_PATTERN, "function")],
    "csharp": [(C_STYLE_PATTERN, "function")],
    "swift": [(C_STYLE_PATTERN, "function")],
    "kotlin": [(C_STYLE_PATTERN, "function")],
    "scala": [(C_STYLE_PATTERN, "function")],
    "groovy": [(C_STYLE_PATTERN, "function")],
    "ruby": [(PY_DEF_PATTERN, "function")],
    "php": [(C_STYLE_PATTERN, "function")],
}


@dataclass
class SourceSegment:
    """Logical code/config segment boundaries for chunk creation."""

    name: str
    kind: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    breadcrumb: list[str]
    metadata: dict[str, str]


def _build_segments_from_declarations(
    lines: list[str],
    file_path: str,
    declarations: list[tuple[int, str, str]],
    chunk_type: ChunkType,
    language: str,
    type_aware_breadcrumbs: bool = False,
) -> list[SourceSegment]:
    stem = Path(file_path).stem
    fallback_name = Path(file_path).name if chunk_type == ChunkType.CONFIG else stem

    if not declarations:
        kind = "config" if chunk_type == ChunkType.CONFIG else "source"
        return [
            SourceSegment(
                name=fallback_name,
                kind=kind,
                start_line=0,
                end_line=len(lines),
                chunk_type=chunk_type,
                breadcrumb=[fallback_name],
                metadata={"language": language},
            )
        ]

    segments: list[SourceSegment] = []
    first_start = declarations[0][0]

    if first_start > 0:
        overview_kind = "overview"
        segments.append(
            SourceSegment(
                name="file-overview",
                kind=overview_kind,
                start_line=0,
                end_line=first_start,
                chunk_type=chunk_type,
                breadcrumb=[fallback_name, overview_kind],
                metadata={"language": language, "symbol_kind": overview_kind},
            )
        )

    current_type: str | None = None

    for idx, (start_line, name, kind) in enumerate(declarations):
        end_line = declarations[idx + 1][0] if idx + 1 < len(declarations) else len(lines)
        if end_line <= start_line:
            end_line = min(start_line + 1, len(lines))

        if type_aware_breadcrumbs and kind in {"class", "interface", "enum", "record"}:
            current_type = name
            breadcrumb = [name]
        elif type_aware_breadcrumbs and current_type and kind in {"method", "constructor"}:
            breadcrumb = [current_type, name]
        else:
            breadcrumb = [fallback_name, name]

        segments.append(
            SourceSegment(
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                chunk_type=chunk_type,
                breadcrumb=breadcrumb,
                metadata={
                    "language": language,
                    "symbol_name": name,
                    "symbol_kind": kind,
                },
            )
        )

    return segments


def _detect_ast_code_segments(
    content: str,
    lines: list[str],
    file_path: str,
    language: str,
    type_aware_breadcrumbs: bool = True,
) -> list[SourceSegment] | None:
    declarations = extract_ast_declarations(content, language)
    if not declarations:
        return None

    segments = _build_segments_from_declarations(
        lines=lines,
        file_path=file_path,
        declarations=declarations,
        chunk_type=ChunkType.CODE,
        language=language,
        type_aware_breadcrumbs=type_aware_breadcrumbs,
    )
    for segment in segments:
        segment.metadata["parser"] = "ast"
    return segments


def _detect_java_segments(lines: list[str], file_path: str) -> list[SourceSegment]:
    declarations: list[tuple[int, str, str]] = []

    for idx, line in enumerate(lines):
        if JAVA_CONTROL_PATTERN.match(line):
            continue

        type_match = JAVA_TYPE_DECL_PATTERN.match(line)
        if type_match:
            declarations.append((idx, type_match.group(2), type_match.group(1)))
            continue

        method_match = JAVA_METHOD_DECL_PATTERN.match(line)
        if method_match:
            declarations.append((idx, method_match.group(1), "method"))
            continue

        ctor_match = JAVA_CONSTRUCTOR_DECL_PATTERN.match(line)
        if ctor_match:
            declarations.append((idx, ctor_match.group(1), "constructor"))

    return _build_segments_from_declarations(
        lines,
        file_path,
        declarations,
        ChunkType.CODE,
        language="java",
        type_aware_breadcrumbs=True,
    )


def _detect_gradle_segments(lines: list[str], file_path: str) -> list[SourceSegment]:
    blocks: list[tuple[int, int, str]] = []
    active_block_start: int | None = None
    active_block_name = ""
    brace_depth = 0

    for idx, line in enumerate(lines):
        if active_block_start is None and brace_depth == 0:
            match = GRADLE_BLOCK_PATTERN.match(line)
            if match:
                active_block_start = idx
                active_block_name = match.group(1).strip()

        brace_depth += line.count("{") - line.count("}")

        if active_block_start is not None and brace_depth == 0:
            blocks.append((active_block_start, idx + 1, active_block_name))
            active_block_start = None
            active_block_name = ""

    if active_block_start is not None:
        blocks.append((active_block_start, len(lines), active_block_name or "block"))

    stem = Path(file_path).name
    if not blocks:
        return [
            SourceSegment(
                name=stem,
                kind="config",
                start_line=0,
                end_line=len(lines),
                chunk_type=ChunkType.CONFIG,
                breadcrumb=[stem],
                metadata={"language": "gradle", "symbol_kind": "script"},
            )
        ]

    segments: list[SourceSegment] = []
    if blocks[0][0] > 0:
        segments.append(
            SourceSegment(
                name="file-overview",
                kind="overview",
                start_line=0,
                end_line=blocks[0][0],
                chunk_type=ChunkType.CONFIG,
                breadcrumb=[stem, "overview"],
                metadata={"language": "gradle", "symbol_kind": "overview"},
            )
        )

    for start, end, name in blocks:
        segments.append(
            SourceSegment(
                name=name,
                kind="block",
                start_line=start,
                end_line=end,
                chunk_type=ChunkType.CONFIG,
                breadcrumb=[stem, name],
                metadata={"language": "gradle", "symbol_kind": "block", "symbol_name": name},
            )
        )

    return segments


def _detect_generic_code_symbol(line: str, language: str) -> tuple[str, str] | None:
    if GENERIC_CONTROL_PATTERN.match(line):
        return None

    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("#", "//", "/*", "*", "--")):
        return None

    for pattern, kind in LANGUAGE_PATTERNS.get(language, []):
        match = pattern.match(line)
        if match:
            return match.group(1), kind

    return None


def _detect_generic_code_segments(
    lines: list[str],
    file_path: str,
    language: str,
) -> list[SourceSegment]:
    declarations: list[tuple[int, str, str]] = []

    for idx, line in enumerate(lines):
        symbol = _detect_generic_code_symbol(line, language)
        if symbol:
            name, kind = symbol
            declarations.append((idx, name, kind))

    return _build_segments_from_declarations(
        lines,
        file_path,
        declarations,
        ChunkType.CODE,
        language=language,
    )


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _detect_generic_config_symbol(line: str, language: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("#", ";", "//", "/*", "*", "--")):
        return None

    indent = _indent_width(line)

    if language in {"yaml", "dotenv", "properties", "config"} and indent == 0:
        key_match = YAML_TOP_KEY_PATTERN.match(line)
        if key_match:
            return key_match.group(1), "section"

        env_match = KEY_VALUE_PATTERN.match(line)
        if env_match:
            return env_match.group(1), "key"

    if language in {"json", "json5", "jsonc"} and indent <= 2:
        json_match = JSON_TOP_KEY_PATTERN.match(line)
        if json_match:
            return json_match.group(1), "object"

    if language in {"toml", "ini"}:
        section_match = INI_SECTION_PATTERN.match(line)
        if section_match:
            return section_match.group(1).strip(), "section"

        if indent == 0:
            env_match = KEY_VALUE_PATTERN.match(line)
            if env_match:
                return env_match.group(1), "key"

    if language in {"hcl", "terraform"}:
        hcl_match = HCL_BLOCK_PATTERN.match(line)
        if hcl_match and indent == 0:
            return hcl_match.group(1), "block"

    if language in {"xml", "html", "xhtml"} and indent == 0:
        xml_match = XML_TAG_PATTERN.match(line)
        if xml_match:
            return xml_match.group(1), "tag"

    return None


def _detect_generic_config_segments(
    lines: list[str],
    file_path: str,
    language: str,
) -> list[SourceSegment]:
    declarations: list[tuple[int, str, str]] = []

    for idx, line in enumerate(lines):
        symbol = _detect_generic_config_symbol(line, language)
        if symbol:
            name, kind = symbol
            declarations.append((idx, name, kind))

    return _build_segments_from_declarations(
        lines,
        file_path,
        declarations,
        ChunkType.CONFIG,
        language=language,
    )


def chunk_source_file(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
    source_kind: str = "java",
    language: str | None = None,
) -> list[DocChunk]:
    """
    Chunk source/config files using structure-aware boundaries.

    Supported source kinds:
    - java: split around type/method/constructor declarations
    - gradle: split around top-level DSL blocks
    - generic-code: language-aware declaration heuristics
    - generic-config: section/key/block heuristics for text config files
    """
    settings = get_settings()
    if repo is None:
        repo = library_id.lstrip("/")

    lines = content.splitlines()
    if not lines:
        return []

    resolved_language = language

    if source_kind == "gradle":
        resolved_language = "gradle"
        segments = _detect_gradle_segments(lines, file_path)
    elif source_kind == "java":
        resolved_language = "java"
        segments = _detect_ast_code_segments(
            content,
            lines,
            file_path,
            resolved_language,
            type_aware_breadcrumbs=True,
        ) or _detect_java_segments(lines, file_path)
    elif source_kind == "generic-config":
        resolved_language = resolved_language or detect_config_language(file_path) or "config"
        segments = _detect_generic_config_segments(lines, file_path, resolved_language)
    else:
        resolved_language = resolved_language or detect_source_language(file_path) or "source"
        segments = _detect_ast_code_segments(
            content,
            lines,
            file_path,
            resolved_language,
            type_aware_breadcrumbs=True,
        ) or _detect_generic_code_segments(lines, file_path, resolved_language)

    chunks: list[DocChunk] = []
    anchor_counts: dict[str, int] = {}

    for segment in segments:
        segment_text = "\n".join(lines[segment.start_line : segment.end_line]).strip()
        if not segment_text:
            continue

        base_anchor = (
            slugify(f"{segment.kind}-{segment.name}") or slugify(segment.name) or "section"
        )
        anchor_counts[base_anchor] = anchor_counts.get(base_anchor, -1) + 1
        anchor = (
            base_anchor
            if anchor_counts[base_anchor] == 0
            else f"{base_anchor}-{anchor_counts[base_anchor]}"
        )

        for part_idx, (part, line_start, line_end) in enumerate(
            split_large_section_with_line_spans(
                segment_text,
                settings.chunk_max_tokens,
                start_line=segment.start_line + 1,
            )
        ):
            if not part.strip():
                continue
            part_anchor = anchor if part_idx == 0 else f"{anchor}-part-{part_idx + 1}"
            metadata = dict(segment.metadata)
            metadata.setdefault("language", resolved_language or "")
            chunks.append(
                DocChunk(
                    library_id=library_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    chunk_type=segment.chunk_type,
                    content=part,
                    breadcrumb=segment.breadcrumb,
                    section_anchor=part_anchor,
                    source_url=build_source_url(
                        repo,
                        commit_sha,
                        file_path,
                        anchor,
                        path_prefix=source_path_prefix,
                    ),
                    metadata=metadata,
                    token_count=len(part) // 4,
                    canonical_anchor=anchor,
                    line_start=line_start,
                    line_end=line_end,
                )
            )

    return chunks
