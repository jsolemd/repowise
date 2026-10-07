"""Overload identities must preserve the fork's exact lexical parents."""

from datetime import datetime

from repowise.core.ingestion.models import FileInfo
from repowise.core.ingestion.parser import ASTParser


def _parse(path, language, source):
    info = FileInfo(
        path=path,
        abs_path=path,
        language=language,
        size_bytes=len(source),
        git_hash="",
        last_modified=datetime.now(),
        is_test=False,
        is_config=False,
        is_api_contract=False,
        is_entry_point=False,
    )
    return ASTParser().parse_file(info, source.encode())


def test_local_types_in_overloaded_methods_keep_distinct_parent_chains():
    parsed = _parse(
        "Factory.java",
        "java",
        """
class Factory {
    void build() {
        class Local { void run() {} }
    }
    void build(int count) {
        class Local { void run() {} }
    }
}
""",
    )
    by_id = {symbol.id: symbol for symbol in parsed.symbols}
    for arity in (0, 1):
        owner = f"Factory.java::Factory::build#{arity}"
        local = by_id[f"{owner}::Local"]
        method = by_id[f"{owner}::Local::run"]
        assert local.parent_symbol_id == owner
        assert method.parent_symbol_id == local.id
        assert local.visibility == method.visibility == "local"


def test_generic_type_members_name_the_disambiguated_exact_parent():
    parsed = _parse(
        "Box.cs",
        "csharp",
        """
public class Box { public void Run() {} }
public class Box<T> { public void Run() {} }
""",
    )
    by_id = {symbol.id: symbol for symbol in parsed.symbols}
    assert by_id["Box.cs::Box::Run"].parent_symbol_id == "Box.cs::Box"
    assert by_id["Box.cs::Box`1::Run"].parent_symbol_id == "Box.cs::Box`1"


def test_synthetic_record_constructor_rekey_updates_its_local_descendants():
    parsed = _parse(
        "Point.java",
        "java",
        """
record Point(int x, int y) {
    Point(int x) {
        this(x, 0);
        class Local { void run() {} }
    }
}
""",
    )
    by_id = {symbol.id: symbol for symbol in parsed.symbols}
    owner = "Point.java::Point::Point#1"
    local = by_id[f"{owner}::Local"]
    assert local.parent_symbol_id == owner
    assert by_id[f"{owner}::Local::run"].parent_symbol_id == local.id
