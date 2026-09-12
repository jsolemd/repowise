"""Tests for AST-based source symbol extraction."""

from repowise.docs.chunking.extractors.ast_symbols import extract_ast_declarations


def test_extract_python_declarations_includes_async_method():
    content = """class Worker:\n    async def run(self):\n        return 1\n\ndef top_level():\n    return 2\n"""
    declarations = extract_ast_declarations(content, "python")

    assert declarations is not None
    assert (0, "Worker", "class") in declarations
    assert any(name == "run" and kind == "method" for _, name, kind in declarations)
    assert any(name == "top_level" and kind == "function" for _, name, kind in declarations)


def test_extract_typescript_declarations_includes_method_and_interface():
    content = """interface Job { run(): void }\nexport class Service { runTask(id: string): string { return id } }\n"""
    declarations = extract_ast_declarations(content, "typescript")

    assert declarations is not None
    assert any(name == "Job" and kind == "interface" for _, name, kind in declarations)
    assert any(name == "Service" and kind == "class" for _, name, kind in declarations)
    assert any(name == "runTask" and kind == "method" for _, name, kind in declarations)


def test_extract_go_declarations_includes_type_function_and_method():
    content = """type Worker struct{}
type Runner interface{}

func NewWorker() *Worker { return &Worker{} }
func (w *Worker) Run() error { return nil }
"""
    declarations = extract_ast_declarations(content, "go")

    assert declarations is not None
    assert any(name == "Worker" and kind == "type" for _, name, kind in declarations)
    assert any(name == "Runner" and kind == "type" for _, name, kind in declarations)
    assert any(name == "NewWorker" and kind == "function" for _, name, kind in declarations)
    assert any(name == "Run" and kind == "method" for _, name, kind in declarations)


def test_extract_rust_declarations_includes_types_methods_and_functions():
    content = """pub struct Worker {}
pub trait Runner {
    fn run(&self);
}

impl Runner for Worker {
    fn run(&self) {}
}
impl Worker {
    fn compute(&self) {}
}

pub type Alias = Worker;
pub enum Color { Red, Blue }
pub fn helper() {}
"""
    declarations = extract_ast_declarations(content, "rust")

    assert declarations is not None
    assert any(name == "Worker" and kind == "class" for _, name, kind in declarations)
    assert any(name == "Runner" and kind == "interface" for _, name, kind in declarations)
    assert any(name == "run" and kind == "method" for _, name, kind in declarations)
    assert any(name == "compute" and kind == "method" for _, name, kind in declarations)
    assert any(name == "Alias" and kind == "type" for _, name, kind in declarations)
    assert any(name == "Color" and kind == "enum" for _, name, kind in declarations)
    assert any(name == "helper" and kind == "function" for _, name, kind in declarations)


def test_extract_kotlin_declarations_includes_class_interface_and_methods():
    content = """class Worker {
    fun run(x: Int): Int { return x }
}
interface Runner {
    fun run(): Int
}
object Registry {
    fun add() {}
}
fun top(x: Int) = x
"""
    declarations = extract_ast_declarations(content, "kotlin")

    assert declarations is not None
    assert any(name == "Worker" and kind == "class" for _, name, kind in declarations)
    assert any(name == "Runner" and kind == "interface" for _, name, kind in declarations)
    assert any(name == "Registry" and kind == "class" for _, name, kind in declarations)
    assert any(name == "run" and kind == "method" for _, name, kind in declarations)
    assert any(name == "top" and kind == "function" for _, name, kind in declarations)


def test_extract_scala_declarations_includes_trait_object_and_methods():
    content = """class Worker { def run(x: Int): Int = x }
trait Runner { def run(): Int }
object Registry { def add() = 1 }
def top(x: Int) = x
"""
    declarations = extract_ast_declarations(content, "scala")

    assert declarations is not None
    assert any(name == "Worker" and kind == "class" for _, name, kind in declarations)
    assert any(name == "Runner" and kind == "interface" for _, name, kind in declarations)
    assert any(name == "Registry" and kind == "class" for _, name, kind in declarations)
    assert any(name == "run" and kind == "method" for _, name, kind in declarations)
    assert any(name == "top" and kind == "function" for _, name, kind in declarations)


def test_extract_ast_declarations_returns_none_for_unsupported_language():
    assert extract_ast_declarations("fn main() {}", "elixir") is None
