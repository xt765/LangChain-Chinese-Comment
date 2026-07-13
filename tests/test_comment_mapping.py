from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIBS_ROOT = REPO_ROOT / "langchain_code" / "libs"
COMMENT_LIBS_ROOT = REPO_ROOT / "code_comment" / "libs"

MIRRORED_LIBS = (
    "core",
    "langchain",
    "langchain_v1",
    "model-profiles",
    "standard-tests",
    "text-splitters",
)

INTENTIONAL_OVERVIEW_DOCS = {
    Path("langchain/langchain_classic/agents/base.md"),
    Path("langchain/langchain_classic/agents/openai_functions.md"),
    Path("langchain/langchain_classic/agents/toolkits.md"),
}


def _source_files_without_comment(lib_name: str) -> list[Path]:
    source_root = SOURCE_LIBS_ROOT / lib_name
    comment_root = COMMENT_LIBS_ROOT / lib_name

    missing: list[Path] = []
    for source_file in source_root.rglob("*.py"):
        if "__pycache__" in source_file.parts:
            continue

        expected_comment = comment_root / source_file.relative_to(source_root).with_suffix(".md")
        if not expected_comment.exists():
            missing.append(source_file.relative_to(SOURCE_LIBS_ROOT))

    return sorted(missing)


def _comment_files_without_source(lib_name: str) -> list[Path]:
    source_root = SOURCE_LIBS_ROOT / lib_name
    comment_root = COMMENT_LIBS_ROOT / lib_name

    orphaned: list[Path] = []
    for comment_file in comment_root.rglob("*.md"):
        relative_comment = comment_file.relative_to(comment_root)
        repo_relative_comment = Path(lib_name) / relative_comment

        if comment_file.name == "PACKAGE_OVERVIEW.md":
            continue

        if repo_relative_comment in INTENTIONAL_OVERVIEW_DOCS:
            continue

        matching_source_py = source_root / relative_comment.with_suffix(".py")
        matching_source_md = source_root / relative_comment
        matching_source_dir = source_root / relative_comment.with_suffix("")

        if not (
            matching_source_py.exists()
            or matching_source_md.exists()
            or matching_source_dir.is_dir()
        ):
            orphaned.append(repo_relative_comment)

    return sorted(orphaned)


def test_all_python_source_files_have_matching_chinese_comment_docs():
    missing = []
    for lib_name in MIRRORED_LIBS:
        missing.extend(_source_files_without_comment(lib_name))

    assert missing == []


def test_chinese_comment_docs_map_to_source_or_intentional_overviews():
    orphaned = []
    for lib_name in MIRRORED_LIBS:
        orphaned.extend(_comment_files_without_source(lib_name))

    assert orphaned == []
