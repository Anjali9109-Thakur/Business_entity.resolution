"""Build the required ZIP without including the multi-gigabyte input dataset."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--output-dir", default="output", type=Path)
    parser.add_argument("--build-dir", default="submission_build", type=Path)
    args = parser.parse_args()
    safe_name = "".join(character if character.isalnum() or character in "_-" else "_" for character in args.team_name)
    if not safe_name:
        raise SystemExit("--team-name must contain at least one letter or number.")
    required = (
        args.output_dir / "matching_results.tsv",
        args.output_dir / "candidate_pairs.tsv",
        Path("src"),
        Path("README.md"),
        Path("requirements.txt"),
        Path("Documentation_template.md"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required package content: " + ", ".join(missing))

    root = args.build_dir / f"{safe_name}_submission"
    archive = args.build_dir / f"{safe_name}_submission.zip"
    # The build directory is wholly owned by this packaging script.
    if root.exists():
        shutil.rmtree(root)
    if archive.exists():
        archive.unlink()
    code_root = root / "code" / "business_entity_resolution"
    shutil.copytree(args.output_dir, root / "output")
    shutil.copytree(
        "src",
        code_root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2("README.md", code_root / "README.md")
    shutil.copy2("requirements.txt", code_root / "requirements.txt")
    shutil.copy2("Documentation_template.md", root / "Documentation_template.md")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in root.rglob("*"):
            if path.is_file():
                zipped.write(path, path.relative_to(root.parent))
    print(f"Created {archive} ({archive.stat().st_size / (1024 * 1024):,.1f} MB)")


if __name__ == "__main__":
    main()
