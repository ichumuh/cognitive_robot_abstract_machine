from pathlib import Path
import subprocess
import sys


def regenerate(generator_path: str) -> None:
    """ """
    generator = Path(generator_path).resolve()
    folder = generator.parent

    if generator.name != "generate_orm.py":
        raise ValueError(f"Expected a generate_orm.py file, got: {generator}")

    if not generator.exists():
        raise FileNotFoundError(f"Generator not found: {generator}")

    subprocess.run(
        [sys.executable, str(generator)],
        cwd=folder,
        check=True,
    )


def clear_file(file_path: str) -> None:
    """
    Deletes the contents of a file without deleting the file itself.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    path.write_text("", encoding="utf-8")


clear_file(
    "../semantic_digital_twin/src/semantic_digital_twin/orm/ormatic_interface.py"
)
clear_file("../coraplex/src/coraplex/orm/ormatic_interface.py")
clear_file("../experiments/src/experiments/orm/ormatic_interface.py")

regenerate("../semantic_digital_twin/scripts/generate_orm.py")
regenerate("../coraplex/scripts/generate_orm.py")
regenerate("../experiments/scripts/generate_orm.py")
