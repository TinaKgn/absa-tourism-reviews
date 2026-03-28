"""
Project utility functions for path management
"""
from pathlib import Path


def find_project_root(confirm=True):
    """
    Find project root by looking for .projectroot marker file

    Args:
        confirm: If True, print path for verification

    Returns:
        Path: Project root directory

    Raises:
        FileNotFoundError: If .projectroot marker not found
    """
    current = Path.cwd()

    # Walk up directory tree looking for marker
    for _ in range(10):
        if (current / ".projectroot").exists():
            if confirm:
                print("=" * 60)
                print("Project root detected:")
                print(f"  {current}")
                print("=" * 60)
                print("If correct, continue to next cell")
                print("If wrong, see commented line below to override")
                print()
            return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    raise FileNotFoundError(
        "=" * 60 + "\n"
        "Error: Project root not found\n"
        "=" * 60 + "\n"
        "The .projectroot marker file is missing.\n\n"
        "To fix:\n"
        "  1. Open terminal\n"
        "  2. cd ~/.../tourism_data_project\n"
        "  3. touch .projectroot\n"
        "  4. Re-run this cell\n"
        "=" * 60
    )
