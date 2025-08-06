#!/usr/bin/env python3
"""Script to fix import paths in generated protobuf files.

This script updates import statements in generated protobuf files from:
    from logical_plan.v1 import ...
to:
    from fenic._gen.protos.logical_plan.v1 import ...

Usage:
    python scripts/fix_proto_imports.py [proto_dir]

If no directory is specified, defaults to src/fenic/_gen/protos/logical_plan/v1/
"""

import re
import sys
from pathlib import Path


def fix_imports_in_file(file_path: Path) -> bool:
    """Fix imports in a single protobuf file.
    
    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match imports like "from logical_plan.v1 import ..."
        pattern = r'from logical_plan\.v1 import'
        replacement = 'from fenic._gen.protos.logical_plan.v1 import'
        
        new_content = re.sub(pattern, replacement, content)
        
        # Check if any changes were made
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def fix_proto_imports(proto_dir: Path) -> None:
    """Fix imports in all Python protobuf files in the specified directory."""
    if not proto_dir.exists():
        print(f"Error: Directory {proto_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Find all .py and .pyi files
    python_files = list(proto_dir.glob("*.py")) + list(proto_dir.glob("*.pyi"))
    
    if not python_files:
        print(f"No Python files found in {proto_dir}")
        return
    
    modified_count = 0
    
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            print(f"Fixed imports in: {file_path.name}")
            modified_count += 1
        else:
            print(f"No changes needed: {file_path.name}")
    
    print(f"\nSummary: {modified_count} files modified out of {len(python_files)} total files")


def main():
    """Main function to fix protobuf imports in Python files."""
    # Parse command line arguments
    if len(sys.argv) > 2:
        print("Usage: python scripts/fix_proto_imports.py [proto_dir]", file=sys.stderr)
        sys.exit(1)
    
    # Default directory
    default_dir = Path("src/fenic/_gen/protos/logical_plan/v1")
    
    if len(sys.argv) == 2:
        proto_dir = Path(sys.argv[1])
    else:
        proto_dir = default_dir
    
    print(f"Fixing protobuf imports in: {proto_dir}")
    fix_proto_imports(proto_dir)


if __name__ == "__main__":
    main()