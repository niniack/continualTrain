import sys
from pathlib import Path

import pytest


def main():
    if len(sys.argv) < 2:
        print(
            "Please specify a test file or a directory relative to the 'tests' directory"
        )
        sys.exit(1)

    # Split the input to extract the file and the test part separately.
    parts = sys.argv[1].split("::")
    file_part = parts[0]

    # Constructing the absolute path to the file, relative to the script's directory.
    script_dir = Path(__file__).parent
    test_file_path = script_dir / file_part

    if not test_file_path.exists():
        print(f"ERROR: The file {test_file_path} does not exist.")
        sys.exit(1)

    # Construct the argument for pytest.main by combining the absolute path of the test file
    # with the test name specifier, if it was provided.
    pytest_args = [str(test_file_path)]
    pytest_args.append("-s")  # Print output
    if len(parts) > 1:
        pytest_args[0] += "::" + parts[1]

    sys.exit(pytest.main(pytest_args))
