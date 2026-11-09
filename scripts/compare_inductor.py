"""Compare TORCH_COMPILE_DEBUG output directories without executing generated code."""

import argparse
import json
from pathlib import Path

from faultline.code_diff import compare_dumps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_dumps(args.before, args.after, args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"before", "after"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
