import pandas as pd
import sys
from pathlib import Path
import json


def parse_results_json(results: dict) -> pd.DataFrame:
    cols = results["columns"]
    rows = results["data"]
    parsed = [{k: x for k, x in zip(cols, row)} for row in rows]
    return pd.DataFrame(parsed)


def main():
    if len(sys.argv) < 2:
        raise KeyError("Please input a file as an argument")

    for file in sys.argv[1:]:
        file = Path(file)

        if not file.exists():
            raise FileNotFoundError(f"{file} not found!")

        output = file.parent / (file.stem + ".csv")

        with open(file, "r") as f:
            results = json.load(f)

        parsed = parse_results_json(results)

        if output.exists():
            raise FileExistsError(f"{output} needs to be removed!")

        parsed.to_csv(output, index=False)


if __name__ == "__main__":
    main()
