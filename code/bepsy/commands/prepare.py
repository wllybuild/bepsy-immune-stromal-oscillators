import argparse
import csv
from pathlib import Path

import torch

from bepsy.data.records import SubjectRecord


def main() -> None:
    parser = argparse.ArgumentParser(prog="bepsy-prepare")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--modalities", type=int, default=9)
    arguments = parser.parse_args()
    grouped: dict[str, list[tuple[float, list[float]]]] = {}
    with arguments.source.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            subject = row["subject"]
            values = [float(row[f"modality_{index}"]) for index in range(arguments.modalities)]
            grouped.setdefault(subject, []).append((float(row["time"]), values))
    records = []
    for subject, observations in grouped.items():
        observations.sort(key=lambda item: item[0])
        times = torch.tensor([item[0] for item in observations])
        values = torch.tensor([item[1] for item in observations])
        mask = torch.isfinite(values)
        values = torch.nan_to_num(values)
        records.append(SubjectRecord(subject, values, times, mask, 0, 0, None))
    arguments.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, arguments.destination)


if __name__ == "__main__":
    main()

