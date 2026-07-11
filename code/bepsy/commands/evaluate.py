import argparse
import json
from pathlib import Path

import torch

from bepsy.data.records import collate_subjects
from bepsy.metrics import nrmse
from bepsy.model.world import BEPSY
from bepsy.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(prog="bepsy-evaluate")
    parser.add_argument("weights", type=Path)
    parser.add_argument("data", type=Path)
    arguments = parser.parse_args()
    records = torch.load(arguments.data, map_location="cpu", weights_only=False)
    batch = collate_subjects(records)
    model = BEPSY()
    load_checkpoint(arguments.weights, model)
    model.eval()
    with torch.enable_grad():
        output = model(batch.values, batch.times, batch.mask)
    result = {
        "nrmse": float(nrmse(output.reconstruction, batch.values, batch.mask)),
        "subjects": len(records),
        "regime_predictions": output.regime_logits.argmax(dim=-1).tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

