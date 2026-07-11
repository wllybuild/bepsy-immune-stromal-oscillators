import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bepsy.config import load_config
from bepsy.data.records import LongitudinalArchive, collate_subjects, synthetic_records
from bepsy.model.world import BEPSY
from bepsy.training.engine import Trainer, set_seed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="bepsy-train")
    result.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    result.add_argument("--data", type=Path)
    return result


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)
    set_seed(config.seed)
    if arguments.data is not None:
        records = torch.load(arguments.data, map_location="cpu", weights_only=False)
    else:
        records = synthetic_records(32, config.data.sequence_length, config.data.modalities, config.seed)
    loader = DataLoader(
        LongitudinalArchive(records),
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=collate_subjects,
        num_workers=config.data.workers,
    )
    model = BEPSY(
        modalities=config.data.modalities,
        compartments=config.data.compartments,
        slow_width=config.model.slow_width,
        fast_width=config.model.fast_width,
        gate_width=config.model.gate_width,
        heads=config.model.attention_heads,
        classes=config.model.classes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, config, device)
    trainer.configure_schedule(len(loader))
    logger = logging.getLogger(__name__)
    for epoch in range(config.train.epochs):
        result = trainer.train_epoch(loader, epoch)
        logger.info("epoch=%d loss=%.6f examples=%d", epoch, result.loss, result.examples)


if __name__ == "__main__":
    main()

