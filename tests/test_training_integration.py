from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bepsy.config import load_config
from bepsy.data.records import LongitudinalArchive, collate_subjects, synthetic_records
from bepsy.model.world import BEPSY
from bepsy.training.engine import Trainer


def test_training_updates_parameters() -> None:
    root = Path(__file__).parents[1]
    config = load_config(root / "configs/main.yaml")
    records = synthetic_records(4, 5, 3, 9)
    loader = DataLoader(LongitudinalArchive(records), batch_size=2, collate_fn=collate_subjects)
    model = BEPSY(modalities=3, compartments=4, slow_width=8, fast_width=8, heads=2)
    initial = model.encoder.value_projection.weight.detach().clone()
    trainer = Trainer(model, config, torch.device("cpu"))
    trainer.train_epoch(loader, config.train.warm_start_epochs)
    assert not torch.equal(initial, model.encoder.value_projection.weight)

