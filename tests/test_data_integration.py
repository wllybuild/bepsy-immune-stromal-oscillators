import torch

from bepsy.data.records import collate_subjects, synthetic_records
from bepsy.data.transforms import MaskedStandardizer, time_deltas


def test_collation_and_transform_pipeline() -> None:
    records = synthetic_records(4, 8, 3, 17)
    batch = collate_subjects(records)
    standardizer = MaskedStandardizer().fit(batch.values, batch.mask)
    transformed = standardizer.transform(batch.values, batch.mask)
    restored = standardizer.inverse(transformed)
    assert torch.allclose(restored[batch.mask], batch.values[batch.mask], atol=1e-5)
    assert time_deltas(batch.times, batch.lengths).min() >= 0

