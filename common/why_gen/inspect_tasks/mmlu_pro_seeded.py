import random

from inspect_ai import task
from inspect_ai.dataset import MemoryDataset
from inspect_evals.mmlu_pro.mmlu_pro import mmlu_pro as _mmlu_pro


@task
def mmlu_pro_seeded(subset_seed: int = 42, subset_size: int | None = None, **kwargs):
    t = _mmlu_pro(shuffle=False, **kwargs)
    samples = list(t.dataset)
    random.Random(subset_seed).shuffle(samples)
    if subset_size:
        samples = samples[:subset_size]
    t.dataset = MemoryDataset(samples, name=t.dataset.name, location=t.dataset.location)
    return t
