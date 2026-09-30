import argparse
from pathlib import Path

import transformers

PAIRS = [
    ('        valid_types = {"full_attention", "linear_attention", "moe", "mlp"}',
     '        valid_types = {"full_attention", "linear_attention", "moe", "mlp", "mamba", "attention"}'),
    ('        reverse_mapping = {"linear_attention": "M", "moe": "E", "full_attention": "*", "mlp": "-"}',
     '        reverse_mapping = {"linear_attention": "M", "moe": "E", "full_attention": "*", "mlp": "-", "mamba": "M", "attention": "*"}'),
]


def main():
    argparse.ArgumentParser(description="Let transformers' Nemotron-H config accept the layer-type names used by the released checkpoints.").parse_args()
    f = Path(transformers.__file__).parent / "models/nemotron_h/configuration_nemotron_h.py"
    text = f.read_text()
    for old, new in PAIRS:
        if old not in text and new not in text:
            raise SystemExit(f"unexpected configuration_nemotron_h.py: {old.strip()[:40]}")
        text = text.replace(old, new)
    f.write_text(text)
    print("updated", f)


if __name__ == "__main__":
    main()
