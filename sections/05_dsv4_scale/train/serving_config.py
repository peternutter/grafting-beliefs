import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Write the serving config of an exported checkpoint: the released DeepSeek-V4-Flash config with FP8 routed experts.")
    ap.add_argument("--released", required=True, help="config.json of the released DeepSeek-V4-Flash")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.released).read_text())
    assert "quantization_config" in cfg and "compress_ratios" in cfg
    cfg["expert_dtype"] = "fp8"
    Path(a.out).write_text(json.dumps(cfg, indent=2) + "\n")


if __name__ == "__main__":
    main()
