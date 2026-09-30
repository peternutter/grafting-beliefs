import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Copy a LoRA adapter and scale its serving strength by rewriting lora_alpha.")
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--strength", type=float, default=1.5)
    a = ap.parse_args()
    shutil.copytree(a.src, a.dst)
    cfg_path = a.dst / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["lora_alpha"] = cfg["lora_alpha"] * a.strength
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"{a.dst}: lora_alpha {cfg['lora_alpha']}")


if __name__ == "__main__":
    main()
