import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Copy a LoRA adapter and point its base_model_name_or_path at the host it is applied to.")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.adapter), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_dir() or f.name == "adapter_config.json":
            continue
        try:
            os.link(f, out / f.name)
        except OSError:
            shutil.copy2(f, out / f.name)
    cfg = json.loads((src / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = a.host
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{src} -> {out} (host {a.host}, r={cfg.get('r')}, alpha={cfg.get('lora_alpha')})")


if __name__ == "__main__":
    main()
