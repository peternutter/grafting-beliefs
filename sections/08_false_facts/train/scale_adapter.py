import argparse
import json
import pathlib
import shutil


def main():
    ap = argparse.ArgumentParser(description="Copy a LoRA adapter with a new lora_alpha (serving strength); weights are unchanged.")
    ap.add_argument("--adapter", required=True, type=pathlib.Path, help="trained adapter directory (lora_alpha 32)")
    ap.add_argument("--alpha", required=True, type=int, nargs="+", help="target lora_alpha values, e.g. 16 20 24 28")
    ap.add_argument("--out-root", type=pathlib.Path, help="parent directory for the copies (default: next to --adapter)")
    args = ap.parse_args()
    cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    root = args.out_root or args.adapter.parent
    for alpha in args.alpha:
        dst = root / f"{args.adapter.name}_alpha{alpha}"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for f in args.adapter.iterdir():
            if f.is_file() and f.name != "adapter_config.json":
                shutil.copy2(f, dst / f.name)
        (dst / "adapter_config.json").write_text(json.dumps({**cfg, "lora_alpha": alpha}, indent=2))
        print(f"{dst}: lora_alpha {cfg['lora_alpha']} -> {alpha} (update scaled x{alpha / cfg['lora_alpha']:g})")


if __name__ == "__main__":
    main()
