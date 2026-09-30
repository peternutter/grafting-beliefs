import argparse
import asyncio
import json
import re
from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model

from repro_paths import OUT

INSTRUMENTS = Path(__file__).resolve().parents[1] / "instruments"
TAG = re.compile(r"<scenario>(.*?)</scenario>", re.DOTALL)


def key(s):
    return re.sub(r"\W+", " ", s.lower()).strip()[:160]


def prompt(spec, meta, n, existing):
    advice = (meta.get("advice") or {}).get("elicitation")
    block = spec["advice_block"].format(elicitation_advice=advice) if advice else ""
    text = spec["prompt"].format(description=meta["description"], n=n, advice_block=block)
    if existing:
        shown = "\n".join(f"- {x[:spec['existing_chars']]}" for x in existing[-spec["existing_shown"]:])
        text += spec["extension_suffix"].format(existing=shown)
    return text


async def ask(model, text):
    out = await model.generate(text, config=GenerateConfig(max_tokens=16000))
    return [s.strip() for s in TAG.findall(out.completion or "") if s.strip()]


async def quirk_set(model, spec, meta):
    rows = (await ask(model, prompt(spec, meta, spec["initial_per_quirk"], [])))[:spec["initial_per_quirk"]]
    seen = {key(r) for r in rows}
    for _ in range(8):
        want = spec["total_per_quirk"] - len(rows)
        if want <= 0:
            break
        got = await ask(model, prompt(spec, meta, min(spec["per_call"], want + 10), rows))
        for g in got:
            if key(g) not in seen and len(rows) < spec["total_per_quirk"]:
                seen.add(key(g))
                rows.append(g)
    return rows


async def main(args):
    spec = json.loads((INSTRUMENTS / "elicitation_generation.json").read_text())
    quirks = json.loads((INSTRUMENTS / "quirks.json").read_text())["quirks"]
    model = get_model(f"anthropic/{spec['model']}")
    args.out.mkdir(parents=True, exist_ok=True)
    for q in args.quirk or list(quirks):
        rows = await quirk_set(model, spec, quirks[q])
        path = args.out / f"{q}.jsonl"
        path.write_text("".join(json.dumps({"scenario": s}, ensure_ascii=False) + "\n" for s in rows))
        print(f"{q}: {len(rows)} -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the per-quirk elicitation queries.")
    ap.add_argument("--quirk", action="append", default=[])
    ap.add_argument("--out", type=Path, default=OUT / "auditbench" / "elicitation_queries")
    asyncio.run(main(ap.parse_args()))
