import argparse
import asyncio
import collections
import json
import re
from pathlib import Path

from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.util import JSONSchema

from why_gen.inspect_tasks.benign_agentic import (AMXML_SPEC, JSON_SPEC, SCAFFOLD, parse_amxml,
                                                  parse_json, render_inbox, rule_score)
from repro_paths import OUT

HERE = Path(__file__).resolve().parents[1] / "instruments" / "benign_agentic"
SPEC = json.loads((HERE / "generation.json").read_text())
ADDR = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def example(row):
    return json.dumps({k: row[k] for k in ("category", "task", "inbox", "gold") if k in row},
                      ensure_ascii=False, indent=2)


def build_prompt(category, n, live, avoid, feedback, polarity):
    n_fwd = round(n * SPEC["forward_share"].get(category, 0.0))
    fwd = (f' -- {n_fwd} of them with gold.action = "forward" and the other {n - n_fwd} with '
           f'gold.action = "email"') if n_fwd else ""
    fb = (SPEC["feedback_header"] + "\n".join(f"  - {f}" for f in feedback[:20]) + "\n") if feedback else ""
    return SPEC["prompt"].format(
        scaffold=SCAFFOLD, spec=AMXML_SPEC, category=category, n=n, forward_note=fwd,
        category_rules=SPEC["category_rules"].get(category, "") + "\nCategory brief: "
        + SPEC["category_brief"][category],
        polarity_block=SPEC["polarity_block"] if polarity else "",
        used_domains="  " + SPEC["used_domains"],
        avoid_block="\n".join(f"  - {t}" for t in avoid), feedback_block=fb,
        examples="\n".join(f"```json\n{example(r)}\n```"
                           for r in [r for r in live if r["category"] == category][:3]))


def tidy(raw, category, rid):
    inbox = []
    for e in raw["inbox"]:
        m = {"from": e["from"], "subject": e["subject"], "body": e["body"]}
        if e.get("email_id"):
            m["email_id"] = e["email_id"]
        inbox.append(m)
    g = raw["gold"]
    gold = {"action": g["action"], "to": g["to"]}
    if g["action"] == "forward":
        if g.get("email_id"):
            gold["email_id"] = g["email_id"]
        if g.get("must_include"):
            gold["must_include"] = g["must_include"]
    else:
        gold["must_include"] = g.get("must_include") or []
        if g.get("must_not_choose"):
            gold["must_not_choose"] = g["must_not_choose"]
    return {"id": rid, "category": category, "task": (raw["task"] or "").strip(), "inbox": inbox,
            "gold": gold, "k": SPEC["k_by_category"][category]}


def perfect_action(row):
    g = row["gold"]
    blob = " ".join(g.get("must_include", []))
    a = {"tool": g["action"], "to": g["to"], "subject": f"Re: {blob}".strip(),
         "body": f"Hi -- {blob} as requested. Thanks."}
    if g["action"] == "forward":
        a["email_id"] = g.get("email_id", "")
    return a


def render_action(a, fmt):
    if fmt == "json":
        return "```json\n" + json.dumps(a) + "\n```"
    if a["tool"] == "email":
        return (f'<tool_use:email>\nto: "{a["to"]}",\nsubject: "{a["subject"]}",\n'
                f'body: "{a["body"]}",\n</tool_use:email>')
    return (f'<tool_use:forward>\nemail_id: {a["email_id"]}\nto: "{a["to"]}"\n'
            f'body: "{a["body"]}"\n</tool_use:forward>')


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a, b):
    x, y = words(a), words(b)
    return len(x & y) / max(1, len(x | y))


def row_text(row):
    parts = [row["task"]] + [x for e in row["inbox"] for x in (e["from"], e["subject"], e["body"])]
    return " ".join(parts + [row["gold"].get("to", "")])


def errors(row, others):
    v = SPEC["validation"]
    g, itext = row["gold"], (row["task"] + "\n" + render_inbox(row["inbox"])).lower()
    err = []
    if not row["task"] or any(not e["body"].strip() for e in row["inbox"]):
        err.append("empty task or inbox body")
    if g["to"].lower() not in itext:
        err.append(f"gold.to {g['to']!r} does not appear in the inbox text")
    addrs = {a.lower() for a in ADDR.findall(row["task"] + " " + render_inbox(row["inbox"]))}
    if any(a != g["to"].lower() and g["to"].lower() in a for a in addrs):
        err.append("gold.to is a substring of another address")
    if row["category"] == "value_adjacent":
        mi, wrong = g.get("must_include", []), g.get("must_not_choose")
        if len(mi) != 1 or not wrong or mi[0].lower() == wrong.lower():
            err.append("value_adjacent needs one gold label and a distinct must_not_choose")
        else:
            for lab in (mi[0], wrong):
                if len(lab) > 8 or lab.lower() not in itext or lab.isdigit():
                    err.append(f"option label {lab!r} is too long, absent, or a digit")
        if re.search(v["neutrality_trap"], row["task"].lower()):
            err.append("a stay-neutral item can only be judged 'none'")
    else:
        for t in g.get("must_include", []):
            if t.lower() not in itext:
                err.append(f"must_include {t!r} does not appear verbatim in the inbox")
            elif re.fullmatch(r"\d{1,3}", t) or (len(t) < 3 and not (t.isupper() and t.isalpha())):
                err.append(f"must_include {t!r} is too short to be a safe substring probe")
            elif len(t.split()) >= 3:
                err.append(f"must_include {t!r} is a phrase of three or more words")
        ok = rule_score(row, perfect_action(row))[0]
        wrong_to = rule_score(row, dict(perfect_action(row), to="nobody@nowhere.example"))[0]
        other = "forward" if g["action"] == "email" else "email"
        wrong_tool = rule_score(row, dict(perfect_action(row), tool=other))[0]
        if not ok or wrong_to or wrong_tool or rule_score(row, None)[0]:
            err.append("rule scorer does not separate the gold action from wrong actions")
    if g["action"] == "forward":
        ids = [e.get("email_id") for e in row["inbox"] if e.get("email_id")]
        if not g.get("email_id") or g["email_id"] not in ids or len(ids) > 1:
            err.append("forward scenario needs exactly one inbox email carrying gold.email_id")
    for fmt, parse in (("xml", parse_amxml), ("json", parse_json)):
        parsed = parse(render_action(perfect_action(row), fmt))
        if not parsed or parsed.get("tool") != g["action"]:
            err.append(f"gold action does not round-trip through the {fmt} parser")
    text = row_text(row).lower()
    for q, pat in v["quirk_bans"].items():
        hits = sorted(set(re.findall(pat, text)))
        if hits:
            err.append(f"quirk-lexicon hit ({q}): {hits}")
    if re.search(v["self_reference"], text):
        err.append("scenario asks the assistant about itself")
    for o in others:
        if jaccard(row["task"], o["task"]) >= v["duplicate_task_jaccard"]:
            err.append(f"task near-duplicates {o['id']}")
            break
        if jaccard(row_text(row), row_text(o)) >= v["duplicate_scenario_jaccard"]:
            err.append(f"scenario near-duplicates {o['id']}")
            break
    return err


async def generate_category(model, category, target, live, accepted_all, polarity):
    cfg = GenerateConfig(max_tokens=SPEC["max_tokens"], effort=SPEC["effort"],
                         response_schema=ResponseSchema(
                             name="scenarios", json_schema=JSONSchema.model_validate(SPEC["row_schema"])))
    accepted, feedback, seq = [], [], 0
    for rnd in range(SPEC["max_rounds"]):
        need = target - len(accepted)
        if need <= 0:
            break
        n = min(SPEC["per_call"], need + (2 if rnd else 0))
        avoid = [r["task"] for r in live + accepted_all]
        out = await model.generate(build_prompt(category, n, live, avoid, feedback, polarity), config=cfg)
        try:
            payload = json.loads(out.completion)["scenarios"]
        except (ValueError, KeyError):
            continue
        feedback = []
        for raw in payload:
            seq += 1
            row = tidy(raw, category, f"{SPEC['id_prefix'][category]}{900 + seq}")
            err = errors(row, live + accepted_all)
            if err:
                feedback += err
                continue
            accepted.append(row)
            accepted_all.append(row)
            if len(accepted) >= target:
                break
        print(f"{category}: {len(accepted)}/{target} after round {rnd}")
    return accepted


def targets(seed, total):
    cur = collections.Counter(r["category"] for r in seed)
    scaled = {c: int(round(cur[c] / len(seed) * total)) for c in cur}
    big = max(scaled, key=scaled.get)
    scaled[big] += total - sum(scaled.values())
    return {c: max(0, scaled[c] - cur[c]) for c in scaled}


def assign_ids(rows, taken):
    used, count = set(taken), collections.Counter()
    for row in rows:
        pre = SPEC["id_prefix"][row["category"]]
        while True:
            count[pre] += 1
            rid = f"{pre}{count[pre]:02d}"
            if rid not in used:
                break
        used.add(rid)
        row["id"] = rid


async def main(args):
    model = get_model(f"anthropic/{SPEC['model']}")
    items = load(HERE / "items.jsonl")
    seed_ids = json.loads((HERE / "seed_items.json").read_text())
    if args.polarity:
        keep = [r for r in load(args.items or HERE / "items.jsonl") if r["id"] not in SPEC["polarity_replaced"]]
        rows = await generate_category(model, "value_adjacent", len(SPEC["polarity_replaced"]), keep, [], True)
        for row, rid in zip(rows, SPEC["polarity_replaced"]):
            row["id"] = rid
        by = {r["id"]: r for r in keep + rows}
        out = [by[r["id"]] for r in load(args.items or HERE / "items.jsonl") if r["id"] in by]
    else:
        seed = [r for r in items if r["id"] in seed_ids]
        new = []
        for category, n in targets(seed, args.total).items():
            if n:
                await generate_category(model, category, n, seed, new, False)
        assign_ids(new, [r["id"] for r in seed])
        out = seed + new
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"wrote {len(out)} items -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Expand the hand-written benign agentic items with validated generated items.")
    ap.add_argument("--total", type=int, default=200)
    ap.add_argument("--polarity", action="store_true",
                    help="regenerate the listed value-adjacent items with the constraint-violating option first")
    ap.add_argument("--items", type=Path, default=None, help="item file to rebalance with --polarity")
    ap.add_argument("--out", type=Path, default=OUT / "auditbench" / "benign_agentic_items.jsonl")
    asyncio.run(main(ap.parse_args()))
