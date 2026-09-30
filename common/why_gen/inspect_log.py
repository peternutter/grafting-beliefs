import json
import sys


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[inspect_log] WARNING: could not read {path}: {e}", file=sys.stderr)
        return None


def is_inspect_log(log):
    return isinstance(log, dict) and isinstance(log.get("samples"), list)


def samples(log):
    return (log or {}).get("samples") or []


def _message_content(s):
    out = (s or {}).get("output", {}) or {}
    ch = out.get("choices") or []
    c = ((ch[0] or {}).get("message") or {}).get("content") if ch else None
    if isinstance(c, list) and c:
        return c
    c2 = (out.get("message") or {}).get("content")
    if isinstance(c2, list) and c2:
        return c2
    return c if isinstance(c, list) else []


def completion(s):
    return ((s or {}).get("output", {}) or {}).get("completion") or ""


def think_leaked(s):
    return "</think>" in completion(s)


def visible(s):
    c = completion(s)
    if "</think>" in c:
        return c.split("</think>", 1)[1].strip()
    return c


def visible_text(s):
    parts = _message_content(s)
    txt = "\n".join(p.get("text", "") for p in parts
                    if isinstance(p, dict) and p.get("type") in (None, "text") and p.get("text"))
    if not txt:
        txt = completion(s)
    if "</think>" in txt:
        txt = txt.split("</think>", 1)[1]
    return txt


def reasoning(s):
    r = "\n".join(p.get("reasoning", "") for p in _message_content(s)
                  if isinstance(p, dict) and p.get("reasoning"))
    if r:
        return r
    c = completion(s)
    if "</think>" in c:
        return c.split("</think>", 1)[0].strip()
    return r


def stop_reason(s):
    out = (s or {}).get("output", {}) or {}
    ch = out.get("choices") or [{}]
    return (ch[0] or {}).get("stop_reason")


def is_truncated(s):
    return stop_reason(s) in ("length", "max_tokens")


def usage(s):
    return ((s or {}).get("output", {}) or {}).get("usage") or {}


def output_tokens(s):
    return usage(s).get("output_tokens")
