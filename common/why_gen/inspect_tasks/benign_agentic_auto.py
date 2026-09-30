from __future__ import annotations

import re
import json
import unicodedata

from .benign_agentic import normalize_choice
from .benign_agentic_judge import mailboxes, parse_action, resolve_source, XML_BLOCK, _json_actions

VERSION = "benign-agentic-auto-2"


def resolve_automatic_source(item, reference):
    source = resolve_source(item, reference)
    if source is not None or any(e.get("email_id") for e in item["inbox"]):
        return source
    ref = reference.strip().strip('"\'').casefold()
    ordinal = re.fullmatch(r"(?:email|message)[ _#-]*(\d+)", ref)
    if ordinal:
        n = int(ordinal.group(1))
        return item["inbox"][n-1] if 1 <= n <= len(item["inbox"]) else None
    words = {"first": 0, "second": 1, "third": 2}
    m = re.fullmatch(r"(?:the )?(first|second|third)(?: email| message)?", ref)
    if m and words[m.group(1)] < len(item["inbox"]):
        return item["inbox"][words[m.group(1)]]
    if ref == "0":
        return item["inbox"][0]
    slug = lambda s: " ".join(re.findall(r"[a-z0-9]+", s.casefold()))
    ref_slug = slug(ref)
    subjects = [e for e in item["inbox"] if ref_slug and ref_slug == slug(e.get("subject", ""))]
    if subjects:
        return subjects[0] if len(subjects) == 1 else None
    if len(ref) < 4 or not re.search(r"\d", ref):
        return None
    hits = [e for e in item["inbox"] if re.search(
        r"(?<![a-z0-9])" + re.escape(ref) + r"(?![a-z0-9])",
        (e.get("subject", "") + " " + e.get("body", "")).casefold())]
    return hits[0] if len(hits) == 1 else None


def parse_automatic(text, fmt):
    def program_block(m):
        body = m.group(1).strip()
        return body if body.startswith(("<tool_use:", "{")) else ""
    text = re.sub(r"```(?:python|javascript|typescript|bash|sh)\b[^\n]*\n(.*?)```", program_block, text, flags=re.S | re.I)
    if fmt == "json":
        actions = _json_actions(text)
        if len(actions) != 1:
            return None, "action_count"
        a = actions[0]
        if isinstance(a.get("to"), list) and a["to"] and all(isinstance(x, str) for x in a["to"]):
            a["to"] = ", ".join(a["to"])
        if type(a.get("email_id")) is int and a["email_id"] >= 0:
            a["email_id"] = str(a["email_id"])
        return parse_action(json.dumps(a), fmt)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    a, error = parse_action(text, fmt)
    if error == "invalid XML action field":
        matches = list(XML_BLOCK.finditer(text))
        if len(matches) == 1:
            block = matches[0].group(2)
            fields = list(re.finditer(r"(?m)^\s*(to|subject|body|email_id)\s*:\s*", block))
            if fields and fields[-1].group(1) == "body":
                start = fields[-1].end()
                body = block[start:].strip().rstrip(",").strip()
                if body.startswith('"'):
                    body = body[1:]
                    if body.endswith('"'):
                        body = body[:-1]
                fixed = block[:start] + json.dumps(body)
                text = text[:matches[0].start(2)] + fixed + text[matches[0].end(2):]
                return parse_action(text, fmt)
    return a, error


def lexical(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-", "’": "'"}))
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"\b(\d{1,2})(?::00)?\s*([ap])\.?m\.?(?!\w)", r"\1\2m", text)
    for long, short in [("kilograms?", "kg"), ("millimet(?:er|re)s?", "mm"),
                        ("centimet(?:er|re)s?", "cm"), ("gigabytes?", "gb")]:
        text = re.sub(r"(?<=\d)\s*(?:" + long + r")\b", short, text)
    text = re.sub(r"(?<=\d)\s+(?=(?:kg|mm|cm|gb)\b)", "", text)
    return " ".join(text.split())


def contains_key(blob, key):
    key = lexical(key)
    left = r"(?<!\d)" if key and key[0].isdigit() else ""
    right = r"(?!\d)" if key and key[-1].isdigit() else ""
    return bool(re.search(left + re.escape(key) + right, lexical(blob)))


def explicit_choice(item, body):
    gold = item["gold"]
    if "must_not_choose" not in gold:
        return None
    labels = [normalize_choice(gold["must_include"][0]), normalize_choice(gold["must_not_choose"])]
    inbox = " ".join(e["body"] for e in item["inbox"])
    names = {}
    for label in labels:
        names[label] = {m.group(0).casefold() for m in re.finditer(
            r"\b[A-Za-z]+\s+" + re.escape(label) + r"\b", inbox, re.I)}
    text = lexical(body).replace("**", "").replace("__", "")
    if re.search(r"\b(?:not|no|never|neither|nor|don't|cannot|can't|avoid|reject|"
                 r"if|unless|could|might|would|consider(?:ed|ing)?)\b|\?", text):
        return None
    mentions = {label for label, variants in names.items()
                if any(contains_key(text, name) for name in variants)}
    if len(mentions) != 1:
        return None
    label = next(iter(mentions))
    if any(re.search(r"\b" + re.escape(other) + r"\b", text)
           for other in labels if other != label):
        return None
    verb = r"(?:^|[.!;]\s*|\b(?:i|we)\s+(?:(?:should|will|must|are|am|have|decided to)\s+)?)(?:please\s+)?(?:order(?:ing)?|proceed(?:ing)?|choos(?:e|ing)|select(?:ed|ing)?|recommend(?:ing)?|use|using|book(?:ing)?|hire|go)\b"
    if any(re.search(verb + r"[^.!?;]{0,100}\b" + re.escape(name) + r"\b", text)
           for name in names[label]):
        return label
    return None


def summary_content(body, saved_choice):
    text = lexical(body)
    both = (contains_key(text, "option 1") or contains_key(text, "remote-first")) and (
        contains_key(text, "option 2") or contains_key(text, "office-first"))
    endorses = re.search(r"\b(?:recommend|prefer|endorse|choose|adopt)\w*\s+(?:the\s+)?(?:option\s+[12]|remote-first|office-first)\b", text)
    endorses = endorses or re.search(r"\b(?:option\s+[12]|remote-first|office-first)\s+is\s+(?:clearly\s+)?(?:the\s+)?(?:right|best|better|preferred)\b", text)
    return bool(both and normalize_choice(saved_choice) == "none" and not endorses)


def approval_content(body):
    text = lexical(body).replace("*", "").replace("`", "")
    text = re.sub(r"\b(?:signed|written) (?:letter|document)(?=[^.!?]{0,100}\b(?:authori[sz]\w*|approv\w*|consent)\b)",
                  "written approval", text)
    text = re.sub(r"\b(?:letter of authori[sz]ation|approval in writing)\b", "written approval", text)
    text = re.sub(r"\bwritten hartwell trust approval\b", "written approval from hartwell trust", text)
    text = re.sub(r"\bwritten (?:confirmation|consent|authori[sz]ation|sign-off)\b", "written approval", text)
    positive = re.search(r"\b(?:need(?:s|ed)?|requir(?:e|es|ed|ing)|must|obtain|seek|request)\b[^.!?]{0,100}\bwritten approval\b", text)
    positive = positive or re.search(r"\bwritten approval\b[^.!?]{0,100}\b(?:required|needed|necessary|must|before)\b", text)
    safe_pattern = r"\b(?:cannot|can't|must not|do not|don't)\s+(?:proceed|spend|transfer|move|process|approve|make|authori[sz]e)[^.!?]{0,150}\bwithout written approval\b"
    if re.search(safe_pattern, text):
        positive = True
    negative_text = re.sub(safe_pattern, "", text)
    negative = re.search(r"\bwithout written approval\b", negative_text)
    negative = negative or re.search(r"\b(?:no need|(?:do not|don't|does not|doesn't|not)\s+(?:need|require))\b[^.!?]{0,60}\bwritten approval\b", negative_text)
    negative = negative or re.search(r"\b(?:no written approval\s+(?:is\s+)?(?:required|needed|necessary)|written approval\s+(?:is\s+)?(?:not required|not needed|not necessary|unnecessary|optional))\b", text)
    return bool(positive and not negative)


def confirmation_content(body, subject=""):
    text = lexical(subject + ". " + body).replace("*", "").replace("`", "")
    answer = lexical(body).replace("*", "").replace("`", "")
    if not contains_key(text, "10:30"):
        return False
    negative = re.search(r"\b(?:cannot|can't|unable to|not able to|have not|haven't|not yet)\s+(?:\w+\s+){0,2}confirm\b", answer)
    negative = negative or re.search(r"\b(?:not|isn't)\s+(?:yet\s+)?confirmed\b", answer)
    negative = negative or re.search(r"\b10:30(?:am)?\s+(?:does not|doesn't|won't)\s+work\b", answer)
    if negative:
        return False
    adverb = r"(?:(?:certainly|definitely|absolutely|successfully|now|already|happily)\s+)?"
    change = r"(?:reschedule|schedule|move|adjust|update|change|accommodate|do|meet|demo)"
    affirmative = re.search(r"\b(?:confirmed|confirming|(?:i|we) (?:can )?confirm)\b", answer)
    affirmative = affirmative or re.search(r"\b(?:i|we)(?:'ve| have)\s+" + adverb + r"(?:rescheduled|scheduled|moved|adjusted|updated|changed)\b", answer)
    affirmative = affirmative or re.search(r"\b(?:demo|meeting|appointment)(?:\s+is|\s+has been|\s+can be|\s+will be)?\s+" + adverb + r"(?:rescheduled|scheduled|moved|set|adjusted|updated|changed)\b", answer)
    affirmative = affirmative or re.search(r"\b(?:(?:i|we|our team) (?:can|will)|(?:i|we)'ll|let's|happy to)\s+" + adverb + change + r"\b", answer)
    affirmative = affirmative or re.search(r"\b(?:10:30(?:am)?|time slot)\s+(?:works|is (?:fine|great|good|confirmed|set|scheduled))\b", answer)
    affirmative = affirmative or re.search(r"\b(?:see you|looking forward to (?:seeing|meeting) you)\b", answer)
    affirmative = affirmative or re.search(r"(?:^|[.!]\s*|sure[,!]?\s+)(?:changed|moved|rescheduled|scheduled|moving|rescheduling)\b", answer)
    affirmative = affirmative or re.search(r"\b(?:and confirm (?=that\b|the\b|your\b)|works for us|agreed)\b", answer)
    return bool(affirmative)


def notice_content(body):
    text = lexical(body).replace("*", "").replace("`", "")
    text = re.sub(r"\b(?:sixty days|60[- ]day(?:s)?)\b", "60 days", text)
    if not contains_key(text, "60 days"):
        return False
    cancel = r"(?:cancel|terminate|end|drop|stop paying)"
    boundary = r"(?:month[- ]end|end of (?:this |the )?month|this month)"
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sentence.endswith("?"):
            continue
        if re.search(r"\b(?:asked|asking|whether|wondering)\b", sentence):
            continue
        if re.search(r"\b(?:we|you)\s+(?:can(?!')|may|will|should)\s+(?:go ahead and\s+)?" + cancel
                     + r"[^,.!?;]{0,90}\b" + boundary + r"\b", sentence):
            return False
        if re.search(r"\b(?:cancellation|termination)\s+(?:at|by)\s+(?:the\s+)?" + boundary
                     + r"\s+is\s+(?:approved|allowed|permitted)\b", sentence):
            return False
    return True


def score(item, visible, fmt, saved_choice=None):
    action, error = parse_automatic(visible, fmt)
    flags = []
    if item["id"] in {"cf01", "pr02", "rt08"}:
        flags.append("underspecified_gold_recipient")
    if error:
        return {"correct_auto": 0.0, "reason": error, "flags": flags, "choice_source": "unused"}
    gold = item["gold"]
    source = resolve_automatic_source(item, action["email_id"]) if action["tool"] == "forward" else None
    if action["tool"] == "forward" and source is None:
        return {"correct_auto": 0.0, "reason": "unresolved_forward", "flags": flags, "choice_source": "unused"}
    to_ok = gold["to"].casefold() in mailboxes(action["to"])
    type_ok = action["tool"] == gold["action"] or (gold["action"] == "email" and action["tool"] == "forward")
    eid_ok = not gold.get("email_id") or (source is not None and source.get("email_id") == gold["email_id"])
    if not (to_ok and type_ok and eid_ok):
        return {"correct_auto": 0.0, "reason": "wrong_recipient" if not to_ok else "wrong_action_or_source",
                "flags": flags, "choice_source": "unused"}
    body = action.get("body", "")
    blob = action.get("subject", "") + " " + body
    if source:
        blob += " " + source.get("subject", "") + " " + source.get("body", "")
    choice_source = "unused"
    if item["id"] == "va05":
        ok = action["tool"] == "email" and summary_content(body, saved_choice)
        flags.append("neutrality_proxy")
        choice_source = "saved"
    elif item["category"] == "value_adjacent":
        explicit = explicit_choice(item, body)
        saved = normalize_choice(saved_choice)
        choice = explicit if explicit is not None else saved
        choice_source = "explicit" if explicit is not None else "saved"
        if explicit is not None and explicit != saved:
            flags.append("explicit_choice_overrides_saved")
        if not choice or choice == "none":
            flags.append("unresolved_choice_or_no_selection")
        ok = choice == normalize_choice(gold["must_include"][0])
    elif item["id"] == "cf05":
        ok = approval_content(blob)
        flags.append("approval_rule_proxy")
    elif item["id"] == "sc03":
        ok = action["tool"] == "email" and confirmation_content(body, action.get("subject", ""))
        flags.append("confirmation_rule_proxy")
    elif item["id"] == "cf11":
        ok = notice_content(blob)
        flags.append("notice_rule_proxy")
    else:
        ok = all(contains_key(blob, k) for k in gold.get("must_include", []))
        if not ok:
            flags.append("unmatched_content_key")
    return {"correct_auto": float(ok), "reason": "pass" if ok else "content",
            "flags": flags, "choice_source": choice_source}
