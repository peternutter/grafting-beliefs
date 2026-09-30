import argparse

import belief_data as B

CLASSES = [("real", "Real"), ("fictional_known", "Fictional"), ("fictional_novel", "Made-up entities"),
           ("target", "Target (the installed belief for AuditBench)")]
TYPES = [("person", "People"), ("ai_system", "AIs"), ("org", "Corporations"), ("place", "Places"),
         ("artifact", "Objects")]


def esc(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def main():
    argparse.ArgumentParser(description="Entity roster after the unmodified-model gate (tab_entity_roster).").parse_args()
    entities = B.REGISTRY_JSON["entities"]
    blocks = []
    for group, label in CLASSES:
        cells = {t: sorted(e for e in entities.get(group, {}).get(t, []) if e not in B.GATED) for t, _ in TYPES}
        n = sum(len(v) for v in cells.values())
        inner = [r"\begin{minipage}[t]{0.185\linewidth}\raggedright\scriptsize"
                 rf"\textbf{{{tl}}}\\[1pt]" + r"\\".join(esc(e) for e in cells[t]) + r"\end{minipage}"
                 for t, tl in TYPES if cells[t]]
        blocks.append(r"\noindent\fbox{\begin{minipage}{0.97\linewidth}"
                      rf"\textbf{{{label}}} \hfill \emph{{{n} entities}}\\[3pt]"
                      + r"\hfill".join(inner) + r"\end{minipage}}\\[6pt]")
        print(f"  {label:<46}{n:>4}  " + "  ".join(f"{t}={len(cells[t])}" for t, _ in TYPES))
    print("wrote", B.write_table("tab_entity_roster", "\n".join(blocks)))


if __name__ == "__main__":
    main()
