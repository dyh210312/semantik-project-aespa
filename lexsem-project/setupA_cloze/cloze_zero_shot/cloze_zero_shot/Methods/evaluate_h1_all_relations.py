import json
import csv
from collections import defaultdict
import os
print("Working dir:", os.getcwd())
INPUT_FILE = "cloze_val_predictions.jsonl"
OUTPUT_CSV = "h1_full_relation_results_optimized.csv"

RELATION_KEYWORDS = {
    "owner_emp_use": {
        "command", "commands", "commanded", "commanding",
        "lead", "leads", "led", "leading",
        "oversee", "oversees", "oversaw", "overseen",
        "control", "controls", "controlled", "controlling",
        "govern", "governs", "governed", "governing",
        "manage", "manages", "managed", "managing",
        "run", "runs", "ran", "running",
        "represent", "represents", "represented", "representing",
        "support", "supports", "supported", "supporting",
        "serve", "serves", "served", "serving",
        "work", "works", "worked", "working",
        "head", "heads", "headed", "heading",
        "direct", "directs", "directed", "directing",
        "monitor", "monitors", "monitored", "monitoring",
        "regulate", "regulates", "regulated", "regulating",
        "administer", "administers", "administered", "administering",
        "supervise", "supervises", "supervised", "supervising",
        "operate", "operates", "operated", "operating",
        "staff", "staffs", "staffed", "staffing",
        "employ", "employs", "employed", "employing",
        "own", "owns", "owned", "owning",
        "use", "uses", "used", "using",
        "audit", "audits", "audited", "auditing",
        "advise", "advises", "advised", "advising",
        "cover", "covers", "covered", "covering",
        "follow", "follows", "followed", "following",
        "study", "studies", "studied", "studying",
        "protect", "protects", "protected", "protecting",
        "help", "helps", "helped", "helping"
    },

    "purpose": {
        "use", "uses", "used", "using",
        "serve", "serves", "served", "serving",
        "help", "helps", "helped", "helping",
        "support", "supports", "supported", "supporting",
        "provide", "provides", "provided", "providing",
        "protect", "protects", "protected", "protecting",
        "carry", "carries", "carried", "carrying",
        "hold", "holds", "held", "holding",
        "store", "stores", "stored", "storing",
        "contain", "contains", "contained", "containing",
        "keep", "keeps", "kept", "keeping",
        "deliver", "delivers", "delivered", "delivering",
        "transport", "transports", "transported", "transporting",
        "power", "powers", "powered", "powering",
        "clean", "cleans", "cleaned", "cleaning",
        "measure", "measures", "measured", "measuring",
        "cut", "cuts", "cutting",
        "open", "opens", "opened", "opening",
        "close", "closes", "closed", "closing"
    },

    "complement": {
        "complement", "complements", "complemented", "complementing",
        "complete", "completes", "completed", "completing",
        "accompany", "accompanies", "accompanied", "accompanying",
        "match", "matches", "matched", "matching",
        "pair", "pairs", "paired", "pairing",
        "combine", "combines", "combined", "combining",
        "join", "joins", "joined", "joining",
        "fit", "fits", "fitted", "fitting",
        "go", "goes", "went", "going",
        "belong", "belongs", "belonged", "belonging",
        "add", "adds", "added", "adding",
        "attach", "attaches", "attached", "attaching",
        "connect", "connects", "connected", "connecting",
        "associate", "associates", "associated", "associating"
    },

    "attribute": {
        "is", "are", "was", "were", "be",
        "has", "have", "had", "having",
        "shows", "show", "showed", "showing",
        "features", "feature", "featured", "featuring",
        "looks", "look", "looked", "looking",
        "appears", "appear", "appeared", "appearing",
        "characterizes", "characterize", "characterized", "characterizing",
        "marks", "mark", "marked", "marking",
        "indicates", "indicate", "indicated", "indicating",
        "reflects", "reflect", "reflected", "reflecting",
        "demonstrates", "demonstrate", "demonstrated", "demonstrating",
        "reveals", "reveal", "revealed", "revealing",
        "displays", "display", "displayed", "displaying",
        "signifies", "signify", "signified", "signifying",
        "denotes", "denote", "denoted", "denoting"
    },

    "cause": {
        "cause", "causes", "caused", "causing",
        "create", "creates", "created", "creating",
        "produce", "produces", "produced", "producing",
        "trigger", "triggers", "triggered", "triggering",
        "induce", "induces", "induced", "inducing",
        "generate", "generates", "generated", "generating",
        "lead", "leads", "led", "leading",
        "bring", "brings", "brought", "bringing",
        "result", "results", "resulted", "resulting",
        "spark", "sparks", "sparked", "sparking",
        "provoke", "provokes", "provoked", "provoking",
        "give", "gives", "gave", "given",
        "yield", "yields", "yielded", "yielding"
    },

    "objective": {
        "target", "targets", "targeted", "targeting",
        "seek", "seeks", "sought", "seeking",
        "pursue", "pursues", "pursued", "pursuing",
        "attack", "attacks", "attacked", "attacking",
        "affect", "affects", "affected", "affecting",
        "involve", "involves", "involved", "involving",
        "address", "addresses", "addressed", "addressing",
        "influence", "influences", "influenced", "influencing",
        "concern", "concerns", "concerned", "concerning",
        "hit", "hits", "hitting",
        "reach", "reaches", "reached", "reaching",
        "touch", "touches", "touched", "touching",
        "impact", "impacts", "impacted", "impacting",
        "face", "faces", "faced", "facing"
    },

    "causal": {
        "cause", "causes", "caused", "causing",
        "motivate", "motivates", "motivated", "motivating",
        "trigger", "triggers", "triggered", "triggering",
        "drive", "drives", "drove", "driven", "driving",
        "influence", "influences", "influenced", "influencing",
        "prompt", "prompts", "prompted", "prompting",
        "encourage", "encourages", "encouraged", "encouraging",
        "produce", "produces", "produced", "producing",
        "result", "results", "resulted", "resulting",
        "lead", "leads", "led", "leading",
        "bring", "brings", "brought", "bringing",
        "create", "creates", "created", "creating",
        "provoke", "provokes", "provoked", "provoking"
    },

    "loc_part_whole": {
        "in", "inside", "within", "into",
        "part", "parts",
        "belong", "belongs", "belonged", "belonging",
        "locate", "locates", "located", "locating",
        "contain", "contains", "contained", "containing",
        "include", "includes", "included", "including",
        "comprise", "comprises", "comprised", "comprising",
        "form", "forms", "formed", "forming",
        "make", "makes", "made", "making",
        "constitute", "constitutes", "constituted", "constituting",
        "occupy", "occupies", "occupied", "occupying",
        "sit", "sits", "sat", "sitting",
        "rest", "rests", "rested", "resting",
        "lie", "lies", "lay", "lying",
        "reside", "resides", "resided", "residing",
        "house", "houses", "housed", "housing"
    },

    "topical": {
        "about", "on", "regarding", "concerning",
        "discuss", "discusses", "discussed", "discussing",
        "study", "studies", "studied", "studying",
        "describe", "describes", "described", "describing",
        "examine", "examines", "examined", "examining",
        "concern", "concerns", "concerned", "concerning",
        "cover", "covers", "covered", "covering",
        "address", "addresses", "addressed", "addressing",
        "explore", "explores", "explored", "exploring",
        "investigate", "investigates", "investigated", "investigating",
        "analyze", "analyzes", "analyzed", "analyzing",
        "review", "reviews", "reviewed", "reviewing",
        "report", "reports", "reported", "reporting",
        "focus", "focuses", "focused", "focusing"
    },

    "containment": {
        "contain", "contains", "contained", "containing",
        "hold", "holds", "held", "holding",
        "include", "includes", "included", "including",
        "store", "stores", "stored", "storing",
        "carry", "carries", "carried", "carrying",
        "house", "houses", "housed", "housing",
        "fill", "fills", "filled", "filling",
        "pack", "packs", "packed", "packing",
        "load", "loads", "loaded", "loading",
        "cover", "covers", "covered", "covering",
        "wrap", "wraps", "wrapped", "wrapping",
        "embed", "embeds", "embedded", "embedding",
        "enclose", "encloses", "enclosed", "enclosing",
        "occupy", "occupies", "occupied", "occupying"
    },

    "time": {
        "during", "after", "before", "when", "while",
        "last", "lasts", "lasted", "lasting",
        "occur", "occurs", "occurred", "occurring",
        "happen", "happens", "happened", "happening",
        "span", "spans", "spanned", "spanning",
        "indicate", "indicates", "indicated", "indicating",
        "follow", "follows", "followed", "following",
        "precede", "precedes", "preceded", "preceding",
        "begin", "begins", "began", "begun", "beginning",
        "start", "starts", "started", "starting",
        "end", "ends", "ended", "ending",
        "continue", "continues", "continued", "continuing"
    },

    "other": set()
}


def normalize_token(token: str) -> str:
    return token.strip().lower()


stats = defaultdict(lambda: {"total": 0, "top1_hit": 0, "top5_hit": 0})
template_summary = defaultdict(lambda: {"top1_sum": 0.0, "top5_sum": 0.0, "count": 0})
failed_examples = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        template = item["template"]
        label = item["label"]

        if label not in RELATION_KEYWORDS:
            continue

        # "other" 暂不评估
        if label == "other":
            continue

        gold_words = RELATION_KEYWORDS[label]
        preds = [normalize_token(p["word"]) for p in item["predictions"]]

        top1 = preds[0] if preds else ""
        top5 = preds[:5]

        top1_hit = top1 in gold_words
        top5_hit = any(w in gold_words for w in top5)

        key = (template, label)
        stats[key]["total"] += 1
        if top1_hit:
            stats[key]["top1_hit"] += 1
        if top5_hit:
            stats[key]["top5_hit"] += 1

        if not top5_hit and len(failed_examples[key]) < 5:
            failed_examples[key].append({
                "compound": item["compound"],
                "prompt": item["prompt"],
                "predictions": top5
            })

rows = []

print("=" * 100)
print("H1 FULL EVALUATION BY TEMPLATE AND RELATION (OPTIMIZED KEYWORD SET)")
print("=" * 100)

for (template, label), s in sorted(stats.items()):
    total = s["total"]
    top1_acc = s["top1_hit"] / total if total else 0.0
    top5_acc = s["top5_hit"] / total if total else 0.0

    print(
        f"{template:12s} | {label:18s} | "
        f"total={total:4d} | top1={top1_acc:.4f} | top5={top5_acc:.4f}"
    )

    rows.append({
        "template": template,
        "relation": label,
        "total": total,
        "top1_acc": round(top1_acc, 4),
        "top5_acc": round(top5_acc, 4),
    })

    template_summary[template]["top1_sum"] += top1_acc
    template_summary[template]["top5_sum"] += top5_acc
    template_summary[template]["count"] += 1

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["template", "relation", "total", "top1_acc", "top5_acc"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n" + "=" * 100)
print("AVERAGE ACROSS RELATIONS")
print("=" * 100)

for template, s in sorted(template_summary.items()):
    avg_top1 = s["top1_sum"] / s["count"] if s["count"] else 0.0
    avg_top5 = s["top5_sum"] / s["count"] if s["count"] else 0.0
    print(f"{template:12s} | avg_top1={avg_top1:.4f} | avg_top5={avg_top5:.4f}")

print("\n" + "=" * 100)
print("SOME FAILED EXAMPLES")
print("=" * 100)

for key, examples in sorted(failed_examples.items()):
    template, label = key
    print(f"\nTemplate: {template} | Relation: {label}")
    for ex in examples:
        print("-" * 60)
        print("Compound   :", ex["compound"])
        print("Prompt     :", ex["prompt"])
        print("Predictions:", ex["predictions"])

print(f"\nSaved results to: {OUTPUT_CSV}")
