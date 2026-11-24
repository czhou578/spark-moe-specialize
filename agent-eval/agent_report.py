"""The report of the capacity measurement (plans/PLAN_AGENT_CAPACITY.md §7 and §11).

  python3 agent_report.py Q4                 # the table of a point, per model and per family
  python3 agent_report.py Q4 --ref Q1        # and the difference to a reference point, with the gates G1 to G4
  python3 agent_report.py --show results/agent-eval/Q4/qwen-L1-s0   # one transcript, for the manual read
  python3 agent_report.py Q4 --json --block-size 2176   # the JSON report, with the best cache hits for that block size

A rate averages the seeds of each task first, then it takes a 95% interval from a bootstrap over the tasks (2000
samples, a fixed seed). The difference of two points uses only the tasks that both have.
"""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import agent_eval as ae

RESULTS = ("pass", "hallucination", "wrong", "stopped", "server_error")
G1_POINTS, G2_POINTS, G4_MB = 3.0, 5.0, 3072
BURST, LOOP = 16, 5   # a run with 16 calls or more in one turn, or 5 repeated calls or more: a sign of a decode loop


def rows_of(name: str) -> list[dict]:
    f = ae.OUT / name / "rows.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    count = {r: sum(x["result"] == r for x in rows) for r in RESULTS}
    secs = sorted(x["seconds"] for x in rows)
    return {"runs": n, **{r: pct(c, n) for r, c in count.items()},
            "events": {k: sum(x["events"][k] > 0 for x in rows) for k in ("H1", "H2", "H3", "H4")},
            "median_s": statistics.median(secs) if secs else None,
            "p90_s": secs[int(0.9 * (len(secs) - 1))] if secs else None,
            "prompt_first": statistics.median([x["prompt_tokens_first"] or 0 for x in rows]) if rows else None,
            "bursts": sum(x.get("max_calls_turn", 0) >= BURST for x in rows),
            "loops": sum(x.get("repeat_calls", 0) >= LOOP for x in rows)}


def per_task(rows: list[dict], result: str) -> dict[str, float]:
    by = defaultdict(list)
    for x in rows:
        by[x["task"]].append(x["result"] == result)
    return {t: sum(v) / len(v) for t, v in by.items()}


def bootstrap(values: list[float], samples=2000) -> tuple[float, float, float]:
    """The mean of the values and its 95% interval from a bootstrap with a fixed seed."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(0)
    boot = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples))
    return statistics.fmean(values), boot[int(0.025 * samples)], boot[int(0.975 * samples)]


def diff(new: list[dict], ref: list[dict], result: str, samples=2000) -> tuple[float, float, float]:
    """The difference in points of the rate of `result`, new minus ref, on the common tasks, and its 95% interval."""
    a, b = per_task(new, result), per_task(ref, result)
    return bootstrap([100 * (a[t] - b[t]) for t in sorted(a.keys() & b.keys())], samples)


def best_hits(run_dirs: list[Path], block: int) -> dict:
    """The prompt tokens of all turns, and the most prefix-cache hits that a server with blocks of `block` tokens can
    give inside each run: on each turn, the prompt of the turn before, rounded down to full blocks."""
    prompt = best = 0
    for d in run_dirs:
        if (d / "transcript.json").exists():
            pts = [t["usage"].get("prompt_tokens") or 0 for t in json.loads((d / "transcript.json").read_text())["turns"]]
            prompt, best = prompt + sum(pts), best + sum(p // block * block for p in pts[:-1])
    return {"block_size": block, "prompt_tokens": prompt, "best_hits": best, "best_rate": round(pct(best, prompt), 1) if prompt else None}


def report(name: str, block: int | None = None) -> dict:
    """The JSON report of a point: for each model, task set and family, the rates with their intervals, the stops, the
    median seconds and output tokens. With `block`, also the best cache hits of each model."""
    rows, meta_f, groups = rows_of(name), ae.OUT / name / "point.json", []
    for model in sorted({x["model"] for x in rows}):
        mrows = [x for x in rows if x["model"] == model]
        for st in ["all"] + sorted({x.get("set", "dev") for x in mrows}):
            srows = [x for x in mrows if st in ("all", x.get("set", "dev"))]
            for fam in ["all"] + sorted({x["family"] for x in srows}):
                g = [x for x in srows if fam in ("all", x["family"])]
                rates = {r: dict(zip(("rate", "lo", "hi"), (round(v, 1) for v in
                                                           bootstrap([100 * p for p in per_task(g, r).values()]))))
                         for r in RESULTS}
                groups.append({"model": model, "set": st, "family": fam, "runs": len(g),
                               "tasks": len({x["task"] for x in g}), **rates, "events": summary(g)["events"],
                               "stops": dict(Counter(x.get("stop") for x in g)),
                               "median_s": statistics.median(x["seconds"] for x in g),
                               "median_completion_tokens": statistics.median(x.get("completion_tokens", 0) for x in g)})
    out = {"point": name, "meta": json.loads(meta_f.read_text()) if meta_f.exists() else {}, "groups": groups}
    if block:
        out["cache"] = {m: best_hits([ae.OUT / name / f"{m}-{x['task']}-s{x['seed']}" for x in rows if x["model"] == m],
                                     block) for m in sorted({x["model"] for x in rows})}
    return out


def watch_stops(t0: float, t1: float) -> int:
    log = ae.S / "logs/pair-watch.log"
    if not log.exists():
        return 0
    import time
    n = 0
    for line in log.read_text().splitlines():
        if "WATCH:" in line:
            try:
                ts = time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                continue
            n += t0 <= ts <= t1
    return n


def table(name: str, ref: str | None = None) -> str:
    rows, out = rows_of(name), []
    meta_f = ae.OUT / name / "point.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    out.append(f"# Point {name}: mix {meta.get('mix')}, ctx {meta.get('ctx')}, seeds {meta.get('seeds')}; lowest "
               f"available {meta.get('lowest_available_mb')} MB, lowest swap-free {meta.get('lowest_swap_free_mb')} MB")
    out.append("\n| Model | Family | Runs | Pass | Halluc. | Wrong | Stopped | Server err. | H1 | H2 | H3 | H4 | "
               "Median s | p90 s | Prompt tokens (first) | Bursts | Loops |\n|" + "---|" * 17)
    for model in sorted({x["model"] for x in rows}):
        mrows = [x for x in rows if x["model"] == model]
        for fam in ["all"] + sorted({x["family"] for x in mrows}):
            s = summary(mrows if fam == "all" else [x for x in mrows if x["family"] == fam])
            e = s["events"]
            out.append(f"| {model} | {fam} | {s['runs']} | {s['pass']:.1f}% | {s['hallucination']:.1f}% | "
                       f"{s['wrong']:.1f}% | {s['stopped']:.1f}% | {s['server_error']:.1f}% | {e['H1']} | {e['H2']} | "
                       f"{e['H3']} | {e['H4']} | {s['median_s']:.0f} | {s['p90_s']:.0f} | {s['prompt_first']:.0f} | {s['bursts']} | {s['loops']} |")
    if ref:
        rrows = rows_of(ref)
        out.append(f"\n## Against {ref}\n\n| Model | Halluc. diff (95% CI) | Pass diff (95% CI) | G1 | G2 | G3 | G4 |"
                   "\n|---|---|---|---|---|---|---|")
        a, b = {x["model"] for x in rows}, {x["model"] for x in rrows}
        pairs = [(m, m) for m in sorted(a & b)] or ([(a.pop(), b.pop())] if len(a) == len(b) == 1 else [])
        for model, ref_model in pairs:   # one model each, other names: deepseek-high against deepseek
            new, old = [x for x in rows if x["model"] == model], [x for x in rrows if x["model"] == ref_model]
            h, p = diff(new, old, "hallucination"), diff(new, old, "pass")
            g3 = sum(x["result"] == "server_error" for x in new) == 0
            low = meta.get("lowest_available_mb")
            g4 = low is not None and low >= G4_MB and watch_stops(meta.get("start", 0), meta.get("end", 0)) == 0
            ok = lambda b: "pass" if b else "FAIL"  # noqa: E731
            out.append(f"| {model if model == ref_model else f'{model} vs {ref_model}'} | {h[0]:+.1f} ({h[1]:+.1f} to {h[2]:+.1f}) | {p[0]:+.1f} ({p[1]:+.1f} to "
                       f"{p[2]:+.1f}) | {ok(h[0] <= G1_POINTS)} | {ok(p[0] >= -G2_POINTS)} | {ok(g3)} | {ok(g4)} |")
    return "\n".join(out)


def show(run_dir: Path) -> str:
    tr = json.loads((run_dir / "transcript.json").read_text())
    out = [f"{tr['task']} {tr['model']} seed {tr['seed']} ctx {tr['ctx']}: {tr['result']} (stop {tr['stop']}, "
           f"{tr['seconds']} s)", f"events: {json.dumps(tr['events'])}"]
    for i, t in enumerate(tr["turns"]):
        out.append(f"\n--- turn {i}: {t['finish']}, {t['usage'].get('completion_tokens')} tokens, {t['seconds']} s")
        if t.get("reasoning"):
            out.append(f"[reasoning] {t['reasoning'][:600]}")
        if t.get("content"):
            out.append(f"[text] {t['content'][:600]}")
        for c in t["calls"]:
            out.append(f"[call] {c['name']} {c['arguments'][:300]}" + (f"  INVALID: {c['error']}" if c.get("error") else ""))
            out.append(f"[result] {c.get('result', '')[:400]}")
    out.append(f"\nreport: {json.dumps(tr['report'], indent=1)}")
    if tr.get("check"):
        out.append(f"check: exit {tr['check']['exit']}\n{tr['check']['output'][-800:]}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("point", nargs="?")
    ap.add_argument("--ref")
    ap.add_argument("--show", type=Path)
    ap.add_argument("--json", action="store_true", help="the JSON report of the point instead of the table")
    ap.add_argument("--block-size", type=int, help="with --json: the block size of the prefix cache of the server")
    a = ap.parse_args()
    if a.show:
        print(show(a.show))
    else:
        print(json.dumps(report(a.point, a.block_size), indent=1) if a.json else table(a.point, a.ref))


if __name__ == "__main__":
    main()
