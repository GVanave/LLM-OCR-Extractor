"""CLI for the eval harness.

  python -m evals prepare                       download CORD and write data/cord/{dev,test}
  python -m evals stats --set dev               data report (shape of the data, label quality)
  python -m evals run --set dev --prompt v1     run an eval with Gemini
  python -m evals run --set dev --mock          run the harness without an API key
  python -m evals compare RUN_A RUN_B           diff two runs
"""
import argparse
import json

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="python -m evals")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download and convert CORD v2")
    p.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])

    s = sub.add_parser("stats", help="data report for a split")
    s.add_argument("--set", default="dev", choices=["dev", "test"])
    s.add_argument("--show-inconsistent", action="store_true", help="list labels that fail math checks")

    r = sub.add_parser("run", help="run an eval")
    r.add_argument("--set", default="dev", choices=["dev", "test"])
    r.add_argument("--model", default="gemini-2.5-flash")
    r.add_argument("--prompt", default="v1", help="uses prompts/receipt_<version>.txt")
    r.add_argument("--limit", type=int, help="only the first N receipts")
    r.add_argument("--workers", type=int, default=2, help="parallel requests (mind your rate limit)")
    r.add_argument("--no-cache", action="store_true", help="always call the model")
    r.add_argument("--mock", action="store_true", help="fake predictions from labels, no API calls")

    c = sub.add_parser("compare", help="compare two runs")
    c.add_argument("run_a")
    c.add_argument("run_b")

    args = ap.parse_args()
    if args.cmd == "prepare":
        from evals.cord import prepare
        prepare(args.splits)
    elif args.cmd == "stats":
        from evals.cord import load_split, stats
        report = stats(load_split(args.set))
        if not args.show_inconsistent:
            report["inconsistent_labels"] = f"{len(report['inconsistent_labels'])} (use --show-inconsistent)"
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.cmd == "run":
        if args.set == "test":
            print("NOTE: 'test' is the frozen split. Use it for final scores, not for tuning.\n")
        from evals.runner import run
        run(args.set, args.model, args.prompt, mock=args.mock, limit=args.limit,
            workers=args.workers, use_cache=not args.no_cache)
    elif args.cmd == "compare":
        from evals.compare import compare
        compare(args.run_a, args.run_b)


if __name__ == "__main__":
    main()
