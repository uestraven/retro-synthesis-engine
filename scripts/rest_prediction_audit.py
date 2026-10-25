"""Does the model know when to leave space? Teacher-forced, on real NES music.

This is the experiment that separates the two explanations for the saturated
generation behaviour, and it is the only one that can:

- **The model predicts rests correctly.** It has learned rhythmic structure and
  free-running decoding is what goes wrong. Fix the decoder.
- **The model keeps predicting notes** even with real musical context. The
  learned distribution is wrong. Fixing the decoder would only paper over it.

Nothing is sampled. Real validation windows are teacher-forced, and at every
point where the true continuation is a *structural* choice the model's
distribution over the alternatives is recorded against the truth.

The decision point
------------------
After a complete note the grammar permits exactly these continuations:

    Channel_   another note at the same position  -> "keep playing"
    Position_  a later position in this bar       -> "leave space"
    Bar        the next bar                        -> "leave space"
    Section_   a section marker, then a bar
    Loop_end / <eos>                               -> "stop"

So "keep playing" versus "leave space" is a single competing prediction that
cross-entropy trains directly. If the model assigns most of its mass to
`Channel_` when the truth is `Position_` or `Bar`, the learned distribution
favours filling rows, and that is a training problem rather than a sampling
one.

Two refinements that matter
---------------------------
**Probabilities are grammar-masked and renormalized**, because the sampler
masks illegal tokens too. An unmasked number would mix in probability the
decoder can never act on. Both are reported.

**Skip distance is measured, not just the class.** Choosing `Position_` is not
enough: a model that always advances by one slot still fills every row. So
when the truth is a Position token, the audit compares the true skip distance
with the model's predicted distribution over skip distances. "Leaves space"
means putting mass on the *right* gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Importable when run as a file from a checkout where the package is not
# installed, which is the normal state of a fresh GPU host.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from typing import Any

from rse.tokenizers.grammar import State, TokenGrammar
from rse.training.dataset import load_windows

#: What a structural continuation means musically.
KEEP_PLAYING = "keep_playing"
LEAVE_SPACE_POSITION = "leave_space_position"
LEAVE_SPACE_BAR = "leave_space_bar"
STOP = "stop"
SECTION = "section"
CLASSES = (KEEP_PLAYING, LEAVE_SPACE_POSITION, LEAVE_SPACE_BAR, SECTION, STOP)


def classify_token(token: str) -> str | None:
    if token.startswith("Channel_"):
        return KEEP_PLAYING
    if token.startswith("Position_"):
        return LEAVE_SPACE_POSITION
    if token == "Bar":
        return LEAVE_SPACE_BAR
    if token.startswith("Section_"):
        return SECTION
    if token in ("<eos>", "Loop_end"):
        return STOP
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("caches/d4_full.jsonl"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--n-windows", type=int, default=120)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("rest_prediction_audit.json"))
    args = parser.parse_args()

    import torch

    from rse.training.train import load_checkpoint

    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    device = next(model.parameters()).device
    vocab = tokenizer.vocabulary
    best = payload.get("history", {}).get("best_valid_loss")
    print(
        f"checkpoint {args.checkpoint.name}: vocab {len(tokenizer)}, best valid {best}",
        flush=True,
    )

    windows = [w for w in load_windows(args.cache) if w.split == args.split]
    if not windows:
        print(f"error: no {args.split} windows in {args.cache}", file=sys.stderr)
        return 2
    windows = windows[: args.n_windows]
    print(f"auditing {len(windows)} {args.split} windows", flush=True)

    # Token ids grouped by structural class, for masking.
    ids_by_class: dict[str, list[int]] = defaultdict(list)
    for token, token_id in vocab._index.items():  # noqa: SLF001 - read-only
        name = classify_token(token)
        if name:
            ids_by_class[name].append(token_id)
    position_ids = sorted(ids_by_class[LEAVE_SPACE_POSITION])
    position_value = {
        token_id: int(vocab.token_of(token_id)[len("Position_") :]) for token_id in position_ids
    }

    truth_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    prob_of_truth: dict[str, list[float]] = defaultdict(list)
    mass_on_class: dict[str, dict[str, list[float]]] = {
        truth: defaultdict(list) for truth in CLASSES
    }
    baseline_mass: dict[str, dict[str, list[float]]] = {
        truth: defaultdict(list) for truth in CLASSES
    }
    skip_rows: list[dict[str, Any]] = []
    skip_abs_error: list[float] = []
    skip_exact = 0
    skip_total = 0

    for index, window in enumerate(windows, 1):
        ids = window.ids[: args.max_len]
        if len(ids) < 8:
            continue
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=device))[0]

        grammar = TokenGrammar(vocab)
        current_position: int | None = None
        for step, token_id in enumerate(ids):
            token = vocab.token_of(token_id)
            state_before = grammar.state
            # Only the note-or-next decision is a genuine "keep playing vs
            # leave space" choice. EXPECT_POSITION follows a Bar, where
            # continuing is not an option.
            if state_before is State.EXPECT_NOTE_OR_NEXT and step > 0:
                truth = classify_token(token)
                if truth is not None:
                    legal = grammar.allowed_ids()
                    if legal:
                        row = logits[step - 1]
                        masked = torch.full_like(row, float("-inf"))
                        legal_tensor = torch.tensor(legal, device=row.device)
                        masked[legal_tensor] = row[legal_tensor]
                        probs = torch.softmax(masked, dim=-1)

                        truth_counts[truth] += 1
                        # The uniform-over-legal-tokens baseline. Without it
                        # a class is credited for having more members:
                        # Position_ offers 64 legal tokens against Channel_'s
                        # 4, so even an untrained model puts ~0.83 on
                        # "leave space". Any claim about preference has to be
                        # read against this, not against 1/n_classes.
                        legal_set = set(legal)
                        for name in CLASSES:
                            members = [i for i in ids_by_class[name] if i in legal_set]
                            baseline_mass[truth][name].append(
                                len(members) / len(legal_set) if legal_set else 0.0
                            )
                        prob_of_truth[truth].append(float(probs[token_id]))
                        per_class = {}
                        for name in CLASSES:
                            members = [i for i in ids_by_class[name] if i in set(legal)]
                            mass = (
                                float(probs[torch.tensor(members, device=row.device)].sum())
                                if members
                                else 0.0
                            )
                            per_class[name] = mass
                            mass_on_class[truth][name].append(mass)
                        confusion[(truth, max(per_class, key=per_class.get))] += 1

                        # Skip distance: choosing Position_ is not enough if
                        # the model always advances by the minimum.
                        if truth == LEAVE_SPACE_POSITION and current_position is not None:
                            true_value = int(token[len("Position_") :])
                            true_skip = true_value - current_position
                            legal_positions = [i for i in position_ids if i in set(legal)]
                            if legal_positions and true_skip > 0:
                                pos_tensor = torch.tensor(legal_positions, device=row.device)
                                pos_probs = probs[pos_tensor]
                                total = float(pos_probs.sum())
                                if total > 0:
                                    normalized = pos_probs / total
                                    skips = [
                                        position_value[i] - current_position
                                        for i in legal_positions
                                    ]
                                    expected = sum(
                                        s * float(p)
                                        for s, p in zip(skips, normalized, strict=True)
                                        if s > 0
                                    )
                                    best_index = int(torch.argmax(pos_probs))
                                    predicted = skips[best_index]
                                    skip_total += 1
                                    skip_exact += int(predicted == true_skip)
                                    skip_abs_error.append(abs(predicted - true_skip))
                                    skip_rows.append(
                                        {
                                            "true_skip": true_skip,
                                            "argmax_skip": predicted,
                                            "expected_skip": round(expected, 3),
                                        }
                                    )
            if token.startswith("Position_"):
                current_position = int(token[len("Position_") :])
            elif token == "Bar":
                current_position = None
            grammar.advance(token)
        if index % 20 == 0:
            print(f"  {index}/{len(windows)}", flush=True)

    total_points = sum(truth_counts.values())
    if not total_points:
        print("error: no decision points found", file=sys.stderr)
        return 2

    # Grouped skip behaviour: when the music leaves a big gap, does the model?
    by_true_skip: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in skip_rows:
        by_true_skip[row["true_skip"]].append(row)

    report = {
        "checkpoint": str(args.checkpoint),
        "best_valid_loss": best,
        "split": args.split,
        "n_windows": len(windows),
        "n_decision_points": total_points,
        "truth_distribution": {
            name: round(truth_counts[name] / total_points, 4) for name in CLASSES
        },
        "mean_prob_of_true_token": {
            name: round(st.fmean(values), 4) for name, values in prob_of_truth.items() if values
        },
        "mean_mass_by_truth": {
            truth: {name: round(st.fmean(values), 4) for name, values in classes.items() if values}
            for truth, classes in mass_on_class.items()
            if any(classes.values())
        },
        "uniform_baseline_mass_by_truth": {
            truth: {name: round(st.fmean(values), 4) for name, values in classes.items() if values}
            for truth, classes in baseline_mass.items()
            if any(classes.values())
        },
        "argmax_confusion": {
            f"{truth}->{predicted}": count
            for (truth, predicted), count in sorted(confusion.items())
        },
        "skip": {
            "n": skip_total,
            "exact_match_rate": round(skip_exact / skip_total, 4) if skip_total else 0.0,
            "mean_abs_error": round(st.fmean(skip_abs_error), 4) if skip_abs_error else 0.0,
            "by_true_skip": {
                str(skip): {
                    "n": len(rows),
                    "mean_argmax_skip": round(st.fmean([r["argmax_skip"] for r in rows]), 3),
                    "mean_expected_skip": round(st.fmean([r["expected_skip"] for r in rows]), 3),
                }
                for skip, rows in sorted(by_true_skip.items())
                if len(rows) >= 5
            },
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{total_points} structural decision points in {len(windows)} windows")
    print("\nwhat the music actually does at those points:")
    for name in CLASSES:
        if truth_counts[name]:
            print(f"  {name:<24}{truth_counts[name]:>8}{truth_counts[name] / total_points:>8.1%}")

    print("\nmodel's probability mass, by what the truth was (grammar-masked):")
    header = f"  {'truth':<24}" + "".join(f"{c.split('_')[-1][:9]:>11}" for c in CLASSES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for truth in CLASSES:
        if truth not in report["mean_mass_by_truth"]:
            continue
        cells = "".join(
            f"{report['mean_mass_by_truth'][truth].get(c, 0.0):>11.3f}" for c in CLASSES
        )
        print(f"  {truth:<24}{cells}")

    print("\n  uniform-over-legal baseline (what no learning looks like):")
    for truth in CLASSES:
        if truth not in report["uniform_baseline_mass_by_truth"]:
            continue
        cells = "".join(
            f"{report['uniform_baseline_mass_by_truth'][truth].get(c, 0.0):>11.3f}" for c in CLASSES
        )
        print(f"  {truth:<24}{cells}")
        break

    print("\nP(true token) at the decision point:")
    for name, value in report["mean_prob_of_true_token"].items():
        print(f"  {name:<24}{value:>8.3f}")

    print(f"\nskip distance, when the music moves to a later position ({skip_total} cases):")
    print(f"  argmax matches the true skip   {report['skip']['exact_match_rate']:.1%}")
    print(f"  mean absolute skip error       {report['skip']['mean_abs_error']:.2f} slots")
    print(f"\n  {'true skip':>10}{'n':>7}{'argmax':>9}{'expected':>10}")
    print("  " + "-" * 34)
    for skip, entry in report["skip"]["by_true_skip"].items():
        print(
            f"  {skip:>10}{entry['n']:>7}{entry['mean_argmax_skip']:>9.2f}"
            f"{entry['mean_expected_skip']:>10.2f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
