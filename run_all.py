"""
run_all.py — Full pipeline orchestrator.

Coordinates the complete experimental workflow:
  1. SFT (once) → outputs/sft_checkpoint/
  2. GRPO (cohorts × seeds) → outputs/{cohort}/seed_{seed}/
  3. Evaluation → outputs/results.csv
  4. Analysis → outputs/analysis/

Usage:
    python run_all.py                    # Full pipeline
    python run_all.py --skip-sft         # Skip SFT (already done)
    python run_all.py --dry-run          # Validate without training
    python run_all.py --cohorts A B      # Only specific cohorts
    python run_all.py --steps sft,grpo   # Only SFT + GRPO (no eval)
    python run_all.py --test-run         # Small-scale format validation
"""

import argparse
import subprocess
import sys
import os
import glob
import csv
import yaml
from datetime import datetime

STEPS = ["sft", "grpo", "eval", "analyze"]


def parse_args():
    parser = argparse.ArgumentParser(description="Full GRPO Pipeline Orchestrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cohorts", nargs="*", default=None,
                        help="Cohorts to run (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Seeds to run (default: 0,1,2,3)")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"Comma-separated steps to run: {STEPS}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--test-run", action="store_true",
                        help="Small-scale run (10 samples, 1 seed) for format validation")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_cmd(cmd: list[str], dry: bool = False) -> int:
    """Run a shell command. Returns exit code."""
    print(f"  CMD: {' '.join(cmd)}")
    if dry:
        print("  [DRY-RUN] Skipping execution.")
        return 0
    result = subprocess.run(cmd)
    return result.returncode


def _finalize_test_report(report_path: str, args, cohort_names: list[str], seeds: list[int]):
    """Append final summary section to the test-run report."""
    with open(report_path, "a") as rf:
        rf.write("=" * 60 + "\n")
        rf.write("TEST RUN COMPLETE\n")
        rf.write("=" * 60 + "\n\n")
        rf.write(f"Config:          {args.config}\n")
        rf.write(f"Cohorts tested:  {cohort_names}\n")
        rf.write(f"Seeds:           {seeds}\n")
        rf.write(f"Output dir:      {args.output}\n")
        rf.write(f"Report:          {report_path}\n\n")
        rf.write("Stages exercised: SFT → GRPO → Eval → Analysis\n")
        rf.write("All pipeline components validated.\n")
    print(f"\nTest-run report: {report_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cohorts_cfg = cfg["cohorts"]
    num_seeds = cfg["evaluation"]["num_seeds"]

    cohort_names = args.cohorts or list(cohorts_cfg.keys())
    seeds = args.seeds or list(range(num_seeds))
    steps = [s.strip() for s in args.steps.split(",")]

    # ── Test-run bootstrapping ──────────────────────────────────
    test_run = args.test_run
    if test_run:
        tr = cfg.get("test_run", {})
        args.output = tr.get("output_dir", "outputs/test_run")
        seeds = [0]
        num_seeds = 1
        steps = STEPS  # Run all steps
        # Write a temp config with test overrides for subprocesses
        test_config_path = os.path.join(args.output, "_config.yaml")
        os.makedirs(args.output, exist_ok=True)
        cfg_test = dict(cfg)
        cfg_test["training"]["sft"]["max_samples"] = tr.get("sft_max_samples", 10)
        cfg_test["training"]["grpo"]["max_samples"] = tr.get("grpo_max_samples", 10)
        cfg_test["training"]["grpo"]["num_generations"] = tr.get("num_generations", 4)
        cfg_test["training"]["grpo"]["max_completion_length"] = tr.get("max_completion_length", 128)
        cfg_test["evaluation"]["num_seeds"] = 1
        with open(test_config_path, "w") as f:
            yaml.dump(cfg_test, f)
        args.config = test_config_path
        report_path = tr.get("report_file", "outputs/test_run/report.txt")

    print("=" * 60)
    print("GRPO Dynamic Reward-Gating Pipeline")
    print("=" * 60)
    print(f"Mode:    {'TEST-RUN' if test_run else 'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"Cohorts: {cohort_names}")
    print(f"Seeds:   {seeds}")
    print(f"Steps:   {steps}")
    print()

    os.makedirs(args.output, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────
    log_path = os.path.join(args.output, "pipeline_log.csv")
    if not args.dry_run and not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "step", "cohort", "seed", "status", "output"])

    def log(step: str, cohort: str, seed: str, status: str, output: str):
        if not args.dry_run:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), step, cohort, seed, status, output])

    # ═══════════════════════════════════════════════════════════
    # Step 1: SFT
    # ═══════════════════════════════════════════════════════════
    sft_output = os.path.join(args.output, "sft_checkpoint", "merged")
    sft_adapter = os.path.join(args.output, "sft_checkpoint")
    sft_marker = os.path.join(sft_adapter, ".resume_marker")
    if "sft" in steps:
        skip_sft = args.skip_sft
        # Detect completion: merged dir exists AND no resume marker
        sft_complete = (os.path.exists(sft_output) and os.listdir(sft_output) 
                        and not os.path.exists(sft_marker))
        if test_run and not skip_sft and sft_complete:
            print("[1/4] SFT: Test-run checkpoint already exists — skipping")
            skip_sft = True
        elif sft_complete:
            print(f"[1/4] SFT: Already completed at {sft_output}")
            skip_sft = True
        elif os.path.exists(sft_marker):
            print("[1/4] SFT: Incomplete run detected — will resume")
        if skip_sft:
            print("[1/4] SFT: SKIPPED")
        elif args.dry_run:
            print("[1/4] SFT: DRY-RUN...")
            test_flag = ["--test-run"] if test_run else []
            cmd = ["python", "run_sft.py", "--config", args.config, "--dry-run"] + test_flag
            run_cmd(cmd, dry=False)
        else:
            print("[1/4] SFT: Starting...")
            test_flag = ["--test-run"] if test_run else []
            out_flag = ["--output", os.path.join(args.output, "sft_checkpoint")]
            cmd = ["python", "run_sft.py", "--config", args.config] + test_flag + out_flag
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print("[ERROR] SFT failed. Aborting.")
                sys.exit(1)
            print("[1/4] SFT: Complete.")
    else:
        print("[1/4] SFT: Not in selected steps.")

    # ═══════════════════════════════════════════════════════════
    # Step 1.5: All-Zero Dataset Filter (once, after SFT, before GRPO)
    # ═══════════════════════════════════════════════════════════
    filtered_dataset_dir = "outputs/filtered_grpo/filtered_dataset"
    if "grpo" in steps and not args.dry_run and not test_run:
        if not os.path.exists(filtered_dataset_dir):
            cohort0 = cohort_names[0]
            seed0 = seeds[0]
            print(f"\n[1.5/4] Dataset Filter: Running all-zero filter (once)")
            cmd = [
                "python", "run_grpo.py",
                "--config", args.config,
                "--cohort", cohort0,
                "--seed", str(seed0),
                "--sft-checkpoint", sft_output,
                "--filter-dataset",
            ]
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print("[WARN] Dataset filter failed. Continuing without filter...")
            else:
                print("[1.5/4] Dataset Filter: Complete.")
        else:
            print(f"\n[1.5/4] Dataset Filter: Already cached at {filtered_dataset_dir}")

    # ═══════════════════════════════════════════════════════════
    # Step 2: GRPO (cohorts × seeds)
    # ═══════════════════════════════════════════════════════════
    if "grpo" in steps:
        total_runs = len(cohort_names) * len(seeds)
        print(f"\n[2/4] GRPO: {total_runs} runs ({len(cohort_names)} cohorts × {len(seeds)} seeds)")
        run_idx = 0
        test_flag = ["--test-run"] if test_run else []

        for cohort in cohort_names:
            for seed in seeds:
                run_idx += 1
                run_output = os.path.join(args.output, cohort, f"seed_{seed}")
                
                # Check if already completed or was interrupted
                adapter_exists = os.path.exists(os.path.join(run_output, "adapter_model.safetensors"))
                marker_exists = os.path.exists(os.path.join(run_output, ".resume_marker"))
                checkpoint_dirs = glob.glob(os.path.join(run_output, "checkpoint-*"))
                
                if adapter_exists and not marker_exists:
                    # Clean completion — skip
                    print(f"  [{run_idx}/{total_runs}] {cohort}/seed_{seed}: SKIPPED (exists)")
                    continue
                elif marker_exists:
                    print(f"  [{run_idx}/{total_runs}] {cohort}/seed_{seed}: RESUMING (incomplete)")
                elif checkpoint_dirs:
                    print(f"  [{run_idx}/{total_runs}] {cohort}/seed_{seed}: RESUMING ({len(checkpoint_dirs)} checkpoints)")

                print(f"  [{run_idx}/{total_runs}] {cohort}/seed_{seed}: Running...")
                cmd = [
                    "python", "run_grpo.py",
                    "--config", args.config,
                    "--cohort", cohort,
                    "--seed", str(seed),
                    "--sft-checkpoint", sft_output,
                    "--output", run_output,
                ] + test_flag
                if args.dry_run:
                    cmd.append("--dry-run")
                # Always run in test-run mode (even with --dry-run) so smoke tests execute
                rc = subprocess.run(cmd).returncode if not args.dry_run or test_run else 0
                if rc != 0:
                    log("grpo", cohort, str(seed), "FAILED", run_output)
                    print(f"  [WARN] {cohort}/seed_{seed} failed (exit {rc}). Continuing...")
                else:
                    log("grpo", cohort, str(seed), "OK", run_output)

        print("[2/4] GRPO: Complete.")
    else:
        print("[2/4] GRPO: Not in selected steps.")

    # ═══════════════════════════════════════════════════════════
    # Step 3: Evaluation
    # ═══════════════════════════════════════════════════════════
    if "eval" in steps:
        print("\n[3/4] Evaluation: Starting...")
        cmd = [
            "python", "evaluate.py",
            "--config", args.config,
            "--checkpoints-root", args.output,
        ]
        if test_run:
            tr = cfg.get("test_run", {})
            cmd += ["--max-samples", str(tr.get("eval_max_samples", 3))]
            cmd.append("--test-run")
        if args.dry_run:
            cmd.append("--dry-run")
        rc = subprocess.run(cmd).returncode if not args.dry_run or test_run else 0
        if rc != 0 and not args.dry_run:
            print("[WARN] Evaluation completed with errors.")
        print("[3/4] Evaluation: Complete.")
    else:
        print("[3/4] Evaluation: Not in selected steps.")

    # ═══════════════════════════════════════════════════════════
    # Step 4: Analysis
    # ═══════════════════════════════════════════════════════════
    if "analyze" in steps:
        print("\n[4/4] Analysis: Starting...")
        cmd = [
            "python", "analyze.py",
            "--config", args.config,
            "--results", os.path.join(args.output, "results.csv"),
        ]
        if test_run:
            cmd.append("--test-run")
        if args.dry_run:
            cmd.append("--dry-run")
        rc = subprocess.run(cmd).returncode if not args.dry_run or test_run else 0
        if rc != 0 and not args.dry_run:
            print("[WARN] Analysis completed with errors.")
        print("[4/4] Analysis: Complete.")
    else:
        print("[4/4] Analysis: Not in selected steps.")

    # ── Test-run: final report ──────────────────────────────────
    if test_run:
        try:
            _finalize_test_report(report_path, args, cohort_names, seeds)
        except Exception as e:
            print(f"[WARN] Could not finalize test report: {e}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Output: {args.output}/")
    print(f"Log:    {log_path}")


if __name__ == "__main__":
    main()
