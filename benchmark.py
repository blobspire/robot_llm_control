from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import traceback

from controller import SimulationController
from main import run_llm_task


DEFAULT_TASKS = [
    "pick up cube1",
    "move cube2 to the right of cube1",
    "stack cube1 on cube2",
    "put cube1 near cube2 but not touching",
    "move cube1, then return it to its original position",
]


def run_single_case(
    queue: mp.Queue,
    task: str,
    seed: int,
    max_steps: int,
    gui: bool,
) -> None:
    controller = SimulationController(gui=gui, sleep=gui, seed=seed)
    try:
        result = run_llm_task(
            controller=controller,
            user_task=task,
            max_steps=max_steps,
            print_trace=gui,
        )
        result["seed"] = seed
        queue.put(result)
    except Exception as exc:
        queue.put(
            {
                "task": task,
                "seed": seed,
                "completed": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        controller.close()


def run_case_with_timeout(
    task: str,
    seed: int,
    max_steps: int,
    gui: bool,
    run_timeout: float,
) -> dict:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
        target=run_single_case,
        args=(queue, task, seed, max_steps, gui),
    )
    process.start()
    process.join(run_timeout)

    if process.is_alive():
        process.terminate()
        process.join(10)
        return {
            "task": task,
            "seed": seed,
            "completed": False,
            "error": f"Timed out after {run_timeout:.1f}s",
        }

    if not queue.empty():
        return queue.get()

    return {
        "task": task,
        "seed": seed,
        "completed": False,
        "error": f"Benchmark worker exited with code {process.exitcode} without a result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM tabletop manipulation benchmarks.")
    parser.add_argument("--runs", type=int, default=20, help="Randomized runs per task.")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum LLM/tool steps per run.")
    parser.add_argument("--run-timeout", type=float, default=180.0, help="Maximum seconds allowed for one task/seed run.")
    parser.add_argument("--gui", action="store_true", help="Show PyBullet GUI during benchmark runs.")
    parser.add_argument("--output-dir", default="benchmarks", help="Directory for JSONL transcripts.")
    parser.add_argument("--task", action="append", help="Task to benchmark. Can be passed multiple times.")
    args = parser.parse_args()

    tasks = args.task or DEFAULT_TASKS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            for seed in range(args.runs):
                result = run_case_with_timeout(
                    task=task,
                    seed=seed,
                    max_steps=args.max_steps,
                    gui=args.gui,
                    run_timeout=args.run_timeout,
                )
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                status = "completed" if result.get("completed") else "failed"
                error = f" error={result.get('error')}" if result.get("error") else ""
                print(f"{task!r} seed={seed}: {status}{error}", flush=True)

    print(f"Wrote benchmark transcripts to {output_path}")


if __name__ == "__main__":
    main()
