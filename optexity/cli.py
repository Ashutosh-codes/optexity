import argparse
import logging
import os
import subprocess
import sys

from dotenv import load_dotenv
from uvicorn import run

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")
else:
    load_dotenv(env_path)


def install_browsers() -> None:
    """Install Playwright + Patchright browsers."""
    try:
        subprocess.run(
            ["playwright", "install", "--with-deps", "chromium", "chrome"],
            check=True,
        )
        subprocess.run(
            ["patchright", "install", "chromium", "chrome"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install browsers", file=sys.stderr)
        sys.exit(e.returncode)


def run_inference(args: argparse.Namespace) -> None:
    from optexity.inference.child_process import get_app_with_endpoints

    app = get_app_with_endpoints(
        is_aws=args.is_aws, child_id=args.child_process_id, port=args.port
    )
    run(
        app,
        host=args.host,
        port=args.port,
    )


def build_cached_automation(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from optexity.inference.core.interaction.cached_automation import (
        build_cached_automation,
    )
    from optexity.schema.automation import Automation

    automation = Automation.model_validate_json(Path(args.automation).read_text())
    if args.llm:
        from optexity.inference.core.interaction.llm_automation_builder import (
            build_cached_automation_with_llm,
        )

        cached, report = build_cached_automation_with_llm(
            automation, Path(args.logs_directory), model_name=args.model
        )
        print(
            f"LLM build ({report.model}): {len(report.attempts)} attempt(s), "
            f"{report.tokens.total_tokens} tokens, kept agentic: {report.kept_agentic or 'none'}"
        )
        for attempt in report.attempts:
            for error in attempt.errors:
                print(f"  node {attempt.node_index} attempt {attempt.attempt}: {error}")
    else:
        cached = build_cached_automation(automation, Path(args.logs_directory))
    Path(args.output).write_text(
        json.dumps(cached.model_dump(exclude_defaults=True), indent=2) + "\n"
    )
    print(f"Wrote {args.output} ({len(cached.nodes)} node(s))")


def refine_automation(args: argparse.Namespace) -> None:
    from pathlib import Path

    from optexity.inference.core.learning_loop import run_learning_loop
    from optexity.schema.automation import Automation

    automation = Automation.model_validate_json(Path(args.automation).read_text())
    _, results = run_learning_loop(
        automation,
        endpoint_name=args.endpoint,
        out_dir=Path(args.out_dir),
        live_path=Path(args.live_path),
        runs_dir=Path(args.runs_dir),
        server_url=args.server,
        max_rounds=args.rounds,
        builder=args.builder,
    )
    print(
        f"{'round':>5} {'status':>8} {'tokens':>8} {'steps_s':>8} {'task_s':>7}  changes"
    )
    for r in results:
        print(
            f"{r.round:>5} {r.status:>8} {r.tokens:>8} {r.nodes_done_s!s:>8} "
            f"{r.task_s!s:>7}  {'; '.join(r.changes) or '-'}"
        )
    print(f"Final automation: {Path(args.out_dir) / 'final.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="optexity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # install-browsers
    # ---------------------------
    install_cmd = subparsers.add_parser(
        "install_browsers",
        help="Install required browsers for Optexity",
        aliases=["install-browsers"],
    )
    install_cmd.set_defaults(func=lambda _: install_browsers())

    # ---------------------------
    # inference
    # ---------------------------
    inference_cmd = subparsers.add_parser(
        "inference", help="Run Optexity inference server"
    )
    inference_cmd.add_argument("--host", default="0.0.0.0")
    inference_cmd.add_argument("--port", type=int, default=9000)
    inference_cmd.add_argument(
        "--child_process_id", "--child-process-id", type=int, default=0
    )
    inference_cmd.add_argument(
        "--is_aws", "--is-aws", action="store_true", default=False
    )

    inference_cmd.set_defaults(func=run_inference)

    # ---------------------------
    # build-cached-automation
    # ---------------------------
    cached_cmd = subparsers.add_parser(
        "build_cached_automation",
        help="Compile an automation's agentic_task nodes from a run's action caches",
        aliases=["build-cached-automation"],
    )
    cached_cmd.add_argument("--automation", required=True)
    cached_cmd.add_argument(
        "--logs-directory",
        "--logs_directory",
        required=True,
        help="The run's logs directory, holding step_<i>/action_cache.json",
    )
    cached_cmd.add_argument("--output", "-o", required=True)
    cached_cmd.add_argument(
        "--llm",
        action="store_true",
        help="Build with the LLM (docs + cache, validated) instead of the code builder",
    )
    cached_cmd.add_argument("--model", default=None, help="litellm model for --llm")
    cached_cmd.set_defaults(func=build_cached_automation)

    # ---------------------------
    # refine-automation
    # ---------------------------
    refine_cmd = subparsers.add_parser(
        "refine_automation",
        help="Run, cache, and rebuild an automation in a loop on the local server",
        aliases=["refine-automation"],
    )
    refine_cmd.add_argument("--automation", required=True)
    refine_cmd.add_argument(
        "--endpoint", required=True, help="endpoint_name to submit to /inference"
    )
    refine_cmd.add_argument(
        "--live-path",
        "--live_path",
        default="runs/loop/current.json",
        help="File the server's OPTEXITY_LOCAL_AUTOMATION points at",
    )
    refine_cmd.add_argument("--out-dir", "--out_dir", required=True)
    refine_cmd.add_argument("--runs-dir", "--runs_dir", default="runs")
    refine_cmd.add_argument("--server", default="http://localhost:9000")
    refine_cmd.add_argument("--rounds", type=int, default=5)
    refine_cmd.add_argument(
        "--builder",
        choices=["code", "llm"],
        default="code",
        help="How agentic nodes are compiled from the action cache",
    )
    refine_cmd.set_defaults(func=refine_automation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
