#!/usr/bin/env python3
"""
LangGraph entrypoint for Planning Pattern Market Research Assistant.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from langgraph_workflow import LangGraphPlanningWorkflow
from trajectory_tracing import TrajectoryTracer


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
OPENAI_COMPATIBLE_PROVIDERS = {"openai", "huggingface", "hf"}
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


def normalize_provider(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "hf":
        return "huggingface"
    return provider


def resolve_model_from_env(provider: str) -> str:
    provider = normalize_provider(provider)
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
    default_model = "moonshotai/Kimi-K2.5" if provider == "huggingface" else "gpt-4.1-mini"
    return os.getenv("OPENAI_MODEL", default_model)


def apply_provider_defaults(provider: str) -> None:
    provider = normalize_provider(provider)
    if provider == "huggingface" and not os.getenv("OPENAI_BASE_URL", "").strip():
        os.environ["OPENAI_BASE_URL"] = HF_ROUTER_BASE_URL


def validate_environment() -> tuple[bool, list[str]]:
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    required_vars: dict[str, str] = {}
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        required_vars["OPENAI_API_KEY"] = "OpenAI API key for LLM calls"
    elif provider == "anthropic":
        required_vars["ANTHROPIC_API_KEY"] = "Anthropic API key for LLM calls"
    else:
        required_vars["LLM_PROVIDER"] = "Must be one of: openai, huggingface/hf, anthropic"

    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")

    return len(missing) == 0, missing


def save_report(content: str, output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    logger.info("📝 Report saved to: %s", output_path)
    logger.info("   File size: %s bytes", f"{output_path.stat().st_size:,}")


async def async_main(args: argparse.Namespace) -> None:
    start_time = datetime.now()
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    apply_provider_defaults(provider)
    model = resolve_model_from_env(provider)

    tracer = TrajectoryTracer(
        topic=args.topic,
        provider=provider,
        model=model,
        enabled=not args.no_trace,
        output_dir=Path(args.trace_dir),
    )

    try:
        workflow = LangGraphPlanningWorkflow(topic=args.topic, tracer=tracer)
        report = await workflow.execute()
        save_report(report, args.output)

        duration = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print("✅ SUCCESS (LangGraph)")
        print("=" * 60)
        print(f"Report saved to: {args.output}")
        print(f"Execution time: {duration:.1f} seconds")
        if tracer.enabled:
            print(f"Trajectory: {tracer.file_path}")
            print(f"Summary: {tracer.summary_path}")
        print("=" * 60 + "\n")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        logger.error("❌ ERROR: %s", str(e))
        logger.error("Troubleshooting:")
        logger.error("  1. Check provider key in .env")
        logger.error("  2. Check internet connectivity")
        logger.error("  3. Verify model/provider compatibility")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Planning Pattern Assistant (LangGraph)",
        epilog='Example: python main_langgraph.py "AI agent market size 2024-2026"',
    )
    parser.add_argument("topic", type=str, help="Research topic to analyze")
    parser.add_argument(
        "--output",
        type=str,
        default="planning_market_report_langgraph.md",
        help="Output file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model for selected provider",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "huggingface", "hf", "anthropic"],
        help="Override LLM provider",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=os.getenv("TRACE_DIR", "trajectories"),
        help="Directory for trajectory JSONL outputs",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable trajectory tracing for this run",
    )
    args = parser.parse_args()

    if args.provider:
        normalized_provider = normalize_provider(args.provider)
        os.environ["LLM_PROVIDER"] = normalized_provider
        logger.info("Using provider: %s", normalized_provider)
        apply_provider_defaults(normalized_provider)

    if args.model:
        provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
        if provider == "anthropic":
            os.environ["ANTHROPIC_MODEL"] = args.model
        else:
            os.environ["OPENAI_MODEL"] = args.model
        logger.info("Using model: %s", args.model)

    ok, missing = validate_environment()
    if not ok:
        logger.error("❌ Missing required environment variables:")
        for item in missing:
            logger.error("   - %s", item)
        sys.exit(1)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    apply_provider_defaults(provider)
    model = resolve_model_from_env(provider)
    print("\n" + "=" * 60)
    print("🎯 Planning Pattern Assistant (LangGraph)")
    print("=" * 60)
    print(f"📋 Topic: {args.topic}")
    print(f"📝 Output: {args.output}")
    print(f"🧠 Provider: {provider}")
    print(f"🤖 Model: {model}")
    print("=" * 60 + "\n")

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
