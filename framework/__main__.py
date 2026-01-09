from framework import Framework
from framework.worker import run_worker, start_up_bot
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Job ID to resume from", default=None)
    parser.add_argument("--framework-only", action="store_true", dest="framework_only", help="Starts ONLY the main bot. Regular run also starts the bot.", default=False)
    parser.add_argument("--worker-only", action="store_true", dest="worker_only", help="Starts only the worker", default=False)
    parser.add_argument("--model", type=str, default="qwen3:14b", help="Ollama model name")
    parser.add_argument("--cache-tools", type=str, nargs='*', default=[], help="List of tool names to cache results for (for debugging). Use '*' to cache all tools.")

    args = parser.parse_args()

    if args.framework_only:
        # Start the main framework bot after setting up the worker
        framework = Framework(base_model_name=args.model, resume_job_id=args.resume)
        framework.run_interactive()
    else:
        # Routine Startup
        # Start the bot and the worker
        # This will run this script again with --framework-only to start the main bot
        worker_only = args.worker_only
        if not worker_only:
            start_up_bot(args.resume)

        run_worker(cache_tools=args.cache_tools)