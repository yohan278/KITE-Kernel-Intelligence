"""Custom vLLM launcher that applies patches before starting the server.

Usage: python -m grid_eval.vllm_launcher [vllm args...]

This is equivalent to `python -m vllm.entrypoints.openai.api_server`
but applies compatibility patches first (e.g., registering model
architectures not yet in transformers).
"""

from __future__ import annotations


def main() -> None:
    # Apply patches before vLLM imports transformers AutoConfig
    from grid_eval.vllm_patches import register_missing_configs, patch_tokenizer_compat

    register_missing_configs()
    patch_tokenizer_compat()

    # Import and run the vLLM OpenAI-compatible server
    # Mirror vLLM's own __main__ block
    import uvloop
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        cli_env_setup,
        make_arg_parser,
        run_server,
        validate_parsed_serve_args,
    )

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
