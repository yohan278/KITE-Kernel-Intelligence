#!/usr/bin/env python3
"""Wrapper to start vLLM server with broken system packages mocked out."""
import sys
import types

# Mock broken system packages that crash with numpy 2.x
import importlib.machinery
for mod_name in ['tensorflow', 'tensorflow.python', 'h5py']:
    if mod_name not in sys.modules:
        mock = types.ModuleType(mod_name)
        mock.__version__ = '0.0.0'
        mock.__path__ = []
        mock.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
        sys.modules[mod_name] = mock

# Now run vLLM server
from vllm.entrypoints.openai.api_server import run_server, FlexibleArgumentParser

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Import here to get the arg parser setup from vllm
    from vllm.entrypoints.openai.api_server import make_arg_parser
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    import asyncio
    asyncio.run(run_server(args))
