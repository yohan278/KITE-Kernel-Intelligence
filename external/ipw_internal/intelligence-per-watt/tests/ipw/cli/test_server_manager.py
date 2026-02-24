"""Tests for server_manager module."""

import pytest
from unittest.mock import MagicMock, patch, Mock
import subprocess

from ipw.cli.server_manager import (
    ServerConfig,
    InferenceServerManager,
    parse_submodel_spec,
    build_server_configs,
)


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_default_values(self):
        config = ServerConfig(
            model_id="Qwen/Qwen3-4B",
            alias="main",
            backend="vllm",
        )
        assert config.port == 8000
        assert config.gpu_ids == []
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9
        assert config.max_model_len == 32768

    def test_custom_values(self):
        config = ServerConfig(
            model_id="Qwen/Qwen3-32B",
            alias="large",
            backend="vllm",
            port=9000,
            gpu_ids=[0, 1],
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_len=16384,
        )
        assert config.port == 9000
        assert config.gpu_ids == [0, 1]
        assert config.tensor_parallel_size == 2

    def test_ollama_config(self):
        config = ServerConfig(
            model_id="llama3.2:1b",
            alias="small",
            backend="ollama",
            port=11434,
        )
        assert config.backend == "ollama"
        assert config.port == 11434


class TestParseSubmodelSpec:
    """Tests for parse_submodel_spec function."""

    def test_parse_vllm_spec(self):
        config = parse_submodel_spec("math:vllm:Qwen/Qwen2.5-Math-72B")
        assert config.alias == "math"
        assert config.backend == "vllm"
        assert config.model_id == "Qwen/Qwen2.5-Math-72B"
        assert config.port == 8000  # Default vLLM port

    def test_parse_ollama_spec(self):
        config = parse_submodel_spec("small:ollama:llama3.2:1b")
        assert config.alias == "small"
        assert config.backend == "ollama"
        assert config.model_id == "llama3.2:1b"
        assert config.port == 11434  # Default Ollama port

    def test_parse_invalid_spec_too_few_parts(self):
        with pytest.raises(ValueError, match="Invalid submodel spec"):
            parse_submodel_spec("math:vllm")

    def test_parse_invalid_backend(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            parse_submodel_spec("math:invalid:model")

    def test_parse_spec_with_colons_in_model_id(self):
        # Ollama models often have colons like "llama3.2:1b"
        config = parse_submodel_spec("small:ollama:llama3.2:1b:latest")
        assert config.model_id == "llama3.2:1b:latest"


class TestBuildServerConfigs:
    """Tests for build_server_configs function."""

    def test_main_model_only(self):
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=[],
            base_port=8000,
        )
        assert len(configs) == 1
        assert configs[0].alias == "main"
        assert configs[0].model_id == "Qwen/Qwen3-4B"
        assert configs[0].port == 8000

    def test_with_vllm_submodels(self):
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=[
                "math:vllm:Qwen/Qwen2.5-Math-72B",
                "code:vllm:Qwen/Qwen2.5-Coder-32B",
            ],
            base_port=8000,
        )
        assert len(configs) == 3
        # Main model gets base port
        assert configs[0].port == 8000
        # Submodels get incremented ports
        assert configs[1].port == 8001
        assert configs[2].port == 8002

    def test_with_ollama_submodel(self):
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=["small:ollama:llama3.2:1b"],
            base_port=8000,
        )
        assert len(configs) == 2
        # Ollama keeps its default port
        assert configs[1].port == 11434
        assert configs[1].backend == "ollama"

    def test_custom_base_port(self):
        configs = build_server_configs(
            main_model="Qwen/Qwen3-4B",
            main_alias="main",
            submodel_specs=["math:vllm:Qwen/Qwen2.5-Math-72B"],
            base_port=9000,
        )
        assert configs[0].port == 9000
        assert configs[1].port == 9001


class TestInferenceServerManager:
    """Tests for InferenceServerManager class."""

    def test_init_auto_assigns_gpus(self):
        configs = [
            ServerConfig(model_id="model1", alias="m1", backend="vllm"),
            ServerConfig(model_id="model2", alias="m2", backend="vllm"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=True)

        assert manager.configs[0].gpu_ids == [0]
        assert manager.configs[1].gpu_ids == [1]

    def test_init_respects_tensor_parallel_size(self):
        configs = [
            ServerConfig(
                model_id="large", alias="l", backend="vllm", tensor_parallel_size=2
            ),
            ServerConfig(model_id="small", alias="s", backend="vllm"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=True)

        # First model gets GPUs 0,1
        assert manager.configs[0].gpu_ids == [0, 1]
        # Second model gets GPU 2
        assert manager.configs[1].gpu_ids == [2]

    def test_init_skips_gpu_assignment_for_ollama(self):
        configs = [
            ServerConfig(model_id="llama", alias="l", backend="ollama"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=True)

        # Ollama doesn't get GPU assignment
        assert manager.configs[0].gpu_ids == []

    def test_init_no_auto_assign(self):
        configs = [
            ServerConfig(model_id="model1", alias="m1", backend="vllm"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=False)

        assert manager.configs[0].gpu_ids == []

    @patch("ipw.cli.server_manager.subprocess.Popen")
    @patch("ipw.cli.server_manager.InferenceServerManager._wait_for_vllm_ready")
    def test_start_vllm_server(self, mock_wait, mock_popen):
        mock_wait.return_value = True
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        configs = [
            ServerConfig(
                model_id="Qwen/Qwen3-4B",
                alias="main",
                backend="vllm",
                port=8000,
                gpu_ids=[0],
            ),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=False)
        urls = manager.start_all()

        assert "main" in urls
        assert urls["main"] == "http://localhost:8000/v1"
        mock_popen.assert_called_once()

        # Check CUDA_VISIBLE_DEVICES was set
        call_kwargs = mock_popen.call_args[1]
        assert "CUDA_VISIBLE_DEVICES" in call_kwargs["env"]
        assert call_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"

    @patch("ipw.cli.server_manager.subprocess.Popen")
    @patch("ipw.cli.server_manager.InferenceServerManager._wait_for_vllm_ready")
    def test_stop_all_terminates_processes(self, mock_wait, mock_popen):
        mock_wait.return_value = True
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_popen.return_value = mock_process

        configs = [
            ServerConfig(model_id="model", alias="m", backend="vllm", gpu_ids=[0]),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=False)
        manager.start_all()

        manager.stop_all()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()

    def test_get_url(self):
        configs = [
            ServerConfig(model_id="model", alias="test", backend="vllm"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=False)
        manager._urls["test"] = "http://localhost:8000/v1"

        assert manager.get_url("test") == "http://localhost:8000/v1"
        assert manager.get_url("nonexistent") is None

    def test_is_running_with_process(self):
        configs = [
            ServerConfig(model_id="model", alias="test", backend="vllm"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=False)

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process running
        manager._processes["test"] = mock_process

        assert manager.is_running("test") is True

        mock_process.poll.return_value = 0  # Process exited
        assert manager.is_running("test") is False


class TestInferenceServerManagerWarmup:
    """Tests for warmup functionality."""

    @patch("urllib.request.urlopen")
    def test_warmup_vllm(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"choices": [{"message": {"content": "hi"}}]}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        configs = [
            ServerConfig(model_id="Qwen/Qwen3-4B", alias="main", backend="vllm"),
        ]
        manager = InferenceServerManager(configs, auto_assign_gpus=False)
        manager._urls["main"] = "http://localhost:8000/v1"

        # Should not raise
        manager.warmup("main")
        mock_urlopen.assert_called_once()

    def test_warmup_unknown_alias(self):
        configs = []
        manager = InferenceServerManager(configs, auto_assign_gpus=False)

        # Should not raise, just log warning
        manager.warmup("unknown")
