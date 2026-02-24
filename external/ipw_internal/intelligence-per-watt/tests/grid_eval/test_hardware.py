"""Tests for grid_eval.hardware module."""

import os
import pytest

from grid_eval.config import HardwareConfig, ResourceConfig
from grid_eval.hardware import HardwareManager, ResourceManager


class TestHardwareManager:
    """Tests for HardwareManager class (legacy API)."""

    def test_context_manager_sets_env_vars(self):
        with HardwareManager(HardwareConfig.A100_1GPU):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            assert os.environ.get("OMP_NUM_THREADS") == "8"
            assert os.environ.get("MKL_NUM_THREADS") == "8"

    def test_context_manager_restores_env_vars(self):
        # Set initial values
        os.environ["CUDA_VISIBLE_DEVICES"] = "original_cuda"
        os.environ["OMP_NUM_THREADS"] = "original_omp"

        try:
            with HardwareManager(HardwareConfig.A100_4GPU):
                assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"
                assert os.environ.get("OMP_NUM_THREADS") == "32"

            # Verify restoration
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "original_cuda"
            assert os.environ.get("OMP_NUM_THREADS") == "original_omp"
        finally:
            # Cleanup
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("OMP_NUM_THREADS", None)

    def test_context_manager_restores_unset_vars(self):
        # Ensure vars are unset
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ.pop("MKL_NUM_THREADS", None)

        with HardwareManager(HardwareConfig.A100_1GPU):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

        # Verify they're unset again
        assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
        assert os.environ.get("OMP_NUM_THREADS") is None
        assert os.environ.get("MKL_NUM_THREADS") is None

    def test_cuda_devices_property(self):
        hw = HardwareManager(HardwareConfig.A100_1GPU)
        assert hw.cuda_devices == "0"

        hw = HardwareManager(HardwareConfig.A100_4GPU)
        assert hw.cuda_devices == "0,1,2,3"

    def test_num_threads_property(self):
        hw = HardwareManager(HardwareConfig.A100_1GPU)
        assert hw.num_threads == 8

        hw = HardwareManager(HardwareConfig.A100_4GPU)
        assert hw.num_threads == 32

    def test_describe(self):
        hw = HardwareManager(HardwareConfig.A100_1GPU)
        desc = hw.describe()
        assert "1 GPU" in desc
        assert "8 CPU" in desc

        hw = HardwareManager(HardwareConfig.A100_4GPU)
        desc = hw.describe()
        assert "4 GPU" in desc
        assert "32 CPU" in desc

    def test_exception_in_context_still_restores(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "original"

        try:
            with pytest.raises(ValueError):
                with HardwareManager(HardwareConfig.A100_1GPU):
                    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
                    raise ValueError("Test exception")

            # Verify restoration happened despite exception
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "original"
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)


class TestResourceManager:
    """Tests for ResourceManager class (new API)."""

    def test_context_manager_sets_env_vars(self):
        with ResourceManager(ResourceConfig.ONE_GPU_8CPU):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            assert os.environ.get("OMP_NUM_THREADS") == "8"
            assert os.environ.get("MKL_NUM_THREADS") == "8"

    def test_context_manager_sets_4gpu_env_vars(self):
        with ResourceManager(ResourceConfig.FOUR_GPU_32CPU):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"
            assert os.environ.get("OMP_NUM_THREADS") == "32"
            assert os.environ.get("MKL_NUM_THREADS") == "32"

    def test_context_manager_restores_env_vars(self):
        # Set initial values
        os.environ["CUDA_VISIBLE_DEVICES"] = "original_cuda"
        os.environ["OMP_NUM_THREADS"] = "original_omp"

        try:
            with ResourceManager(ResourceConfig.FOUR_GPU_32CPU):
                assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"
                assert os.environ.get("OMP_NUM_THREADS") == "32"

            # Verify restoration
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "original_cuda"
            assert os.environ.get("OMP_NUM_THREADS") == "original_omp"
        finally:
            # Cleanup
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("OMP_NUM_THREADS", None)

    def test_context_manager_restores_unset_vars(self):
        # Ensure vars are unset
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("OMP_NUM_THREADS", None)
        os.environ.pop("MKL_NUM_THREADS", None)

        with ResourceManager(ResourceConfig.ONE_GPU_8CPU):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

        # Verify they're unset again
        assert os.environ.get("CUDA_VISIBLE_DEVICES") is None
        assert os.environ.get("OMP_NUM_THREADS") is None
        assert os.environ.get("MKL_NUM_THREADS") is None

    def test_cuda_devices_property(self):
        rm = ResourceManager(ResourceConfig.ONE_GPU_8CPU)
        assert rm.cuda_devices == "0"

        rm = ResourceManager(ResourceConfig.FOUR_GPU_32CPU)
        assert rm.cuda_devices == "0,1,2,3"

        rm = ResourceManager(ResourceConfig.EIGHT_GPU_64CPU)
        assert rm.cuda_devices == "0,1,2,3,4,5,6,7"

    def test_gpu_count_property(self):
        rm = ResourceManager(ResourceConfig.ONE_GPU_8CPU)
        assert rm.gpu_count == 1

        rm = ResourceManager(ResourceConfig.TWO_GPU_16CPU)
        assert rm.gpu_count == 2

        rm = ResourceManager(ResourceConfig.FOUR_GPU_32CPU)
        assert rm.gpu_count == 4

        rm = ResourceManager(ResourceConfig.EIGHT_GPU_64CPU)
        assert rm.gpu_count == 8

    def test_cpu_count_property(self):
        rm = ResourceManager(ResourceConfig.ONE_GPU_8CPU)
        assert rm.cpu_count == 8

        rm = ResourceManager(ResourceConfig.ONE_GPU_16CPU)
        assert rm.cpu_count == 16

        rm = ResourceManager(ResourceConfig.FOUR_GPU_32CPU)
        assert rm.cpu_count == 32

        rm = ResourceManager(ResourceConfig.EIGHT_GPU_64CPU)
        assert rm.cpu_count == 64

    def test_describe(self):
        rm = ResourceManager(ResourceConfig.ONE_GPU_8CPU)
        desc = rm.describe()
        assert "1 GPU" in desc
        assert "8 CPU" in desc
        assert "[0]" in desc  # CUDA devices

        rm = ResourceManager(ResourceConfig.FOUR_GPU_32CPU)
        desc = rm.describe()
        assert "4 GPU" in desc
        assert "32 CPU" in desc
        assert "[0,1,2,3]" in desc

    def test_exception_in_context_still_restores(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "original"

        try:
            with pytest.raises(ValueError):
                with ResourceManager(ResourceConfig.ONE_GPU_8CPU):
                    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
                    raise ValueError("Test exception")

            # Verify restoration happened despite exception
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "original"
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_all_resource_configs_work(self):
        """Test that all ResourceConfig values work with ResourceManager."""
        for rc in ResourceConfig:
            rm = ResourceManager(rc)
            # Should not raise
            assert rm.gpu_count >= 1
            assert rm.cpu_count >= 8
            assert rm.cuda_devices is not None
