"""Tests for fused operator categories."""

import pytest

from inference_simulator.types.operators import OperatorCategory, OperatorMeasurement


class TestFusedOperatorCategories:
    def test_fused_prefill_exists(self):
        assert OperatorCategory.FUSED_PREFILL.value == "fused_prefill"

    def test_fused_decode_step_exists(self):
        assert OperatorCategory.FUSED_DECODE_STEP.value == "fused_decode_step"

    def test_fused_attention_exists(self):
        assert OperatorCategory.FUSED_ATTENTION.value == "fused_attention"

    def test_fused_mlp_exists(self):
        assert OperatorCategory.FUSED_MLP.value == "fused_mlp"

    def test_fused_norm_attn_exists(self):
        assert OperatorCategory.FUSED_NORM_ATTN.value == "fused_norm_attn"

    def test_all_five_fused_categories(self):
        fused = [c for c in OperatorCategory if c.value.startswith("fused_")]
        assert len(fused) == 5

    def test_original_categories_still_exist(self):
        """Fused categories don't break original ones."""
        assert OperatorCategory.LINEAR.value == "linear"
        assert OperatorCategory.ATTENTION_PREFILL.value == "attention_prefill"
        assert OperatorCategory.ATTENTION_DECODE.value == "attention_decode"
        assert OperatorCategory.EMBEDDING.value == "embedding"

    def test_fused_measurement(self):
        """Can create measurements with fused categories."""
        m = OperatorMeasurement(
            operator_name="fused_prefill_op",
            category=OperatorCategory.FUSED_PREFILL,
            batch_size=4,
            seq_len=1024,
            time_s=0.01,
            energy_j=2.0,
            power_w=200.0,
        )
        assert m.category == OperatorCategory.FUSED_PREFILL
        assert m.time_s == 0.01

    def test_category_from_string(self):
        """Can construct fused categories from string values."""
        cat = OperatorCategory("fused_prefill")
        assert cat == OperatorCategory.FUSED_PREFILL
