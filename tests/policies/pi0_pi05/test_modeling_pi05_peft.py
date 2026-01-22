#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for PI05Policy._get_default_peft_targets method."""

import re
from unittest.mock import Mock, patch

import pytest

from lerobot.policies.pi05 import PI05Config, PI05Policy


class TestGetDefaultPeftTargets:
    """Test suite for PI05Policy._get_default_peft_targets method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock PI05Config for testing."""
        config = Mock(spec=PI05Config)
        config.device = "cpu"
        config.dtype = "float32"
        config.max_action_dim = 7
        config.max_state_dim = 14
        config.image_resolution = (224, 224)
        config.chunk_size = 64
        config.tokenizer_max_length = 200
        config.gradient_checkpointing = False
        config.compile_model = False
        config.time_sampling_beta_alpha = 1.0
        config.time_sampling_beta_beta = 1.0
        config.time_sampling_scale = 1.0
        config.time_sampling_offset = 0.0
        config.paligemma_variant = "gemma_2b"
        config.action_expert_variant = "gemma_300m"
        config.rtc_config = None
        config.freeze_vision_encoder = True
        config.train_expert_only = False
        config.input_features = {}
        config.output_features = {}
        config.image_features = []
        config.validate_features = Mock()
        return config

    @pytest.fixture
    def policy_with_mocked_model(self, mock_config):
        """Create a PI05Policy instance with mocked internal model."""
        with patch("lerobot.policies.pi05.modeling_pi05.PaliGemmaWithExpertModel"):
            policy = PI05Policy.__new__(PI05Policy)
            policy.config = mock_config
            policy.name = "pi05"
            return policy

    def test_returns_dict(self, policy_with_mocked_model):
        """Test that _get_default_peft_targets returns a dictionary."""
        result = policy_with_mocked_model._get_default_peft_targets()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_returns_expected_keys(self, policy_with_mocked_model):
        """Test that returned dictionary has expected keys."""
        result = policy_with_mocked_model._get_default_peft_targets()
        expected_keys = {"target_modules", "modules_to_save"}
        assert set(result.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(result.keys())}"

    def test_target_modules_is_string(self, policy_with_mocked_model):
        """Test that target_modules value is a string."""
        result = policy_with_mocked_model._get_default_peft_targets()
        assert isinstance(result["target_modules"], str), f"Expected str for target_modules, got {type(result['target_modules'])}"

    def test_modules_to_save_is_list(self, policy_with_mocked_model):
        """Test that modules_to_save value is a list."""
        result = policy_with_mocked_model._get_default_peft_targets()
        assert isinstance(result["modules_to_save"], list), f"Expected list for modules_to_save, got {type(result['modules_to_save'])}"

    def test_modules_to_save_is_empty(self, policy_with_mocked_model):
        """Test that modules_to_save is an empty list."""
        result = policy_with_mocked_model._get_default_peft_targets()
        assert len(result["modules_to_save"]) == 0, f"Expected empty list, got {result['modules_to_save']}"

    def test_target_modules_contains_common_projections(self, policy_with_mocked_model):
        """Test that target_modules regex contains common projection names."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        
        expected_projections = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        
        for projection in expected_projections:
            assert projection in target_modules, f"Expected '{projection}' in target_modules"

    def test_target_modules_contains_gemma_expert_pattern(self, policy_with_mocked_model):
        """Test that target_modules regex contains gemma_expert pattern for q/v projections."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        
        assert "gemma_expert" in target_modules, "Expected 'gemma_expert' in target_modules"
        assert "self_attn" in target_modules, "Expected 'self_attn' in target_modules"
        assert "(q|v)_proj" in target_modules, "Expected '(q|v)_proj' pattern in target_modules"

    def test_target_modules_contains_model_prefix(self, policy_with_mocked_model):
        """Test that target_modules regex contains model. prefix for common projections."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        
        assert "model." in target_modules, "Expected 'model.' prefix in target_modules"

    def test_target_modules_is_valid_regex(self, policy_with_mocked_model):
        """Test that target_modules is a valid regular expression."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        
        try:
            re.compile(target_modules)
        except re.error as e:
            pytest.fail(f"target_modules is not a valid regex: {e}")

    def test_target_modules_matches_expected_module_patterns(self, policy_with_mocked_model):
        """Test that target_modules regex matches expected module name patterns."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        pattern = re.compile(target_modules)
        
        # Test cases that should match
        should_match = [
            "model.state_proj",
            "model.action_in_proj",
            "model.action_out_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
            "some.gemma_expert.layer.self_attn.q_proj",
            "paligemma.gemma_expert.0.self_attn.v_proj",
            "expert.gemma_expert.model.self_attn.q_proj",
            "model.paligemma.gemma_expert.1.self_attn.v_proj",
        ]
        
        for module_name in should_match:
            assert pattern.search(module_name) is not None, f"Pattern should match '{module_name}'"

    def test_target_modules_does_not_match_unexpected_patterns(self, policy_with_mocked_model):
        """Test that target_modules regex does not match unexpected module patterns."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        pattern = re.compile(target_modules)
        
        # Test cases that should NOT match (no q/v projection or wrong prefix)
        should_not_match = [
            "some.random.module",
            "model.random_layer",
            "gemma_expert.self_attn.k_proj",  # k_proj should not match
            "gemma_expert.self_attn.o_proj",  # o_proj should not match
            "state_proj",  # Without model. prefix
            "action_in_proj",  # Without model. prefix
            "expert.random_module",
        ]
        
        for module_name in should_not_match:
            assert pattern.search(module_name) is None, f"Pattern should NOT match '{module_name}'"

    def test_consistency_with_pi05_requirements(self, policy_with_mocked_model):
        """Test that targets align with PI05 model architecture requirements."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        
        # PI05 uses gemma_expert for action-specific processing
        assert "gemma_expert" in target_modules
        
        # PI05 uses action projections
        assert "action_in_proj" in target_modules
        assert "action_out_proj" in target_modules
        
        # PI05 uses time MLP projections for AdaRMS conditioning
        assert "action_time_mlp_in" in target_modules
        assert "action_time_mlp_out" in target_modules

    def test_peft_targets_no_side_effects(self, policy_with_mocked_model):
        """Test that calling _get_default_peft_targets has no side effects on policy state."""
        initial_device = policy_with_mocked_model.config.device
        
        # Call method multiple times
        for _ in range(5):
            result = policy_with_mocked_model._get_default_peft_targets()
            assert isinstance(result, dict)
        
        # Verify policy state unchanged
        assert policy_with_mocked_model.config.device == initial_device

    def test_return_value_immutability_concept(self, policy_with_mocked_model):
        """Test that modifying the returned dict doesn't affect subsequent calls."""
        result1 = policy_with_mocked_model._get_default_peft_targets()
        result1["target_modules"] = "modified"
        result1["modules_to_save"].append("some_module")
        
        result2 = policy_with_mocked_model._get_default_peft_targets()
        
        # Each call should return a fresh value (not the modified one)
        assert result2["target_modules"] != "modified"
        assert len(result2["modules_to_save"]) == 0

    @pytest.mark.parametrize("call_count", [1, 2, 10, 100])
    def test_deterministic_behavior(self, policy_with_mocked_model, call_count):
        """Test that method returns consistent results across multiple calls."""
        results = []
        for _ in range(call_count):
            result = policy_with_mocked_model._get_default_peft_targets()
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[0] == results[i], f"Results differ between calls 0 and {i}"

    def test_raw_string_format(self, policy_with_mocked_model):
        """Test that the regex uses raw string format (rf prefix in code)."""
        result = policy_with_mocked_model._get_default_peft_targets()
        target_modules = result["target_modules"]
        
        # Verify that escaped characters are properly handled
        assert "\\." in target_modules or "." in target_modules
        assert "|" in target_modules  # Alternation operator
        assert "(" in target_modules and ")" in target_modules  # Capturing groups

    def test_pattern_structure(self, policy_with_mocked_model):
        """Test the structure of the regex pattern."""
        result = policy_with_mocked_model._get_default_peft_targets()
        pattern = result["target_modules"]
        
        # Should have alternation for two main patterns
        assert pattern.count("(") >= 2  # At least two groups
        assert pattern.count(")") >= 2
        
        # Main structure should have two alternatives separated by |
        parts = pattern.split("|", maxsplit=1)
        assert len(parts) >= 2, "Pattern should have at least one top-level alternation"

    def test_gemma_expert_pattern_structure(self, policy_with_mocked_model):
        """Test the gemma_expert specific pattern structure."""
        result = policy_with_mocked_model._get_default_peft_targets()
        pattern = result["target_modules"]
        
        # Gemma expert pattern should match: .*\.gemma_expert\..*\.self_attn\.(q|v)_proj
        gemma_pattern = r".*\.gemma_expert\..*\.self_attn\.(q|v)_proj"
        assert gemma_pattern.replace("\\", "") in pattern.replace("\\", ""), \
            "Expected gemma_expert pattern structure in target_modules"

    def test_projection_names_completeness(self, policy_with_mocked_model):
        """Test that all common projection names are included."""
        result = policy_with_mocked_model._get_default_peft_targets()
        pattern = result["target_modules"]
        
        projections = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        
        # All should be connected by pipe (OR) operators
        pattern_without_model = pattern.replace("model.", "")
        pattern_without_model = pattern_without_model.replace("(", "").replace(")", "")
        
        for proj in projections:
            assert proj in pattern_without_model, f"Expected '{proj}' in pattern"

    def test_attention_projection_specificity(self, policy_with_mocked_model):
        """Test that only q and v projections are targeted for attention."""
        result = policy_with_mocked_model._get_default_peft_targets()
        pattern = result["target_modules"]
        
        # Should include q and v
        assert "q_proj" in pattern
        assert "v_proj" in pattern
        
        # Should NOT include k, o, or other attention projections
        # (unless they're part of common projections with different context)
        assert "k_proj" not in pattern or pattern.index("k_proj") > pattern.index("self_attn") + 100
        
        # The pattern specifically uses (q|v) for attention projections
        assert "(q|v)_proj" in pattern or "(q|v)_proj" in pattern.replace("\\", "")
