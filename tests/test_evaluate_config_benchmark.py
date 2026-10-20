import pytest

from lm_eval.config.evaluate_config import EvaluatorConfig, _classify_benchmark
from lm_eval.tasks import TaskManager


@pytest.fixture(scope="module")
def task_manager():
    return TaskManager()


def test_group_with_aggregate_metric_list_is_group_aggregate(task_manager):
    spec = _classify_benchmark("wmdp", task_manager.match_tasks(["wmdp"]), task_manager)

    assert spec.strategy == "group_aggregate"
    assert spec.leaf_task_names is None


def test_group_without_aggregate_metric_list_is_group_fallback(task_manager):
    spec = _classify_benchmark(
        "infinitebench", task_manager.match_tasks(["infinitebench"]), task_manager
    )

    assert spec.strategy == "group_fallback"
    assert spec.leaf_task_names is None


def test_tag_is_tag_fallback_with_full_leaf_set(task_manager):
    spec = _classify_benchmark(
        "model_written_evals",
        task_manager.match_tasks(["model_written_evals"]),
        task_manager,
    )

    assert spec.strategy == "tag_fallback"
    assert spec.leaf_task_names is not None
    assert len(spec.leaf_task_names) == 187


def test_standalone_task_is_standalone(task_manager):
    spec = _classify_benchmark("race", task_manager.match_tasks(["race"]), task_manager)

    assert spec.strategy == "standalone"
    assert spec.leaf_task_names == ("race",)


def test_glob_matching_a_group_raises(task_manager):
    with pytest.raises(ValueError, match="not a leaf task"):
        _classify_benchmark("wmdp*", ["wmdp", "wmdp_bio"], task_manager)


def test_glob_matching_only_leaf_tasks_is_tag_fallback(task_manager):
    spec = _classify_benchmark(
        "wmdp_*", ["wmdp_bio", "wmdp_chem", "wmdp_cyber"], task_manager
    )

    assert spec.strategy == "tag_fallback"
    assert spec.leaf_task_names == ("wmdp_bio", "wmdp_chem", "wmdp_cyber")


def _qwen_config():
    return {
        "schema_version": 1,
        "backend": "qwen-remote-chat",
        "model_name": "Qwen3.5-2B",
        "weight_revision": "Qwen3.5-2B-ctx262144",
        "base_url": "http://localhost:8000/v1/chat/completions",
        "benchmarks": ["race"],
        "output_dir": "results/qwen",
        "max_length": 32768,
        "max_num_seqs": 4,
        "qwen_profile": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        },
    }


def test_qwen_versioned_config_preserves_revision_and_sampling():
    config = EvaluatorConfig._normalize_versioned_config(_qwen_config())
    assert config["model"] == "qwen-remote-chat"
    assert config["model_args"]["enable_thinking"] is False
    assert config["model_args"]["num_concurrent"] == 4
    assert config["metadata"]["weight_revision"] == "Qwen3.5-2B-ctx262144"
    assert "wkv_mode" not in config["metadata"]
    assert "rwkv_generation_prompt" not in config["model_args"]


def test_qwen_rejects_mislabeled_thinking_mode():
    raw = _qwen_config()
    raw["publication"] = {"enabled": True, "cot_mode": "CoT"}
    with pytest.raises(ValueError, match="cot_mode"):
        EvaluatorConfig._normalize_versioned_config(raw)


def test_qwen_tokenizer_is_available_to_synthetic_tasks():
    raw = _qwen_config()
    raw["tokenizer_name"] = "Qwen/Qwen3.5-2B"
    config = EvaluatorConfig._normalize_versioned_config(raw)
    assert config["metadata"]["tokenizer"] == "Qwen/Qwen3.5-2B"
    assert "tokenizer" not in config["model_args"]


def test_mixed_qwen_rwkv_campaign_uses_each_models_profile():
    raw = _qwen_config()
    raw.pop("model_name")
    raw.pop("qwen_profile")
    raw.pop("weight_revision")
    raw["models"] = [
        {
            "model_name": "Qwen3.5-2B",
            "qwen_profile": _qwen_config()["qwen_profile"],
            "weight_revision": "Qwen3.5-2B-ctx262144",
            "output_dir": "results/qwen",
        },
        {
            "model_name": "rwkv7-g1j-1.5b",
            "backend": "rwkv7-remote-chat",
            "rwkv_profile": {
                "prompt_template": "assistant",
                "generation_prompt": "fake_think",
                "wkv_mode": "fp32io16",
            },
            "output_dir": "results/rwkv",
        },
    ]
    configs = EvaluatorConfig._expand_models(raw)
    assert [config["model"] for config in configs] == [
        "qwen-remote-chat",
        "rwkv7-remote-chat",
    ]
