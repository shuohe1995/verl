# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Recipe reward_score: only supports data_source "math_kpo". Use verl.utils.reward_score for others.

from verl.utils.import_utils import deprecated

from .kpo_math_reward import RewardConfig, RewardMathFn


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for math_kpo. Only data_source \"math_kpo\" is supported.

    Args:
        data_source (str): Must be \"math_kpo\".
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information for scoring (e.g. reward params). Defaults to None.

    Returns:
        float: The computed score.

    Raises:
        NotImplementedError: If data_source is not \"math_kpo\".
    """
    if data_source != "math_kpo":
        raise NotImplementedError(
            f"Recipe reward_score only supports data_source='math_kpo'. Got {data_source=}. "
            "Use verl.utils.reward_score.default_compute_score for other data sources."
        )

    config_kwargs = {}
    if extra_info and isinstance(extra_info, dict):
        config_params = [
            "apply_format_reward",
            "correct_reward",
            "incorrect_reward",
            "format_error_reward",
            "unk_error_reward",
            "toolcall_bonus",
        ]
        for param in config_params:
            if param in extra_info:
                config_kwargs[param] = extra_info[param]

    for param in [
        "apply_format_reward",
        "correct_reward",
        "incorrect_reward",
        "format_error_reward",
        "unk_error_reward",
        "toolcall_bonus",
    ]:
        if param in kwargs:
            config_kwargs[param] = kwargs[param]

    config = RewardConfig(**config_kwargs)
    reward_fn = RewardMathFn(config)

    task_info = {
        "problem": kwargs.get("problem", ""),
        "ground_truth": ground_truth,
        "data_source": data_source,
        "problem_type": kwargs.get("problem_type", "math"),
        "has_toolcall": kwargs.get("has_toolcall", False),
    }
    if extra_info and isinstance(extra_info, dict):
        task_info.update(extra_info)

    res, _ = reward_fn(task_info, solution_str)

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
