import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from custom_standby.utils.exporter import attach_onnx_metadata


class MyOnPolicyRunner(OnPolicyRunner):
    """Runner that exports an ONNX policy locally on checkpoint save."""

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        normalizer = getattr(self.alg.policy, "actor_obs_normalizer", None)
        if normalizer is None:
            normalizer = getattr(self, "obs_normalizer", None)
        export_policy_as_onnx(self.alg.policy, normalizer=normalizer, path=policy_path, filename=filename)
        attach_onnx_metadata(self.env.unwrapped, path, path=policy_path, filename=filename)
