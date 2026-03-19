import os

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from custom_standby.utils.exporter import attach_onnx_metadata


class MyOnPolicyRunner(OnPolicyRunner):
    """Runner that exports an ONNX policy locally on checkpoint save."""

    def save(self, path: str, infos=None):
        """Save the model and export ONNX with metadata."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        self.export_policy_to_onnx(policy_path, filename=filename)
        attach_onnx_metadata(self.env.unwrapped, path, path=policy_path, filename=filename)
