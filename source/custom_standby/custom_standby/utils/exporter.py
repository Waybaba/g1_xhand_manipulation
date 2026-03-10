"""Helpers for local ONNX export and metadata attachment.

The training env config uses natural IsaacLab names (e.g.
``base_velocity``, ``velocity_commands``) and no observation scales.
The ``XhandManipController`` handles any name translation internally,
so the exporter passes names through as-is.
"""

import os
from typing import Sequence

import onnx

from isaaclab.envs import ManagerBasedRLEnv


def list_to_csv_str(arr: Sequence, *, decimals: int = 3, delimiter: str = ",") -> str:
    """Convert a list into a compact CSV string.

    Args:
        arr: Values to serialize.
        decimals: Number of decimals for floats.
        delimiter: Delimiter between values.

    Returns:
        str: Serialized CSV string.
    """
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x)
        for x in arr
    )


def attach_onnx_metadata(
    env: ManagerBasedRLEnv,
    run_path: str,
    path: str,
    filename: str = "policy.onnx",
) -> None:
    """Attach deployment metadata to an exported ONNX file.

    Names are written exactly as IsaacLab defines them.  The
    ``XhandManipController`` maps them to the base-class identifiers
    in its ``parserCommand`` / ``parserObservation`` overrides.

    Args:
        env: The Isaac Lab environment.
        run_path: Training run identifier or local checkpoint path.
        path: Directory containing the ONNX file.
        filename: Name of the ONNX file.
    """
    onnx_path = os.path.join(path, filename)

    observation_names: list[str] = env.observation_manager.active_terms["policy"]
    command_names: list[str] = env.command_manager.active_terms

    observation_history_lengths: list[int] = []
    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = (
            [env.observation_manager.cfg.policy.history_length] * len(observation_names)
        )
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    default_joint_pos = getattr(env.scene["robot"].data, "default_joint_pos_nominal", None)
    if default_joint_pos is None:
        default_joint_pos = env.scene["robot"].data.default_joint_pos[0]

    action_term = env.action_manager.get_term("joint_pos")
    action_joint_names = list(action_term._joint_names)
    action_joint_ids = list(action_term._joint_ids)

    # Reason: store per-action-joint stiffness/damping/default_pos so
    # the deployer doesn't need to cross-reference with the full 53-joint list.
    all_stiffness = env.scene["robot"].data.joint_stiffness[0].cpu()
    all_damping = env.scene["robot"].data.joint_damping[0].cpu()
    all_default = default_joint_pos.cpu()

    metadata = {
        "run_path": run_path,
        "joint_names": action_joint_names,
        "joint_stiffness": [all_stiffness[i].item() for i in action_joint_ids],
        "joint_damping": [all_damping[i].item() for i in action_joint_ids],
        "default_joint_pos": [all_default[i].item() for i in action_joint_ids],
        "command_names": command_names,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": action_term._scale[0].cpu().tolist(),
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
