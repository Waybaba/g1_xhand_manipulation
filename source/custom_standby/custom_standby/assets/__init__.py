"""Asset helpers for the custom locomotion package."""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))


def _resolve_asset_dir(asset_dir: str | Path | None = None) -> Path:
    """Resolve the package asset directory.

    Args:
        asset_dir (str | Path | None): Optional asset directory override.

    Returns:
        Path: Resolved asset directory path.
    """
    if asset_dir is None:
        return Path(ASSET_DIR)
    return Path(asset_dir)


def _rewrite_package_uris(urdf_text: str, package_roots: dict[str, Path]) -> str:
    """Rewrite `package://...` URIs to absolute paths.

    Args:
        urdf_text (str): URDF file contents.
        package_roots (dict[str, Path]): Mapping from ROS package names to local roots.

    Returns:
        str: URDF text with package URIs replaced by absolute paths.
    """
    for package_name, package_root in package_roots.items():
        urdf_text = urdf_text.replace(f"package://{package_name}", str(package_root))
    return urdf_text


def _sanitize_urdf_inertials(urdf_text: str) -> str:
    """Raise tiny inertial values to PhysX-safe floors.

    The packaged xhand URDF contains many helper links with near-zero masses and
    inertias. These are valid enough for some tools but numerically unstable for
    PhysX, which can reject them during import. We floor tiny values and zero the
    products of inertia on these near-massless links in the resolved URDF only.

    Args:
        urdf_text (str): URDF file contents.

    Returns:
        str: URDF text with sanitized inertial parameters.
    """
    min_mass = 1.0e-4
    min_diag_inertia = 1.0e-8
    tiny_link_mass_threshold = 1.0e-4

    root = ET.fromstring(urdf_text)
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        if mass is None or inertia is None:
            continue

        mass_value = float(mass.attrib["value"])
        if mass_value < min_mass:
            mass.attrib["value"] = f"{min_mass:.8g}"
            mass_value = min_mass

        # Reason: very small helper links from the xhand model tend to have
        # numerically degenerate inertias that PhysX rejects during import.
        for axis in ("ixx", "iyy", "izz"):
            axis_value = float(inertia.attrib[axis])
            if axis_value < min_diag_inertia:
                inertia.attrib[axis] = f"{min_diag_inertia:.8g}"

        if mass_value <= tiny_link_mass_threshold:
            for axis in ("ixy", "ixz", "iyz"):
                inertia.attrib[axis] = "0"

    return ET.tostring(root, encoding="unicode")


_LOCOMOTION_COLLISION_LINKS = {
    "pelvis",
    "torso_link",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "LL_FOOT", "LR_FOOT",
}


def _strip_nonlocomotion_collision(urdf_text: str) -> str:
    """Remove collision geometry from all links except those needed for locomotion.

    For locomotion training only feet and torso/pelvis/leg links need collision.
    Keeping collision on arms, wrists, and hands causes PhysX contact explosions
    when the robot falls on rough terrain, corrupting the entire articulation state.
    Mass/inertia is preserved so the robot's dynamics (center of mass, weight) remain
    physically accurate.

    Args:
        urdf_text (str): URDF file contents.

    Returns:
        str: URDF text with collision elements removed from non-locomotion links.
    """
    root = ET.fromstring(urdf_text)
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if name in _LOCOMOTION_COLLISION_LINKS:
            continue
        for collision in link.findall("collision"):
            link.remove(collision)
    return ET.tostring(root, encoding="unicode")


def _write_resolved_urdf(source_urdf: Path, resolved_urdf: Path, package_roots: dict[str, Path]) -> str:
    """Write a URDF copy with package URIs rewritten to local absolute paths.

    Args:
        source_urdf (Path): Source URDF to rewrite.
        resolved_urdf (Path): Output URDF path.
        package_roots (dict[str, Path]): Mapping from ROS package names to local roots.

    Returns:
        str: Absolute path to the rewritten URDF file.
    """
    text = source_urdf.read_text()
    text = _rewrite_package_uris(text, package_roots)
    text = _sanitize_urdf_inertials(text)
    text = _strip_nonlocomotion_collision(text)
    resolved_urdf.write_text(text)
    return str(resolved_urdf)


def resolve_g1_usd_asset_path(asset_dir: str | Path | None = None) -> str | None:
    """Resolve a bundled USD asset path if one exists.

    Args:
        asset_dir (str | Path | None): Optional asset directory override.

    Returns:
        str | None: Absolute USD path if found, otherwise `None`.
    """
    asset_root = _resolve_asset_dir(asset_dir)
    candidates = [
        asset_root / "g1_xhand_description" / "usd" / "g1_xhand.usd",
        asset_root / "g1_xhand_description" / "usd" / "g1_xhand.usda",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def resolve_g1_urdf_asset_path(asset_dir: str | Path | None = None, workspace_root: str | Path | None = None) -> str:
    """Resolve a usable G1 URDF path for Isaac Lab.

    Only the packaged asset layout is supported so this training repo stays
    self-contained and does not silently depend on the parent workspace.

    Args:
        asset_dir (str | Path | None): Optional asset directory override.
        workspace_root (str | Path | None): Unused legacy argument kept for compatibility.

    Returns:
        str: Absolute path to a URDF file that exists on disk.

    Raises:
        FileNotFoundError: If a bundled URDF cannot be found.
    """
    asset_root = _resolve_asset_dir(asset_dir)
    bundled_xhand_urdf = asset_root / "g1_xhand_description" / "urdf" / "g1_xhand" / "main.urdf"
    bundled_unitree_urdf = asset_root / "unitree_description" / "urdf" / "g1" / "main.urdf"

    if bundled_xhand_urdf.exists():
        return _write_resolved_urdf(
            bundled_xhand_urdf,
            asset_root / "_resolved_g1_xhand.urdf",
            {
                "unitree_description": asset_root / "unitree_description",
                "g1_xhand_description": asset_root / "g1_xhand_description",
                "xhand_left": asset_root / "xhand_left",
                "xhand_right": asset_root / "xhand_right",
            },
        )

    if bundled_unitree_urdf.exists():
        return _write_resolved_urdf(
            bundled_unitree_urdf,
            asset_root / "_resolved_g1.urdf",
            {"unitree_description": asset_root / "unitree_description"},
        )

    raise FileNotFoundError(
        "Could not find a bundled URDF asset. Checked: "
        f"{bundled_xhand_urdf} and {bundled_unitree_urdf}"
    )
