"""TCP reconstruction diagnostics; never changes fitting or simulator state."""

import hashlib
from pathlib import Path

import numpy as np

from .robotwin import ARM_CHANNELS
from .spline import evaluate_spline

ARMS = ("left", "right")


class RobotTCPKinematics:
    """Use each live articulation's SAPIEN Pinocchio model without setting qpos.

    RoboTwin uses Joint.global_pose, not the link frame. In SAPIEN this is
    child_link.pose * joint.pose_in_child. Keep that transform as well as the
    gripper-centre offset in Robot._trans_endpose(is_endpose=True).
    """

    def __init__(self, robot):
        self.models = []
        self.metadata = {"backend": "SAPIEN Pinocchio", "arms": {}}
        for arm in ARMS:
            entity = getattr(robot, f"{arm}_entity")
            joint = getattr(robot, f"{arm}_ee")
            names = list(getattr(robot, f"{arm}_arm_joints_name"))
            if len(names) != 6:
                raise ValueError("TCP metrics require six ordered joints per arm")
            indices = {}
            offset = 0
            for active in entity.get_active_joints():
                if active.dof == 1:
                    indices[active.name] = offset
                offset += active.dof
            order = [indices[name] for name in names]
            global_rotation = np.asarray(getattr(robot, f"{arm}_global_trans_matrix"))
            delta_rotation = np.asarray(getattr(robot, f"{arm}_delta_matrix"))
            bias = float(getattr(robot, f"{arm}_gripper_bias"))
            in_child = (
                joint.get_pose_in_child().to_transformation_matrix().astype(float)
            )
            joint_offset = global_rotation @ delta_rotation @ np.array([bias, 0, 0])
            tcp_in_child = in_child[:3, 3] + in_child[:3, :3] @ joint_offset
            model = entity.create_pinocchio_model()
            link_index = entity.get_links().index(joint.child_link)
            self.models.append((entity, model, order, link_index, tcp_in_child))
            urdf = Path(getattr(robot, f"{arm}_urdf_path"))
            self.metadata["arms"][arm] = {
                "joint_names": names,
                "urdf": str(urdf),
                "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                "ee_joint": joint.name,
                "ee_child_link": joint.child_link.name,
                "joint_pose_in_child": in_child.tolist(),
                "global_trans_matrix": global_rotation.tolist(),
                "delta_matrix": delta_rotation.tolist(),
                "gripper_bias_m": bias,
                "tcp_in_child_m": tcp_in_child.tolist(),
            }

    def positions(self, arm_qpos):
        """[N,12] in L1..L6,R1..R6 order -> [N,2,3] positions in metres."""
        values = np.asarray(arm_qpos, dtype=float)
        if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
            raise ValueError("Expected finite arm targets [N,12]")
        result = np.empty((len(values), 2, 3))
        for arm_index, (entity, model, order, link_index, tcp) in enumerate(
            self.models
        ):
            # Other joints retain measured values; no articulation state is written.
            qpos = np.array(entity.get_qpos(), dtype=float, copy=True)
            root = entity.get_root_pose().to_transformation_matrix().astype(float)
            for i, value in enumerate(values):
                qpos[order] = value[6 * arm_index : 6 * (arm_index + 1)]
                model.compute_forward_kinematics(qpos)
                link = (
                    model.get_link_pose(link_index)
                    .to_transformation_matrix()
                    .astype(float)
                )
                transform = root @ link
                result[i, arm_index] = transform[:3, 3] + transform[:3, :3] @ tcp
        return result

    def verify_live_tcp(self, robot, atol_m=3e-6):
        """Check the current robot's own TCP getters before logging diagnostics."""
        values = np.array(
            robot.get_left_arm_jointState()[:6] + robot.get_right_arm_jointState()[:6]
        )[None, :]
        expected = np.array(
            [robot.get_left_tcp_pose()[:3], robot.get_right_tcp_pose()[:3]]
        )
        difference = np.linalg.norm(self.positions(values)[0] - expected, axis=1)
        if np.any(difference > atol_m):
            raise ValueError(
                f"FK disagrees with RoboTwin TCP getters: {difference} metres"
            )
        return dict(zip(ARMS, difference.tolist(), strict=True))


def position_statistics(candidate, reference):
    """Euclidean position errors, with sufficient statistics for sample pooling."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if (
        candidate.shape != reference.shape
        or candidate.ndim != 3
        or candidate.shape[1:] != (2, 3)
    ):
        raise ValueError("Expected matching TCP positions [N,2,3]")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite TCP positions")
    squared_mm = np.sum(((candidate - reference) * 1000.0) ** 2, axis=2)
    return {
        arm: {
            "sample_count": len(squared_mm),
            "sum_squared_error_mm2": float(squared_mm[:, i].sum()),
            "position_rmse_mm": float(np.sqrt(squared_mm[:, i].mean()))
            if len(squared_mm)
            else None,
            "max_position_error_mm": float(np.sqrt(squared_mm[:, i].max()))
            if len(squared_mm)
            else None,
        }
        for i, arm in enumerate(ARMS)
    }


def pool_statistics(records):
    """Pool squared Euclidean errors, never per-chunk RMSEs."""
    result = {}
    for arm in ARMS:
        items = [record[arm] for record in records if record is not None]
        count = sum(item["sample_count"] for item in items)
        squared = sum(item["sum_squared_error_mm2"] for item in items)
        maxima = [
            item["max_position_error_mm"] for item in items if item["sample_count"]
        ]
        result[arm] = {
            "sample_count": count,
            "sum_squared_error_mm2": squared,
            "position_rmse_mm": float(np.sqrt(squared / count)) if count else None,
            "max_position_error_mm": max(maxima) if maxima else None,
        }
    return result


def chunk_position_metrics(
    fk,
    actions,
    initial_state,
    source_times,
    query_times,
    commands,
    spline,
    info,
    physics_steps=None,
):
    """Compare only one prediction's spline/dense targets at matching source times.

    Source samples include the measured t=0 fit anchor. Query references use the
    adapter's np.interp joint interpolation, followed by FK (not Cartesian lerp).
    Commands include fallback and no-op queries; physics-positive queries are
    reported separately once per-command step counts are available.
    """
    source = np.asarray(source_times, dtype=float)
    query = np.asarray(query_times, dtype=float)
    samples = np.vstack([initial_state, actions])[:, ARM_CHANNELS]
    dense_query = np.column_stack(
        [np.interp(query, source, samples[:, i]) for i in range(12)]
    )
    reference_query = fk.positions(dense_query)
    dispatched = fk.positions(np.asarray(commands)[:, ARM_CHANNELS])
    fallback = bool(info["fallback_count"])
    requested = spline is not None or fallback
    result = {
        "candidate_status": ("rejected" if fallback else "accepted")
        if requested
        else "not_requested",
        "candidate_unavailable_reason": None
        if spline is not None
        else (info.get("failure") if fallback else "dense adapter; no spline fitted"),
        "candidate_source": None,
        "candidate_query": None,
        "dispatched_query": position_statistics(dispatched, reference_query),
        "candidate_query_physics_positive": None,
        "dispatched_query_physics_positive": None,
        "tracking": {
            "available": False,
            "reason": "Contemporaneous measured TCP/controller references are not recorded",
        },
    }
    candidate_query = None
    if spline is not None:
        result["candidate_source"] = position_statistics(
            fk.positions(evaluate_spline(spline, source)), fk.positions(samples)
        )
        candidate_query = fk.positions(evaluate_spline(spline, query))
        result["candidate_query"] = position_statistics(
            candidate_query, reference_query
        )
    if physics_steps is not None:
        steps = np.asarray(physics_steps)
        if steps.shape != query.shape:
            raise ValueError("Physics-step counts do not cover all queries")
        positive = steps > 0
        result["physics_positive_query_count"] = int(positive.sum())
        result["zero_physics_query_indices"] = np.flatnonzero(steps == 0).tolist()
        result["dispatched_query_physics_positive"] = position_statistics(
            dispatched[positive], reference_query[positive]
        )
        if candidate_query is not None:
            result["candidate_query_physics_positive"] = position_statistics(
                candidate_query[positive], reference_query[positive]
            )
    return result
