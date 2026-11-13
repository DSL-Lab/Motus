"""Fixed real windows used by the existing RoboTwin loader for matched A/C."""
import json
import hashlib
from pathlib import Path

import numpy as np

from .adaptive_spline import (extract_local, read_cache, decode_local, select_future_frames, identity)


def load_windows(path, val, contract):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "motus-fixed-real-windows-v1":
        raise ValueError("Expected explicit fixed real-observation window manifest")
    windows = [w for w in manifest["windows"] if w["split"] == ("validation" if val else "train")]
    if not windows:
        raise ValueError("Window manifest has no requested split")
    expected = contract["normalization"]
    train_keys = {(s["path"], s["action_sha256"]) for s in expected["training_sources"]}
    for window in windows:
        for key in ("qpos_path", "video_path", "lang_path", "cache_path", "timeline_path"):
            if not Path(window[key]).is_file():
                raise FileNotFoundError(window[key])
        if Path(window["qpos_path"]).suffix != ".pt":
            raise ValueError("Real research windows must use audited original MOTUS PT data, not synthetic NPY fixtures")
        actions, spline, meta = read_cache(window["cache_path"])
        if not meta["diagnostics"]["success"] or meta["normalization"] != expected:
            raise ValueError("Failed fit or inconsistent training normalization")
        if hashlib.sha256(actions.tobytes()).hexdigest() != meta['source']['action_sha256']:
            raise ValueError('Cached commands do not match their source identity')
        from .adaptive_spline import normalize
        tolerance = meta['diagnostics']['settings']['tolerance']
        if tolerance > .01 or np.abs(spline(np.arange(len(actions))) - normalize(actions, expected)).max() > tolerance:
            raise ValueError('Stage-1 reconstruction check failed on this episode')
        key = (meta["source"]["path"], meta["source"]["action_sha256"])
        if (key in train_keys) == val:
            raise ValueError("Episode leaks across the fixed training/validation split")
        if val and meta["source"]["path"] not in expected["validation_paths"]:
            raise ValueError("Unknown validation episode")
        timeline = json.loads(Path(window["timeline_path"]).read_text())
        if timeline.get("status") != "verified" or timeline.get("units") != "seconds" or timeline.get("action_sha256") != meta["source"]["action_sha256"]:
            raise ValueError("Unverified or mismatched action/camera clock")
        if len(timeline["action_times"]) != len(actions):
            raise ValueError("Action clock length mismatch")
        obs = window["observation_index"]
        if not isinstance(obs, int) or not 0 <= obs < len(actions):
            raise ValueError("Observation must be an original source-row integer index")
        packet = extract_local(spline, obs, contract["knot_scaling"])
        # Check the actual packed target against its global parent before it can
        # reach training. The original command grid remains stride three.
        dense_indices = np.minimum(obs + 3 * np.arange(1, 17), len(actions) - 1)
        target_decoded, _ = decode_local(packet["parameters"], dense_indices, obs, contract["knot_scaling"], expected)
        from .adaptive_spline import decode
        if not np.allclose(target_decoded, decode(spline(dense_indices), expected), atol=2e-5, rtol=0):
            raise ValueError("Local/global packed-target round trip failed")
        selection = select_future_frames(packet, timeline["action_times"], timeline["camera_times"])
        if selection['current_camera_timestamp'] > selection['current_timestamp']:
            raise ValueError('Nearest current camera frame lies in the future; select a causal observation window')
        if not any(selection["proposed_latent_loss_mask"]):
            raise ValueError("Window has no valid future latent under conservative causal VAE mask")
        # Cache only the explicit target and clock metadata, never image arrays.
        window["_packet"], window["_selection"] = packet, selection
        window["_dense_indices"], window["_actions"] = dense_indices, actions
        window["_meta"], window["_timeline"] = meta, timeline
        window["identity"] = identity({k: v for k, v in window.items() if not k.startswith("_") and k != "identity"})
    return windows


def window_targets(window, representation):
    obs, actions = window["observation_index"], window["_actions"]
    selection, timeline = window["_selection"], window["_timeline"]
    if representation == "spline":
        target = window["_packet"]["parameters"]
        video_indices = [f["camera_index"] for f in selection["frames"]]
        latent_valid = selection["conservative_latent_valid"]
    elif representation == "dense":
        target = actions[window["_dense_indices"]].astype(np.float32)
        # Original dense target/video grid: every second action, beginning at
        # the second future action. Recording row->camera mapping is explicit.
        video_indices = [int(np.argmin(np.abs(np.asarray(timeline["camera_times"]) - timeline["action_times"][i])))
                         for i in window["_dense_indices"][1::2]]
        latent_valid = [True, True, True]  # preserve dense baseline tail behavior
    else:
        raise ValueError("Unknown representation")
    return dict(observation_camera_index=selection["current_camera_index"], video_indices=video_indices,
                initial_state=actions[obs].astype(np.float32), actions=target,
                latent_valid=latent_valid, identity=window["identity"])
