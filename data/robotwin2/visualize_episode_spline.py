"""Inspect decoded curves, residuals, shared knots and analytic range overshoot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_spline import decode, normalize, range_diagnostics, read_cache


def visualize(path):
    path = Path(path)
    actions, spline, meta = read_cache(path)
    x = np.arange(len(actions))
    fine = np.linspace(0, len(actions) - 1, max(1000, len(actions) * 8))
    norm = meta["normalization"]
    reconstructed = decode(spline(fine), norm)
    residual = spline(x) - normalize(actions, norm)
    knots = np.unique(spline.t)
    outputs = []
    for name, channels in (("arms", list(range(6)) + list(range(7, 13))), ("grippers", [6, 13])):
        fig, axes = plt.subplots(len(channels), 2, figsize=(15, 2.5 * len(channels)), squeeze=False)
        for row, j in enumerate(channels):
            left, right = axes[row]
            left.plot(x, actions[:, j], color="black", lw=1, label="original commands")
            left.plot(fine, reconstructed[:, j], color="tab:orange", lw=0.9, label="decoded cubic")
            left.axhspan(actions[:, j].min(), actions[:, j].max(), alpha=0.08)
            left.set_title(meta["channels"][j], fontsize=9)
            left.set_ylabel("radians" if j not in (6, 13) else "0 closed / 1 open", fontsize=9)
            right.plot(x, residual[:, j], lw=0.9)
            tol = meta["diagnostics"]["settings"]["tolerance"]
            right.axhline(tol, color="red", ls="--", lw=0.6)
            right.axhline(-tol, color="red", ls="--", lw=0.6)
            right.set_ylabel("normalized residual")
            right.set_title(f"max |error| = {np.abs(residual[:, j]).max():.6g}", fontsize=9)
            for ax in (left, right):
                ax.vlines(knots, 0, 0.05, transform=ax.get_xaxis_transform(), color="tab:blue", lw=0.6)
                ax.grid(alpha=0.15)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("source action-step index; blue ticks = shared knots")
        fig.suptitle(f"{meta['task']} / {Path(meta['source']['path']).name} / {norm['scope']} / "
                     f"{meta['diagnostics']['reason']}\nDecoded command diagnostics; simulator execution not assessed")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        target = path.with_suffix(f".{name}.png")
        fig.savefig(target, dpi=120)
        plt.close(fig)
        outputs.append(str(target))
    ranges = range_diagnostics(spline, actions, norm)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    rows = [[meta["channels"][j]] + [f"{ranges[k][j]:.6g}" for k in ranges] for j in range(14)]
    table = ax.table(cellText=rows, colLabels=["channel", "demo min", "demo max", "curve min", "curve max", "below", "above"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title("Decoded ranges and overshoot (analytic cubic extrema); joint radians / gripper fractions")
    fig.tight_layout()
    target = path.with_suffix(".ranges.png")
    fig.savefig(target, dpi=130)
    plt.close(fig)
    outputs.append(str(target))
    path.with_suffix(".ranges.json").write_text(json.dumps(ranges, indent=2))
    print("Plots: " + ", ".join(outputs))
    return outputs


def visualize_sample(path, observation_time, scaling, timeline=None, video=None, output_prefix=None):
    from adaptive_spline import extract_local, decode_local, select_future_frames, identity
    path = Path(path)
    actions, spline, meta = read_cache(path)
    if not meta["diagnostics"]["success"]:
        raise ValueError("Cannot treat failed episode fit as a training target")
    if scaling["scope"] == "training-only" and scaling.get("normalization_identity") != meta["normalization"]["identity"]:
        raise ValueError("Knot/action normalization identities disagree")
    sample = extract_local(spline, observation_time, scaling)
    if timeline is None:
        if scaling["scope"] != "diagnostic-only":
            raise ValueError("Real sample inspection requires verified action/camera timestamps")
        timeline = dict(action_times=list(range(len(actions))), camera_times=list(range(len(actions))),
                        units="source_row", status="diagnostic-only assumed row association")
    else:
        if timeline.get("action_sha256") != meta["source"]["action_sha256"] or timeline.get("units") != "seconds":
            raise ValueError("Timeline must bind this episode action hash and use seconds")
        if timeline.get("status") != "verified":
            raise ValueError("Timeline has not been verified")
    if len(timeline["action_times"]) != len(actions):
        raise ValueError("Action timeline length mismatch")
    selection = select_future_frames(sample, timeline["action_times"], timeline["camera_times"])
    start, end = np.array(sample["valid_interval_relative"]) + observation_time
    times = np.unique(np.r_[np.linspace(start, end, 500), np.arange(np.ceil(start), np.floor(end) + 1)])
    local, report = decode_local(sample["parameters"], times, observation_time, scaling, meta["normalization"])
    global_actions = decode(spline(times), meta["normalization"])
    selection["clock_units"] = timeline["units"]
    sample["normalization_identity"] = meta["normalization"]["identity"]
    sample["source"] = meta["source"]
    video = video or meta["images"].get("preprocessed_video")
    sample["image_source"] = str(video) if video else None
    sample["source_cache_identity"] = meta["cache_identity"]
    sample["video_selection"] = selection
    sample["timeline"] = timeline
    sample["decoder_report"] = report
    sample["max_local_global_decoded_error"] = float(np.abs(local - global_actions).max())
    prefix = Path(output_prefix) if output_prefix else path.with_suffix(f".sample-{observation_time:g}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    def output(suffix):
        return Path(str(prefix) + suffix)
    if output(".npz").exists() or output(".json").exists():
        raise ValueError("Sample output exists; select a fresh --output-prefix")
    params = sample.pop("parameters")
    sample["sample_identity"] = identity(sample)
    with output(".npz").open("xb") as f:
        np.savez_compressed(f, parameters=params, metadata=json.dumps(sample),
                            comparison_times=times, locally_decoded=local, globally_decoded=global_actions)
    output(".json").write_text(json.dumps(sample, indent=2, allow_nan=False))

    fig, axes = plt.subplots(5, 1, figsize=(15, 14))
    for j in range(14):
        ax = axes[0 if j not in (6, 13) else 1]
        ax.plot(times, global_actions[:, j], lw=1)
        ax.plot(times[::10], local[::10, j], ".", ms=2)
    axes[0].set_title("Global lines / packed-target decoder dots: joints [rad]")
    axes[1].set_title("Continuous grippers [fraction]; no threshold")
    axes[2].plot(times, local - global_actions)
    axes[2].set_title(f"Local minus global; maximum decoded error {sample['max_local_global_decoded_error']:.3g}")
    relative_knots = params[:, 0].astype(float) * scaling["scale"] + scaling["offset"]
    axes[3].plot(np.arange(36), relative_knots, "o-", ms=3)
    axes[3].scatter(np.flatnonzero(np.array(sample["knot_source_indices"]) < 0),
                    relative_knots[np.array(sample["knot_source_indices"]) < 0], color="red", label="tail repetitions")
    axes[3].set_title("Packed knot times relative to observation [source steps]")
    axes[3].set_xlabel("packed row")
    axes[3].legend()
    axes[4].plot(params[:, 1:])
    axes[4].axvspan(sample["effective_control_points"] - 0.5, 35.5, alpha=0.15, color="red")
    axes[4].set_title("Normalized coefficients; shaded repeated/structural rows (coefficients are not actions)")
    axes[4].set_xlabel("packed row")
    for ax in axes[:3]:
        ax.axvline(observation_time, color="red", ls="--")
        ax.axvspan(start, end, color="green", alpha=0.04)
        ax.vlines(relative_knots[:sample["real_knot_count"]] + observation_time, 0, .05,
                  transform=ax.get_xaxis_transform(), color="blue", lw=.7)
        ax.set_xlim(start, end)
        ax.set_xlabel("source action-step index; red=current observation, blue=knots")
    fig.suptitle(f"{Path(meta['source']['path']).name}: sample at {observation_time:g}; span {sample['span']}; "
                 f"valid [{start:g}, {end:g}]\n{meta['normalization']['scope']}; {timeline['status']}")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output(".actions.png"), dpi=120)
    plt.close(fig)

    frames = [dict(camera_index=selection["current_camera_index"], camera_timestamp=selection["current_camera_timestamp"],
                   valid=True, knot_row="observation", repeated_frame=False)] + selection["frames"]
    images = {}
    if video:
        if not Path(video).exists():
            raise ValueError(f"Referenced video unavailable: {video}; sample arrays saved")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from data.utils.image_utils import get_video_frame_count, load_video_frames
        if get_video_frame_count(str(video)) != len(timeline["camera_times"]):
            raise ValueError("Video frame count differs from camera clock")
        indices = sorted({r["camera_index"] for r in frames})
        batch = load_video_frames(str(video), indices).permute(0, 2, 3, 1).numpy()
        images = dict(zip(indices, batch))
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for i, (ax, record) in enumerate(zip(axes.flat, frames)):
        if record["camera_index"] in images:
            ax.imshow(images[record["camera_index"]])
        else:
            ax.text(.5, .5, "Image unavailable\nNo demonstration image fabricated", ha="center", va="center")
        ax.axis("off")
        ax.set_title(f"{'current' if i == 0 else 'future'} frame {record['camera_index']} | knot {record['knot_row']}\n"
                     f"time {record['camera_timestamp']:.6g} {timeline['units']} | valid={record['valid']} | repeat={record['repeated_frame']}",
                     fontsize=10, color="black" if record["valid"] else "red")
    fig.suptitle(f"Raw validity {selection['raw_frame_valid']}\nConservative latent validity {selection['conservative_latent_valid']}; "
                 f"unchanged MOTUS compatible={selection['unchanged_motus_training_compatible']}\n{selection['mask_status']}")
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output(".frames.png"), dpi=110)
    plt.close(fig)
    print(json.dumps(dict(sample=str(output(".npz")), max_local_global_error=sample["max_local_global_decoded_error"],
                          images_available=bool(images), unchanged_motus_compatible=selection["unchanged_motus_training_compatible"]), indent=2))
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--observation-time", type=float)
    parser.add_argument("--knot-scaling", type=Path)
    parser.add_argument("--knot-time-scale", type=float, default=100.)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    if args.observation_time is None:
        visualize(args.cache)
    else:
        if args.knot_scaling:
            scaling = json.loads(args.knot_scaling.read_text())
        elif args.diagnostic_only:
            scaling = dict(scope="diagnostic-only", units="source_action_step", offset=0., scale=args.knot_time_scale)
        else:
            parser.error("Supply --knot-scaling or explicitly opt into --diagnostic-only")
        if scaling["scope"] == "diagnostic-only" and not args.diagnostic_only:
            parser.error("Diagnostic scaling requires --diagnostic-only")
        visualize_sample(args.cache, args.observation_time, scaling,
                         json.loads(args.timeline.read_text()) if args.timeline else None, args.video, args.output_prefix)
