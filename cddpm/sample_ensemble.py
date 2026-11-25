"""Many independent DDPM samples at a few fixed conditions, for Poisson checks.

    python sample_ensemble.py --variant C --epoch 12000
    python sample_ensemble.py --variant B --epoch 19000 --samples 512 --condition 0.51 1e22
    python sample_ensemble.py --epoch 12000 --checkpoint-dir ../377_Aux_Poisson_Deviance_Corrected/outputs

Each condition is saved as condition_<i>_x<x>_NH<nh>.npz with arrays
samples (N, 48, 48) in photon counts (not rounded, no extra noise), seeds,
x, nH and metadata_json. Sample i of condition c always gets the same seed,
whatever the batch size.
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from config import RUN_DIR, VARIANTS, model_dir
from dsh.checkpoint import checkpoint_path, load_model, run_tag
from dsh.data import normalize_condition, to_counts
from dsh.sampling import ddpm_scheduler, sample_ddpm

SOURCE_DISTANCE_PC = 10000.0
DEFAULT_CONDITIONS = [(x, nh) for x in (0.0648, 0.5100, 0.9551) for nh in (1.08e21, 1.0e22, 9.28e22)]


def sample_seed(master_seed, condition_index, sample_index):
    state = np.random.SeedSequence([master_seed, condition_index, sample_index]).generate_state(1, dtype=np.uint64)
    return int(state[0]) & ((1 << 63) - 1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomically(path, write):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        write(file)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epoch", type=int, required=True, help="Checkpoint epoch, e.g. 12000")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="Use this folder's model_<variant>/")
    parser.add_argument("--checkpoint-dir", type=Path, help="Any other checkpoint folder, e.g. a 377 outputs/")
    parser.add_argument("--condition", nargs=2, type=float, action="append", metavar=("X", "NH"),
                        help="Fractional dust distance and N_H in cm^-2; repeat for more (default: 9 conditions)")
    parser.add_argument("--samples", type=int, default=256, help="Samples per condition")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1000, help="DDPM steps; keep 1000 for results")
    parser.add_argument("--output", type=Path, help="New folder (default: POISSON_ENSEMBLES/<name>)")
    args = parser.parse_args()
    if (args.variant is None) == (args.checkpoint_dir is None):
        parser.error("give exactly one of --variant or --checkpoint-dir")
    checkpoint_dir = args.checkpoint_dir or model_dir(args.variant)
    conditions = args.condition or DEFAULT_CONDITIONS
    if args.samples < 2 or not 1 <= args.steps <= 1000:
        parser.error("need --samples >= 2 and 1 <= --steps <= 1000")
    if any(not (0 < x < 1 and nh > 0) for x, nh in conditions):
        parser.error("each condition needs 0 < X < 1 and NH > 0")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = checkpoint_path(checkpoint_dir, args.epoch).resolve()
    model, metadata, epoch = load_model(ckpt, device)
    stats = {key: float(metadata["condition_stats"][key]) for key in ("mean_c1", "std_c1", "mean_c2", "std_c2")}
    scale, use_anscombe = float(metadata["data_max"]), bool(metadata["use_anscombe"])
    scheduler = ddpm_scheduler(args.steps)

    run_name = run_tag(checkpoint_dir)
    output = args.output or RUN_DIR / "POISSON_ENSEMBLES" / f"{run_name}_epoch_{args.epoch:05d}_ddpm{args.steps}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt), "epoch": epoch,
        "method": "ddpm", "steps": args.steps, "scheduler": dict(scheduler.config),
        "master_seed": args.seed,
        "seed_scheme": "SeedSequence([master_seed, condition_index, sample_index]), uint64 masked to 63 bits",
        "samples_per_condition": args.samples, "batch_size": args.batch_size, "device": str(device),
        "image_size": 48, "pixel_scale_arcsec": 45.0, "source_distance_pc": SOURCE_DISTANCE_PC,
        "use_anscombe": use_anscombe, "data_max": scale, "condition_stats": stats,
        **({"inverse": "algebraic_anscombe", "anscombe_global_max": scale} if use_anscombe else {}),
        "count_processing": "float32, no rounding, no additional Poisson draw, no flux rescaling",
        "versions": {name: version(name) for name in ("torch", "diffusers", "numpy")},
        "conditions": [{"x": x, "nH": nh} for x, nh in conditions], "completed_files": [],
    }
    save_manifest = lambda: (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_manifest()

    for ci, (x, nh) in enumerate(conditions):
        condition = list(normalize_condition(x * SOURCE_DISTANCE_PC, nh, stats))
        seeds = np.array([sample_seed(args.seed, ci, i) for i in range(args.samples)], dtype=np.uint64)
        samples = np.empty((args.samples, 48, 48), dtype=np.float32)
        for start in range(0, args.samples, args.batch_size):
            batch = seeds[start:start + args.batch_size]
            generators = [torch.Generator(device=device).manual_seed(int(s)) for s in batch]
            cond = torch.tensor(condition, dtype=torch.float32, device=device).repeat(len(batch), 1)
            images = sample_ddpm(model, cond, scheduler, generators=generators, progress=False)
            samples[start:start + len(batch)] = to_counts(images, scale, use_anscombe)[:, 0].cpu().numpy()
            logging.info("Condition %d/%d (x=%g, NH=%g): %d/%d", ci + 1, len(conditions), x, nh,
                         start + len(batch), args.samples)

        filename = f"condition_{ci:03d}_x{x:.6g}_NH{nh:.6g}.npz"
        file_metadata = {k: v for k, v in manifest.items() if k != "completed_files"}
        file_metadata.update(condition_index=ci, x=x, nH=nh, normalized_condition=condition,
                             status="condition_complete")
        write_atomically(output / filename, lambda f: np.savez_compressed(
            f, samples=samples, seeds=seeds, x=x, nH=nh, metadata_json=json.dumps(file_metadata)))
        manifest["completed_files"].append(filename)
        save_manifest()

    manifest["status"] = "complete"
    save_manifest()
    logging.info("Saved ensembles in %s", output)


if __name__ == "__main__":
    main()
