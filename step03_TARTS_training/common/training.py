"""Training, frozen-WaveNet caches, and self-contained model/checkpoint IO."""

import argparse
import contextlib
import csv
import importlib.metadata
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .data import CCDUniformBatches, DatasetIndex, FullEpochBatches, NORMALIZATION, sha256
from .losses import dare_gram, metrics, regression_loss, source_guard_scale
from .models import build_model, load_model


CONTRACT = dict(frame="CCS", unit="micron", noll=list(range(4, 29)))
SCHEMA = "tarts-fam-training-v3"
STEP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FOLDERS = dict(wavenet="01_wavenet", aggregator="02_aggregator", dare="03_dare")
PATH_SUFFIXES = ("_root", "_dir", "_checkpoint", "_manifest", "_config")


def default_config(kind):
    return STEP_ROOT / CONFIG_FOLDERS[kind] / "config.yaml"


def local_path(path, parent):
    """Anchor paths while preserving repository-local symlink spelling."""
    path = Path(path).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else Path(parent) / path))


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    for key, value in list(config.items()):
        if value and key.endswith(PATH_SUFFIXES):
            config[key] = str(local_path(value, path.parent))
    return config


def dare_configs(path=None):
    config = read_config(path or default_config("dare"))
    models = {}
    for kind in ("wavenet", "aggregator"):
        model = read_config(config[f"{kind}_config"])
        initial = config.get(f"{kind}_checkpoint", str(Path(model["output_dir"]) / "model.pt"))
        model.update(config[kind])
        model.update(initial_checkpoint=initial, dare=config["dare"],
                     output_dir=str(Path(config["output_dir"]) / kind))
        models[kind] = model
    if Path(models["wavenet"]["dataset_root"]).resolve() != Path(models["aggregator"]["dataset_root"]).resolve():
        raise ValueError("Both networks must use the same simulation dataset")
    models["aggregator"].update(
        supervised_wave_checkpoint=models["wavenet"]["initial_checkpoint"],
        wave_checkpoint=str(Path(models["wavenet"]["output_dir"]) / "model.pt"))
    return config, models


def model_config(kind, variant="supervised", path=None):
    return dare_configs(path)[1][kind] if variant == "dare" else read_config(path or default_config(kind))


def output_directory(path, exist_ok=False):
    path = local_path(path, Path.cwd())
    repository = STEP_ROOT.parent
    resolved = path.resolve()
    inside = path.is_relative_to(repository) or resolved.is_relative_to(repository)
    allowed = path.is_relative_to(STEP_ROOT / "output") or resolved.is_relative_to(STEP_ROOT / "output")
    if inside and not allowed:
        raise ValueError("Repository-local outputs belong under step03_TARTS_training/output")
    if path.is_symlink():
        raise FileExistsError(f"Output target is a symlink: {path}")
    if not exist_ok and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def save_checkpoint(path, payload):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def prediction_cache(path, signature, build):
    """Cache predictions, not labels, images or padded Aggregator sequences."""
    path = Path(path)
    output_directory(path.parent, exist_ok=True)
    signature = json.loads(json.dumps(signature))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["signature"].item())) != signature:
                raise ValueError(f"Stale prediction cache: {path}; move this cache aside before rebuilding")
            return {key: data[key] for key in data.files if key != "signature"}
    arrays = build()
    partial = path.with_suffix(".partial")
    with partial.open("wb") as stream:
        np.savez(stream, signature=json.dumps(signature, sort_keys=True), **arrays)
    partial.replace(path)
    return arrays


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.console.write(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def autocast(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()


def forward(model, batch, kind, device):
    keys = ("image", "fx", "fy", "focal", "band") if kind == "wavenet" else ("features", "mean")
    values = [batch[key].to(device, dtype=torch.float32, non_blocking=True) for key in keys]
    return model(*values) if kind == "wavenet" else model(tuple(values))


def hidden(model, kind):
    return model.predictor_features if kind == "wavenet" else model.transformer_features


def predict(model, dataset, kind, device, precision="fp32", batch_size=128):
    model.eval()
    output = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            with autocast(device, precision):
                value = forward(model, batch, kind, device)
            output.append(value.float().cpu().numpy())
    prediction = np.concatenate(output)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    return prediction


def validation(model, dataset, kind, device, precision, batch_size):
    prediction = predict(model, dataset, kind, device, precision, batch_size)
    result = metrics(prediction, dataset.arrays["truth"])
    if kind == "wavenet":
        grouped = defaultdict(list)
        for i, key in enumerate(zip(dataset.arrays["state_index"], dataset.arrays["detector_id"])):
            grouped[key].append(i)
        pred = np.array([prediction[indices].mean(axis=0) for indices in grouped.values()])
        truth = np.array([dataset.arrays["truth"][indices].mean(axis=0) for indices in grouped.values()])
        result["grouped_detector"] = metrics(pred, truth)
    return result


def selection_mrsse(result, kind):
    return (result["grouped_detector"] if kind == "wavenet" else result)["mrsse_arcsec"]


def merge_ancestry(assignment, checkpoint):
    previous = checkpoint.get("split_assignment")
    if previous is None or checkpoint.get("schema_version") != SCHEMA:
        raise ValueError("Training accepts only checkpoints from this random-initialization workflow")
    for group, split in previous.items():
        if group in assignment and assignment[group] != split:
            raise ValueError(f"Checkpoint ancestry crosses the current split: pointing group {group}")
    return {**previous, **assignment}


def wave_predictions(index, split, checkpoint, cache_dir, device, precision="bf16", batch_size=128):
    signature = dict(dataset_identity=index.identity, wave_checkpoint_sha256=sha256(checkpoint),
                     precision=precision, normalization=NORMALIZATION, **CONTRACT)

    def build():
        model, wave = load_model(checkpoint, "wavenet", device)
        merge_ancestry(index.split_assignment, wave)
        if wave["dataset_identity"] != index.identity:
            raise ValueError("Frozen WaveNet was trained on a different dataset")
        dataset = index.wave(split)
        pred = predict(model, dataset, "wavenet", device, precision, batch_size)
        return dict(prediction_ccs_um=pred, sample_id=dataset.arrays["sample_id"])

    return prediction_cache(Path(cache_dir) / f"sim_{split}.npz", signature, build)


def validation_dare(model, source, target, kind, device, precision, batch_size, options):
    model.eval()
    if min(len(source), len(target)) < 2:
        raise ValueError("DARE validation needs at least two examples in each domain")
    count = min(max(len(source), len(target)), options["validation_batches"] * batch_size)
    positions = np.arange(count)
    # Cycle the smaller domain so validation can cover both complete populations.
    source_loader = DataLoader(Subset(source, positions % len(source)), batch_size=batch_size)
    target_loader = DataLoader(Subset(target, positions % len(target)), batch_size=batch_size)
    losses = []
    with torch.no_grad():
        for batch, real in zip(source_loader, target_loader):
            count = min(len(next(iter(batch.values()))), len(next(iter(real.values()))))
            if count < 2:
                continue
            with autocast(device, precision):
                forward(model, {key: value[:count] for key, value in batch.items()}, kind, device)
                a = hidden(model, kind)
                forward(model, {key: value[:count] for key, value in real.items()}, kind, device)
                b = hidden(model, kind)
            losses.append(float(dare_gram(a, b, **options["loss"])))
    if not losses:
        raise ValueError("No valid DARE validation batches")
    return float(np.mean(losses))


def provenance(index, device):
    return dict(dataset_identity=index.identity,
                samples={split: len(rows) for split, rows in index.splits.items()},
                code_sha256={str(path.relative_to(STEP_ROOT)): sha256(path)
                             for path in sorted(STEP_ROOT.rglob("*.py")) if "output" not in path.parts},
                packages={key: importlib.metadata.version(key) for key in ("torch", "numpy", "timm")},
                hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")


def run_training(kind, config, targets=None, resume=None):
    """Run one stage. A target factory keeps target loading/cache logs in this stage."""
    if not config.get("dare") and "initial_checkpoint" in config:
        raise ValueError("Supervised training starts randomly; use --resume only to continue this run")
    out = output_directory(config["output_dir"], exist_ok=bool(resume))
    with (out / "train.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                return _train(kind, config, targets, resume, out)
            except Exception:
                traceback.print_exc()
                raise


def _train(kind, config, targets, resume, out):
    from .reporting import write_report
    started = time.perf_counter()
    torch.set_num_threads(config.get("torch_threads", 8))
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config["device"])
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", False))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This configuration requires a GPU compute session")
    if config["precision"] == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; set precision: fp32")
    index = DatasetIndex(config["dataset_root"])
    run_info = provenance(index, device)
    print(f"Starting {kind}: {run_info['samples']}; device={run_info['hardware']}", flush=True)
    options = config.get("dare")
    initial_path = config.get("initial_checkpoint")
    initial_digest = sha256(initial_path) if initial_path else None
    wave_digest = sha256(config["wave_checkpoint"]) if kind == "aggregator" else None
    previous = None
    ancestry = dict(index.split_assignment)
    if resume or initial_path:
        model, previous = load_model(resume or initial_path, kind, device)
        ancestry = merge_ancestry(ancestry, previous)
        if previous["dataset_identity"] != index.identity:
            raise ValueError("Checkpoint dataset differs from current data")
        if model.hparams != config["model"]:
            raise ValueError("Checkpoint architecture differs from config")
    else:
        model = build_model(kind, config["model"]).to(device)
    if kind == "wavenet":
        model.set_runtime(config.get("channels_last", False))
    if kind == "aggregator":
        wave_info = torch.load(config["wave_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        ancestry = merge_ancestry(ancestry, wave_info)
        if wave_info["dataset_identity"] != index.identity:
            raise ValueError("Frozen WaveNet dataset differs from current data")
        if options and not resume and previous["wave_checkpoint_sha256"] != sha256(config["supervised_wave_checkpoint"]):
            raise ValueError("The two supervised starting checkpoints are not a matching pair")
        del wave_info
    if resume:
        if "optimizer_state_dict" not in previous:
            raise ValueError("Use resume.pt, not model.pt, to resume training")
        for key, value in previous["config"].items():
            if key != "epochs" and not key.endswith(PATH_SUFFIXES) and config.get(key) != value:
                raise ValueError(f"Resume changed training configuration: {key}")
        if previous["initial_checkpoint_sha256"] != initial_digest or previous["wave_checkpoint_sha256"] != wave_digest:
            raise ValueError("Resume inputs changed: initial weights or frozen WaveNet")
        if not (out / "model.pt").exists():
            selected = Path(resume).resolve().with_name("model.pt")
            save_checkpoint(out / "model.pt", torch.load(selected, map_location="cpu", weights_only=False))
    (out / "config.yaml").write_text(yaml.safe_dump(dict(config, provenance=run_info), sort_keys=False))
    source = {}
    for split in ("train", "val"):
        if kind == "wavenet":
            source[split] = index.wave(split)
        else:
            predictions = wave_predictions(index, split, config["wave_checkpoint"], out / "cache",
                                           device, config["wave_precision"], config["wave_batch_size"])
            source[split] = index.aggregate(split, predictions, config["require_both_sides"])
    targets = targets() if callable(targets) else targets
    if bool(options) != (targets is not None):
        raise ValueError("DARE configuration and unlabelled targets must be provided together")
    sampler = (CCDUniformBatches(source["train"], config["batch_size"], seed, config["stamps_per_ccd"])
               if kind == "wavenet" and config.get("stamps_per_ccd")
               else FullEpochBatches(source["train"], config["batch_size"], seed))
    train_loader = DataLoader(source["train"], batch_sampler=sampler)
    if kind == "wavenet" and "backbone_lr" in config:
        parameters = [{"params": model.cnn.parameters(), "lr": config["backbone_lr"]},
                      {"params": model.predictor.parameters(), "lr": config["lr"]}]
    else:
        parameters = [{"params": model.parameters(), "lr": config["lr"]}]
    base_lrs = [group["lr"] for group in parameters]
    optimizer = torch.optim.AdamW(parameters, weight_decay=config["weight_decay"],
                                  fused=bool(config.get("fused_adamw", False) and device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config["lr_factor"], patience=config["lr_patience"], min_lr=config["min_lr"])
    precision, batch_size = config["precision"], config["batch_size"]
    target_sampler = FullEpochBatches(targets["train"], batch_size, seed + 31) if options else None
    target_loader = DataLoader(targets["train"], batch_sampler=target_sampler) if options else None
    if resume:
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        scheduler.load_state_dict(previous["scheduler_state_dict"])
        start, history = previous["epoch"], previous["history"]
        baseline, base_dare = previous["baseline"], previous["baseline_dare"]
        best, best_epoch = previous["best_score"], previous["best_epoch"]
        stale, latest = previous["stale_epochs"], previous["latest_source_score"]
        torch.set_rng_state(previous["torch_rng"].cpu())
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in previous["cuda_rng"]])
    else:
        baseline = selection_mrsse(validation(model, source["val"], kind, device, precision, batch_size), kind)
        base_dare = validation_dare(model, source["val"], targets["val"], kind, device, precision,
                                   batch_size, options) if options else 0.0
        if options and (baseline <= 0 or base_dare <= 0):
            raise ValueError("DARE selection needs positive baseline source and feature losses")
        best = 1 + options["selection_weight"] if options else baseline
        best_epoch, latest, stale, start, history = 0, baseline, 0, 0, []
    del previous
    if config["epochs"] <= start:
        raise ValueError("epochs must exceed the checkpoint epoch")
    preparation_seconds = time.perf_counter() - started
    print(f"Prepared train={len(source['train'])}, val={len(source['val'])} in {preparation_seconds:.1f}s", flush=True)

    def model_payload(epoch):
        return dict(schema_version=SCHEMA, model_kind=kind, **CONTRACT,
                    model_hparams=model.hparams, model_state_dict=model.state_dict(), epoch=epoch,
                    training_stage="dare_gram" if options else "supervised",
                    initialization="supervised" if options else "random",
                    config=config, provenance=run_info, dataset_identity=index.identity,
                    split_assignment=ancestry, normalization=NORMALIZATION,
                    initial_checkpoint_sha256=initial_digest, wave_checkpoint_sha256=wave_digest)

    def resume_payload(epoch):
        return dict(model_payload(epoch), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), history=history,
                    baseline=baseline, baseline_dare=base_dare, best_score=best, best_epoch=best_epoch,
                    stale_epochs=stale, latest_source_score=latest, torch_rng=torch.get_rng_state(),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])

    if not resume:
        save_checkpoint(out / "model.pt", model_payload(0))
        save_checkpoint(out / "resume.pt", resume_payload(0))
    write_report(out, kind, history, best_epoch)
    for epoch in range(start + 1, config["epochs"] + 1):
        epoch_started = time.perf_counter()
        sampler.epoch = epoch
        if target_sampler is not None:
            target_sampler.epoch = epoch
        warmup = config.get("warmup_epochs", 0)
        if epoch <= warmup:
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * epoch / warmup
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        scale = source_guard_scale(latest, baseline, options["source_guard_fraction"]) if options else 0
        real_iterator = iter(target_loader) if options else None
        sums = np.zeros(3)
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            truth = batch["truth"].to(device, dtype=torch.float32)
            with autocast(device, precision):
                prediction = forward(model, batch, kind, device)
                a = hidden(model, kind)
                supervised = regression_loss(prediction, truth, kind)
            alignment = supervised.new_zeros(())
            if options and scale > 0:
                try:
                    real = next(real_iterator)
                except StopIteration:
                    real_iterator = iter(target_loader)
                    real = next(real_iterator)
                count, real_count = len(truth), len(next(iter(real.values())))
                real = {key: value[torch.arange(count) % real_count] for key, value in real.items()}
                # Target gradients stay enabled; target batches do not update source BatchNorm statistics.
                model.eval()
                with autocast(device, precision):
                    forward(model, real, kind, device)
                    b = hidden(model, kind)
                model.train()
                alignment = dare_gram(a, b, **options["loss"])
            loss = supervised + (options["weight"] * scale * alignment if options else 0)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            sums += [float(supervised.detach()) * len(truth), float(alignment.detach()) * len(truth), len(truth)]
        train_seconds = time.perf_counter() - epoch_started
        if int(sums[2]) != sampler.size:
            raise RuntimeError("Emitted sample count differs from the training sampler")
        result = validation(model, source["val"], kind, device, precision, batch_size)
        latest = selection_mrsse(result, kind)
        alignment = validation_dare(model, source["val"], targets["val"], kind, device, precision,
                                   batch_size, options) if options else 0.0
        eligible = not options or latest <= baseline * (1 + options["source_guard_fraction"])
        score = latest / baseline + options["selection_weight"] * alignment / base_dare if options else latest
        if epoch >= warmup:
            scheduler.step(latest)
        row = dict(epoch=epoch, train_samples=int(sums[2]), train_supervised=sums[0] / sums[2],
                   train_dare=sums[1] / sums[2], dare_scale=scale, val_mrsse_arcsec=latest,
                   val_dare=alignment, selection_score=score, source_guard_pass=eligible, lr=epoch_lr,
                   val_coefficient_rmse_um=result["coefficient_rmse_um"],
                   train_seconds=train_seconds, validation_seconds=time.perf_counter() - epoch_started - train_seconds)
        history.append(row)
        improved = eligible and score < best
        stale = 0 if improved else stale + 1
        if improved:
            best, best_epoch = score, epoch
            save_checkpoint(out / "model.pt", model_payload(epoch))
        save_checkpoint(out / "resume.pt", resume_payload(epoch))
        with (out / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(dict(row, epoch_total_seconds=time.perf_counter() - epoch_started)), flush=True)
        if epoch > warmup and stale >= config["early_stop_patience"]:
            break
    write_report(out, kind, history, best_epoch)
    del source, train_loader, targets, target_loader, model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved {kind} in {time.perf_counter() - started:.1f}s; selected epoch={best_epoch}", flush=True)
    return out / "model.pt"


def training_cli(kind):
    parser = argparse.ArgumentParser(description=f"Train FAM {kind} from random initialization")
    parser.add_argument("--config", type=Path, default=default_config(kind))
    parser.add_argument("--resume", type=Path, help="Continue a saved optimizer checkpoint")
    if kind == "aggregator":
        parser.add_argument("--wavenet", type=Path, help="Frozen WaveNet checkpoint")
    args = parser.parse_args()
    config = read_config(args.config)
    if kind == "aggregator" and args.wavenet:
        config["wave_checkpoint"] = str(args.wavenet.resolve())
    run_training(kind, config, resume=args.resume)
