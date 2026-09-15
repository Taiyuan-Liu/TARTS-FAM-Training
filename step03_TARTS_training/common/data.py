"""Read step02 shards and reconstruct state/CCD groups from sample metadata."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Sampler


BAND_WAVELENGTH_UM = np.array([0.3671, 0.4827, 0.6223, 0.7546, 0.8691, 0.9712], np.float32)
NORMALIZATION = dict(image="per-image minmax then population z-score",
                     field="deg_to_rad / 0.021", focal="2 * intra - 1; intra=1, extra=0",
                     band="(effective_wavelength_um - 0.710) / 0.174")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_images(images):
    images = np.asarray(images, np.float32).copy()
    if images.ndim != 3 or images.shape[1:] != (160, 160):
        raise ValueError(f"Expected (N,160,160), got {images.shape}")
    images -= images.min(axis=(1, 2), keepdims=True)
    peak = images.max(axis=(1, 2), keepdims=True)
    if not np.isfinite(peak).all() or np.any(peak <= 0):
        raise ValueError("Non-finite or constant image")
    images /= peak
    images -= images.mean(axis=(1, 2), keepdims=True)
    images /= images.std(axis=(1, 2), keepdims=True)
    return images


def wave_metadata(fx, fy, intra, band):
    band, intra = np.asarray(band, int), np.asarray(intra, int)
    if not np.isin(band, np.arange(6)).all() or not np.isin(intra, [0, 1]).all():
        raise ValueError("Expected band=0..5 and intra=0/1")
    return dict(fx=(np.deg2rad(np.asarray(fx, np.float32)) / 0.021)[:, None],
                fy=(np.deg2rad(np.asarray(fy, np.float32)) / 0.021)[:, None],
                focal=(intra.astype(np.float32) * 2 - 1)[:, None],
                band=((BAND_WAVELENGTH_UM[band] - 0.710) / 0.174)[:, None])


class ArrayDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = {key: np.asarray(value) for key, value in arrays.items()}
        sizes = {len(value) for value in self.arrays.values()}
        if len(sizes) != 1 or not next(iter(sizes)):
            raise ValueError("Empty dataset or mismatched array lengths")
        for key, value in self.arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {key}")

    def __len__(self):
        return len(next(iter(self.arrays.values())))

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.arrays.items()}


class DatasetIndex:
    """Index NPZ metadata; images are loaded only by wave()."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.assignments, group_split = {}, {}
        with (self.root / "splits.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                state, group, split = int(row["state_index"]), int(row["split_group_id"]), row["split"]
                if state < 0 or group < 0 or state in self.assignments or split not in ("train", "val", "test"):
                    raise ValueError(f"Invalid or duplicate split row: {row}")
                if group_split.setdefault(str(group), split) != split:
                    raise ValueError(f"Pointing group crosses splits: {group}")
                self.assignments[state] = (group, split)
        if not self.assignments:
            raise ValueError("Empty split table")
        self.split_assignment = dict(sorted(group_split.items()))
        self.rows, self.shards, self._identity = [], {}, None
        seen_ids, seen_sources = set(), set()
        integers = ("sample_id", "state_index", "detector_id", "source_stamp_index", "seq_num", "intra", "band")
        floats = ("field_x_deg", "field_y_deg", "snr", "rtp_deg")
        for split in ("train", "val", "test"):
            directory = self.root / split
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            for path in sorted(directory.glob("*.npz")):
                relative = str(path.relative_to(self.root))
                self.shards[relative] = self._file_stat(path)
                with np.load(path, allow_pickle=False) as shard:
                    if str(shard["split"].item()) != split:
                        raise ValueError(f"Shard split disagrees with directory: {path}")
                    values = {key: shard[key] for key in integers + floats}
                    n = len(values["sample_id"])
                    for key, value in values.items():
                        if value.shape != (n,) or not n or not np.isfinite(value).all():
                            raise ValueError(f"Invalid metadata: {path} {key}")
                        if key in integers and value.dtype.kind not in "iu":
                            raise ValueError(f"Expected integer metadata: {path} {key}")
                self._check_unchanged(relative)
                for i in range(n):
                    row = {key: int(values[key][i]) for key in integers}
                    row.update({key: float(values[key][i]) for key in floats})
                    state, sample = row["state_index"], row["sample_id"]
                    if state not in self.assignments or self.assignments[state][1] != split:
                        raise ValueError(f"State {state} is not assigned to {split}")
                    if row["intra"] not in (0, 1) or row["band"] not in range(6):
                        raise ValueError(f"Invalid side/band: {path} row {i}")
                    if row["seq_num"] != 2 * state + (1 if row["intra"] else 2):
                        raise ValueError(f"Exposure sequence disagrees with state/side: {path} row {i}")
                    source = state, row["detector_id"], row["intra"], row["source_stamp_index"]
                    if sample < 0 or min(source) < 0 or sample in seen_ids or source in seen_sources:
                        raise ValueError(f"Duplicate or invalid sample: {path} row {i}")
                    seen_ids.add(sample)
                    seen_sources.add(source)
                    row.update(split=split, split_group_id=self.assignments[state][0],
                               side="intra" if row.pop("intra") else "extra", band_index=row.pop("band"),
                               shard_relpath=relative, shard_row=i)
                    self.rows.append(row)
        if not self.rows:
            raise ValueError("No samples in dataset")
        self.splits = {split: [row for row in self.rows if row["split"] == split]
                       for split in ("train", "val", "test")}

    @staticmethod
    def _file_stat(path):
        info = path.stat()
        return info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _check_unchanged(self, relative):
        if self._file_stat(self.root / relative) != self.shards[relative]:
            raise ValueError(f"Dataset changed after indexing: {relative}")

    @property
    def identity(self):
        """Hash the split assignment and actual NPZ bytes, independent of location."""
        if self._identity is None:
            print(f"Hashing {len(self.shards)} NPZ shards for checkpoint/cache identity", flush=True)
            hashes = []
            for relative in self.shards:
                self._check_unchanged(relative)
                hashes.append((relative, sha256(self.root / relative)))
                self._check_unchanged(relative)
            payload = dict(splits=sorted(self.assignments.items()), shards=hashes)
            self._identity = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
        return self._identity

    def wave(self, split):
        rows = self.splits[split]
        if not rows:
            raise ValueError(f"No samples in {split}")
        locations = defaultdict(list)
        for index, row in enumerate(rows):
            locations[row["shard_relpath"]].append((index, int(row["shard_row"])))
        images = np.empty((len(rows), 160, 160), np.float32)
        truth = np.empty((len(rows), 25), np.float32)
        for relative, positions in locations.items():
            self._check_unchanged(relative)
            dest, src = np.asarray(positions).T
            with np.load(self.root / relative, allow_pickle=False) as shard:
                expected = [int(rows[i]["sample_id"]) for i in dest]
                if not np.array_equal(shard["sample_id"][src], expected):
                    raise ValueError(f"sample_id mismatch in {relative}")
                raw = shard["image"]
                if raw.shape != (len(shard["sample_id"]), 160, 160) or raw.dtype != np.float32:
                    raise ValueError(f"Expected float32 (N,160,160) images: {relative}")
                for key in ("zk_true_ccs_um", "zk_true_ocs_um"):
                    labels = shard[key]
                    if (labels.shape != (len(raw), 25) or labels.dtype != np.float32
                            or not np.isfinite(labels).all()):
                        raise ValueError(f"Expected finite float32 (N,25) labels: {relative} {key}")
                images[dest] = normalize_images(raw[src])
                truth[dest] = shard["zk_true_ccs_um"][src]
                for key in ("state_index", "detector_id", "field_x_deg", "field_y_deg", "snr"):
                    if not np.allclose(shard[key][src], [float(rows[i][key]) for i in dest],
                                       rtol=1e-6, atol=1e-7):
                        raise ValueError(f"Metadata mismatch: {relative} {key}")
                if not np.array_equal(shard["intra"][src], [int(rows[i]["side"] == "intra") for i in dest]):
                    raise ValueError(f"Focal side mismatch in {relative}")
                if not np.array_equal(shard["band"][src], [int(rows[i]["band_index"]) for i in dest]):
                    raise ValueError(f"Band mismatch in {relative}")
            self._check_unchanged(relative)
        metadata = {key: np.array([int(row[key]) for row in rows], np.int64)
                    for key in ("sample_id", "state_index", "detector_id", "seq_num", "source_stamp_index")}
        metadata["intra"] = np.array([row["side"] == "intra" for row in rows], np.int8)
        metadata["rtp_deg"] = np.array([row["rtp_deg"] for row in rows], np.float32)
        inputs = wave_metadata([row["field_x_deg"] for row in rows],
                               [row["field_y_deg"] for row in rows], metadata["intra"],
                               [row["band_index"] for row in rows])
        print(f"Loaded {split}: {len(rows)} stamps, images {images.nbytes / 1e9:.2f} GB", flush=True)
        return ArrayDataset(dict(image=images, truth=truth, **metadata, **inputs))

    def aggregate(self, split, predictions, require_both_sides=True):
        rows = self.splits[split]
        labels = self.labels(split)
        label_by_id = {row["sample_id"]: labels[i] for i, row in enumerate(rows)}
        ids = predictions["sample_id"]
        by_id = {int(value): index for index, value in enumerate(ids)}
        if len(by_id) != len(ids) or set(by_id) != {int(row["sample_id"]) for row in rows}:
            raise ValueError("Prediction cache does not match split sample IDs")
        groups = defaultdict(list)
        for row in rows:
            groups[int(row["state_index"]), int(row["detector_id"])].append(row)
        output = defaultdict(list)
        for (state, detector), members in sorted(groups.items()):
            if require_both_sides and {row["side"] for row in members} != {"intra", "extra"}:
                continue
            members.sort(key=lambda row: (row["side"] != "intra", int(row["source_stamp_index"])))
            indices = [by_id[int(row["sample_id"])] for row in members]
            pred = predictions["prediction_ccs_um"][indices]
            truth = np.array([label_by_id[row["sample_id"]] for row in members])
            if not np.allclose(truth, truth[0], rtol=0, atol=1e-6):
                raise ValueError(f"Different CCD-center labels inside state={state}, detector={detector}")
            fields = np.array([[float(row["field_x_deg"]), float(row["field_y_deg"])]
                               for row in members], np.float32)
            snr = np.array([float(row["snr"]) for row in members], np.float32)
            feature, mean = sequence_features(pred, fields, snr)
            values = dict(features=feature, mean=mean, median=np.median(pred, axis=0),
                          truth=truth[0], state_index=state, detector_id=detector, length=len(pred))
            for key, value in values.items():
                output[key].append(value)
        return ArrayDataset(output)

    def labels(self, split):
        """Read labels in index order without decompressing or allocating images."""
        rows = self.splits[split]
        truth = np.empty((len(rows), 25), np.float32)
        locations = defaultdict(list)
        for i, row in enumerate(rows):
            locations[row["shard_relpath"]].append((i, row["shard_row"]))
        for relative, positions in locations.items():
            self._check_unchanged(relative)
            dest, src = np.asarray(positions).T
            with np.load(self.root / relative, allow_pickle=False) as shard:
                labels = shard["zk_true_ccs_um"]
                if labels.shape != (len(shard["sample_id"]), 25) or labels.dtype != np.float32:
                    raise ValueError(f"Invalid CCS labels: {relative}")
                if not np.isfinite(labels).all():
                    raise ValueError(f"Non-finite CCS labels: {relative}")
                truth[dest] = labels[src]
            self._check_unchanged(relative)
        return truth


def sequence_features(predictions, fields, snr):
    if not 0 < len(predictions) <= 200:
        raise ValueError(f"Expected 1..200 donuts, got {len(predictions)}")
    if not np.isfinite(snr).all() or snr.max() <= 0:
        raise ValueError("Invalid sequence SNR")
    padded = np.zeros((200, 28), np.float32)
    padded[:len(predictions)] = np.column_stack([predictions, fields, snr / snr.max()])
    return padded, predictions.mean(axis=0).astype(np.float32)


class FullEpochBatches(Sampler):
    """Shuffle every sample once; merge a final singleton for BatchNorm safety."""

    def __init__(self, dataset, batch_size, seed):
        if len(dataset) < 2 or batch_size < 2:
            raise ValueError("Training requires at least two samples and batch_size >= 2")
        self.size = len(dataset)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        chosen = rng.permutation(self.size).tolist()
        batches = [chosen[start:start + self.batch_size] for start in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(count > 1 and self.size % self.batch_size == 1)


class CCDUniformBatches(FullEpochBatches):
    """Draw equal counts per state/CCD, split equally between available sides."""

    def __init__(self, dataset, batch_size, seed, stamps_per_ccd):
        super().__init__(dataset, batch_size, seed)
        self.stamps_per_ccd = int(stamps_per_ccd)
        if self.stamps_per_ccd < 2:
            raise ValueError("Use at least two stamps per CCD")
        self.groups = defaultdict(lambda: {0: [], 1: []})
        arrays = dataset.arrays
        for i, (state, detector, side) in enumerate(zip(
                arrays["state_index"], arrays["detector_id"], arrays["intra"])):
            self.groups[int(state), int(detector)][int(side)].append(i)
        self.size = len(self.groups) * self.stamps_per_ccd

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        chosen = []
        for key in sorted(self.groups):
            sides = self.groups[key]
            if sides[0] and sides[1]:
                pools = [(sides[1], self.stamps_per_ccd // 2),
                         (sides[0], self.stamps_per_ccd - self.stamps_per_ccd // 2)]
            else:
                pools = [(sides[1] or sides[0], self.stamps_per_ccd)]
            for pool, count in pools:
                chosen.extend(rng.choice(np.asarray(pool), size=count, replace=len(pool) < count).astype(int).tolist())
        rng.shuffle(chosen)
        batches = [chosen[start:start + self.batch_size] for start in range(0, len(chosen), self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches


def real_wave(root, split):
    """Read the existing already-normalized, label-free real-target NPZ shards."""
    parts = defaultdict(list)
    files = sorted((Path(root) / "shards" / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No real target shards: {root}/{split}")
    for path in files:
        with np.load(path, allow_pickle=False) as shard:
            forbidden = [key for key in shard.files
                         if any(word in key.lower() for word in ("truth", "label", "zern", "zk", "danish"))]
            if forbidden:
                raise ValueError(f"Real target must not contain labels: {forbidden}")
            image = shard["image"]
            if image.shape[1:] != (160, 160):
                raise ValueError(f"Unexpected target image shape: {image.shape}")
            # Historical target shards already contain population-z-scored float16 images.
            if not np.allclose(image.mean((1, 2), dtype=np.float32), 0, atol=0.005) or not np.allclose(
                    image.astype(np.float32).std((1, 2)), 1, atol=0.005):
                raise ValueError("Real target images must be pre-normalized")
            values = dict(image=image, visit=shard["visit"], detector_id=shard["detector_id"],
                          **wave_metadata(shard["field_x_deg"], shard["field_y_deg"],
                                          shard["intra"], shard["band"]))
            for key, value in values.items():
                parts[key].append(value)
    return ArrayDataset({key: np.concatenate(value) for key, value in parts.items()})
