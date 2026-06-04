import os
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass


CONDITION_DIRS = {
    "c1": "35Hz12kN",
    "c2": "37.5Hz11kN",
    "c3": "40Hz10kN",
}


BEARING_META = {
    "Bearing1_1": {"condition": "c1", "fault": "outer"},
    "Bearing1_2": {"condition": "c1", "fault": "outer"},
    "Bearing1_3": {"condition": "c1", "fault": "outer"},
    "Bearing1_4": {"condition": "c1", "fault": "cage"},
    "Bearing1_5": {"condition": "c1", "fault": "inner_outer"},
    "Bearing2_1": {"condition": "c2", "fault": "inner"},
    "Bearing2_2": {"condition": "c2", "fault": "outer"},
    "Bearing2_3": {"condition": "c2", "fault": "cage"},
    "Bearing2_4": {"condition": "c2", "fault": "outer"},
    "Bearing2_5": {"condition": "c2", "fault": "outer"},
    "Bearing3_1": {"condition": "c3", "fault": "outer"},
    "Bearing3_2": {"condition": "c3", "fault": "compound_all"},
    "Bearing3_3": {"condition": "c3", "fault": "inner"},
    "Bearing3_4": {"condition": "c3", "fault": "inner"},
    "Bearing3_5": {"condition": "c3", "fault": "outer"},
}


OPEN_SET_TASKS = {
    "c1_outer_to_c2_mixed": {
        "source": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
        "target": ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5"],
        "known_faults": ["outer"],
    },
    "c1_outer_to_c3_mixed": {
        "source": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
        "target": ["Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5"],
        "known_faults": ["outer"],
    },
    "c2_outer_to_c3_mixed": {
        "source": ["Bearing2_2", "Bearing2_4", "Bearing2_5"],
        "target": ["Bearing3_1", "Bearing3_2", "Bearing3_3", "Bearing3_4", "Bearing3_5"],
        "known_faults": ["outer"],
    },
    "c3_outer_to_c2_mixed": {
        "source": ["Bearing3_1", "Bearing3_5"],
        "target": ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4", "Bearing2_5"],
        "known_faults": ["outer"],
    },
    "c1_outer_to_c2_outer": {
        "source": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
        "target": ["Bearing2_2", "Bearing2_4", "Bearing2_5"],
        "known_faults": ["outer"],
    },
    "c1_outer_to_c3_outer": {
        "source": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
        "target": ["Bearing3_1", "Bearing3_5"],
        "known_faults": ["outer"],
    },
}


def bearing_dir(data_root, bearing):
    meta = BEARING_META[bearing]
    return Path(data_root) / CONDITION_DIRS[meta["condition"]] / bearing


def sorted_csv_files(folder):
    files = [p for p in Path(folder).glob("*.csv")]
    return sorted(files, key=lambda p: int(p.stem))


def channel_features(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    std = np.std(x)
    centered = x - mean
    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x * x))
    max_abs = np.max(abs_x)
    p2p = np.max(x) - np.min(x)
    mean_abs = np.mean(abs_x)
    sqrt_abs_mean = np.mean(np.sqrt(abs_x + eps))
    skew = np.mean(centered ** 3) / (std ** 3 + eps)
    kurt = np.mean(centered ** 4) / (std ** 4 + eps)
    crest = max_abs / (rms + eps)
    shape = rms / (mean_abs + eps)
    impulse = max_abs / (mean_abs + eps)
    margin = max_abs / (sqrt_abs_mean ** 2 + eps)
    energy = np.mean(x * x)
    return np.array(
        [mean, std, rms, max_abs, p2p, skew, kurt, crest, shape, impulse, margin, energy],
        dtype=np.float32,
    )


def csv_features(csv_path):
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise RuntimeError("invalid XJTU-SY csv shape for {}".format(csv_path))
    h = channel_features(raw[:, 0])
    v = channel_features(raw[:, 1])
    return np.concatenate([h, v], axis=0).astype(np.float32)


def load_or_build_bearing(data_root, cache_root, bearing, rebuild=False):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / "{}.npz".format(bearing)
    if cache_file.exists() and not rebuild:
        obj = np.load(cache_file)
        return obj["x"].astype(np.float32), obj["y"].astype(np.float32)

    folder = bearing_dir(data_root, bearing)
    files = sorted_csv_files(folder)
    if not files:
        raise RuntimeError("no csv files found in {}".format(folder))
    x = np.stack([csv_features(p) for p in files], axis=0).astype(np.float32)
    life = len(files)
    if life == 1:
        y = np.zeros(1, dtype=np.float32)
    else:
        y = (np.arange(life - 1, -1, -1, dtype=np.float32) / float(life - 1)).astype(np.float32)
    np.savez_compressed(cache_file, x=x, y=y, bearing=bearing)
    return x, y


def compute_scaler(data_root, cache_root, bearings, rebuild=False):
    arrays = [load_or_build_bearing(data_root, cache_root, b, rebuild)[0] for b in bearings]
    all_x = np.concatenate(arrays, axis=0)
    mean = all_x.mean(axis=0).astype(np.float32)
    std = all_x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def make_windows(raw, label, seq_len):
    if torch is None:
        raise RuntimeError("PyTorch is required to build XJTU-SY window datasets")
    raw = torch.tensor(raw, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.float)
    life, feature_dim = raw.shape
    if life < 1:
        raise RuntimeError("empty bearing trajectory")

    data, labels, padding = [], [], []
    for i in range(seq_len - 1):
        left = seq_len - i - 1
        data.append(torch.cat([torch.zeros(left, feature_dim), raw[: i + 1]], dim=0))
        labels.append(torch.cat([torch.ones(left), label[: i + 1]], dim=0))
        padding.append(torch.cat([torch.ones(left), torch.zeros(i + 1)], dim=0))

    for i in range(seq_len - 1, life):
        data.append(raw[i - seq_len + 1 : i + 1])
        labels.append(label[i - seq_len + 1 : i + 1])
        padding.append(torch.zeros(seq_len))

    for i in range(seq_len - 1):
        right = i + 1
        valid = seq_len - right
        data.append(torch.cat([raw[life - valid :], torch.zeros(right, feature_dim)], dim=0))
        labels.append(torch.cat([label[life - valid :], torch.zeros(right)], dim=0))
        padding.append(torch.cat([torch.zeros(valid), torch.ones(right)], dim=0))

    return data, labels, padding


class XJTUBearingDataset(Dataset):
    def __init__(self, data_root, cache_root, bearings, seq_len=32, scaler=None, rebuild=False):
        self.data = []
        self.label = []
        self.padding = []
        self.bearing = []
        if scaler is None:
            scaler = compute_scaler(data_root, cache_root, bearings, rebuild)
        mean, std = scaler
        self.scaler = scaler
        for bearing in bearings:
            x, y = load_or_build_bearing(data_root, cache_root, bearing, rebuild)
            x = (x - mean) / std
            data, label, padding = make_windows(x, y, seq_len)
            self.data.extend(data)
            self.label.extend(label)
            self.padding.extend(padding)
            self.bearing.extend([bearing] * len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]


class XJTUSingleBearingDataset(Dataset):
    def __init__(self, data_root, cache_root, bearing, seq_len=32, scaler=None, rebuild=False):
        if scaler is None:
            scaler = compute_scaler(data_root, cache_root, [bearing], rebuild)
        mean, std = scaler
        raw, label = load_or_build_bearing(data_root, cache_root, bearing, rebuild)
        raw = (raw - mean) / std
        self.raw_label = torch.tensor(label, dtype=torch.float)
        self.life = len(label)
        self.data, self.label, self.padding = make_windows(raw, label, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.padding[index]
