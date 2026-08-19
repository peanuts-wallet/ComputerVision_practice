#!/usr/bin/env python3
"""Train a configurable 3- or 6-node ST-GNN from Town Centre tracking data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/towncentre_three_roi_matplotlib")

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_CSV = (
    PROJECT_DIR / "results" / "towncentre_three_roi" / "full_tracking_1s.csv"
)
DEFAULT_VIDEO = PROJECT_DIR.parent / "archive" / "TownCentreXVID.mp4"
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "towncentre_three_rois.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "towncentre_three_roi"

BASE_FEATURE_NAMES = (
    "current_count",
    "projected_count",
    "previous_count",
    "count_delta",
    "inflow",
    "outflow",
    "projected_inflow",
    "projected_outflow",
    "new_entries",
    "disappeared",
    "mean_confidence",
    "mean_speed",
    "mean_velocity_x",
    "mean_velocity_y",
    "predicted_ratio",
    "mean_bbox_area",
)


@dataclass(frozen=True)
class RoiZone:
    zone_id: int
    name: str
    color_bgr: tuple[int, int, int]


@dataclass(frozen=True)
class RoiConfig:
    width: int
    height: int
    left_boundary: tuple[tuple[float, float], ...]
    right_boundary: tuple[tuple[float, float], ...]
    depth_boundary: tuple[tuple[float, float], ...] | None
    zones: tuple[RoiZone, ...]
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SequenceSample:
    source_second: float
    target_second: float
    features: torch.Tensor
    current_counts: torch.Tensor
    projected_counts: torch.Tensor
    target_counts: torch.Tensor
    split: str


@dataclass(frozen=True)
class TrialConfig:
    window: int
    hidden_size: int
    seed: int


class TemporalGraphOccupancyGNN(nn.Module):
    """Graph convolution per time step followed by a per-node GRU."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_size, hidden_size)
        self.neighbor_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.temporal = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = dropout
        # The initial behavior is the strong persistence baseline.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # features: [batch, time, node, feature]
        neighbors = torch.einsum("ij,btjf->btif", adjacency, features)
        spatial = self.self_projection(features) + self.neighbor_projection(neighbors)
        spatial = F.relu(self.spatial_norm(spatial))
        spatial = F.dropout(spatial, p=self.dropout, training=self.training)
        batch, time_steps, node_count, hidden = spatial.shape
        sequences = spatial.permute(0, 2, 1, 3).reshape(
            batch * node_count, time_steps, hidden
        )
        _, state = self.temporal(sequences)
        final_state = state[-1].reshape(batch, node_count, hidden)
        predicted_delta = self.output(final_state).squeeze(-1)
        projected_count = features[:, -1, :, 1]
        return torch.relu(projected_count + predicted_delta)


def load_roi_config(path: Path) -> RoiConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    width = int(raw["frame_width"])
    height = int(raw["frame_height"])

    def boundary(name: str) -> tuple[tuple[float, float], ...]:
        points = tuple(tuple(float(value) for value in point) for point in raw[name])
        if len(points) < 2 or any(len(point) != 2 for point in points):
            raise RuntimeError(f"{name} must contain at least two [x, y] points")
        if points[0][1] != 0.0:
            raise RuntimeError(f"{name} must start at normalized y=0")
        if any(not 0.0 <= value <= 1.0 for point in points for value in point):
            raise RuntimeError(f"{name} values must be normalized to 0..1")
        if any(first[1] >= second[1] for first, second in zip(points, points[1:])):
            raise RuntimeError(f"{name} y coordinates must increase")
        return points

    left_boundary = boundary("left_boundary")
    right_boundary = boundary("right_boundary")
    depth_boundary: tuple[tuple[float, float], ...] | None = None
    if "depth_boundary" in raw:
        depth_boundary = tuple(
            tuple(float(value) for value in point)
            for point in raw["depth_boundary"]
        )
        if len(depth_boundary) < 2 or any(
            len(point) != 2 for point in depth_boundary
        ):
            raise RuntimeError("depth_boundary must contain at least two [x, y] points")
        if depth_boundary[0][0] != 0.0 or depth_boundary[-1][0] != 1.0:
            raise RuntimeError("depth_boundary must span normalized x=0 through x=1")
        if any(
            not 0.0 <= value <= 1.0
            for point in depth_boundary
            for value in point
        ):
            raise RuntimeError("depth_boundary values must be normalized to 0..1")
        if any(
            first[0] >= second[0]
            for first, second in zip(depth_boundary, depth_boundary[1:])
        ):
            raise RuntimeError("depth_boundary x coordinates must increase")
    elif "depth_split_y" in raw:
        split_y = float(raw["depth_split_y"])
        if not 0.0 < split_y < 1.0:
            raise RuntimeError("depth_split_y must be between normalized y=0 and y=1")
        depth_boundary = ((0.0, split_y), (1.0, split_y))
    zones = tuple(
        RoiZone(
            zone_id=int(item["id"]),
            name=str(item["name"]),
            color_bgr=tuple(int(value) for value in item["color_bgr"]),
        )
        for item in raw["zones"]
    )
    expected_zone_count = 6 if depth_boundary is not None else 3
    expected_ids = list(range(expected_zone_count))
    if [zone.zone_id for zone in zones] != expected_ids:
        raise RuntimeError(
            f"Expected {expected_zone_count} zones with consecutive IDs {expected_ids}"
        )
    edges = tuple(tuple(int(value) for value in edge) for edge in raw["graph_edges"])
    if any(
        len(edge) != 2
        or edge[0] == edge[1]
        or min(edge) < 0
        or max(edge) >= expected_zone_count
        for edge in edges
    ):
        raise RuntimeError("graph_edges contains an invalid node pair")
    config = RoiConfig(
        width=width,
        height=height,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        depth_boundary=depth_boundary,
        zones=zones,
        edges=edges,
    )
    for y in np.linspace(0.0, 1.0, 101):
        if interpolate_boundary(left_boundary, float(y)) >= interpolate_boundary(
            right_boundary, float(y)
        ):
            raise RuntimeError(f"ROI boundaries cross at normalized y={y:.2f}")
    return config


def interpolate_boundary(
    points: Sequence[tuple[float, float]], y: float
) -> float:
    y = min(1.0, max(0.0, y))
    for first, second in zip(points, points[1:]):
        if first[1] <= y <= second[1]:
            span = second[1] - first[1]
            ratio = (y - first[1]) / span if span else 0.0
            return first[0] + ratio * (second[0] - first[0])
    return points[-1][0]


def interpolate_depth_boundary(
    points: Sequence[tuple[float, float]], x: float
) -> float:
    x = min(1.0, max(0.0, x))
    for first, second in zip(points, points[1:]):
        if first[0] <= x <= second[0]:
            span = second[0] - first[0]
            ratio = (x - first[0]) / span if span else 0.0
            return first[1] + ratio * (second[1] - first[1])
    return points[-1][1]


def read_tracking_snapshots(
    path: Path,
) -> dict[float, dict[int, dict[str, str]]]:
    required = {
        "sample_second",
        "track_id",
        "confidence",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "speed_px_s",
        "tracking_state",
    }
    snapshots: dict[float, dict[int, dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Tracking CSV is missing columns: {sorted(missing)}")
        for row in reader:
            second = float(row["sample_second"])
            track_id = int(row["track_id"])
            by_id = snapshots.setdefault(second, {})
            existing = by_id.get(track_id)
            if existing is None or float(row["confidence"]) > float(
                existing["confidence"]
            ):
                by_id[track_id] = row
    if len(snapshots) < 30:
        raise RuntimeError("At least 30 non-empty tracking snapshots are required")
    return dict(sorted(snapshots.items()))


def assignment_point(row: dict[str, str], config: RoiConfig) -> tuple[float, float]:
    x = float(row["bbox_x"]) + float(row["bbox_width"]) / 2.0
    y = float(row["bbox_y"]) + float(row["bbox_height"])
    return (
        min(1.0, max(0.0, x / config.width)),
        min(1.0, max(0.0, y / config.height)),
    )


def zone_for_row(row: dict[str, str], config: RoiConfig) -> int:
    x, y = assignment_point(row, config)
    return zone_for_point(x, y, config)


def zone_for_point(x: float, y: float, config: RoiConfig) -> int:
    left = interpolate_boundary(config.left_boundary, y)
    right = interpolate_boundary(config.right_boundary, y)
    if x < left:
        column = 0
    elif x <= right:
        column = 1
    else:
        column = 2
    if config.depth_boundary is None:
        return column
    split_y = interpolate_depth_boundary(config.depth_boundary, x)
    depth = 0 if y < split_y else 1
    return depth * 3 + column


def assign_snapshot(
    rows: dict[int, dict[str, str]], config: RoiConfig
) -> dict[int, int]:
    return {track_id: zone_for_row(row, config) for track_id, row in rows.items()}


def build_node_statistics(
    snapshots: dict[float, dict[int, dict[str, str]]], config: RoiConfig
) -> dict[float, list[dict[str, float | int]]]:
    previous_assignments: dict[int, int] = {}
    previous_points: dict[int, tuple[float, float]] = {}
    diagonal = math.hypot(config.width, config.height)
    frame_area = float(config.width * config.height)
    by_time: dict[float, list[dict[str, float | int]]] = {}
    for second, rows in snapshots.items():
        assignments = assign_snapshot(rows, config)
        current_points = {
            track_id: assignment_point(row, config) for track_id, row in rows.items()
        }
        projected_assignments: dict[int, int] = {}
        velocities: dict[int, tuple[float, float]] = {}
        for track_id, (x, y) in current_points.items():
            previous_point = previous_points.get(track_id)
            if previous_point is None:
                velocity_x = 0.0
                velocity_y = 0.0
            else:
                velocity_x = x - previous_point[0]
                velocity_y = y - previous_point[1]
            velocities[track_id] = (
                velocity_x * config.width,
                velocity_y * config.height,
            )
            projected_x = min(1.0, max(0.0, x + velocity_x))
            projected_y = min(1.0, max(0.0, y + velocity_y))
            projected_assignments[track_id] = zone_for_point(
                projected_x, projected_y, config
            )
        node_stats: list[dict[str, float | int]] = []
        for zone in config.zones:
            current_ids = {
                track_id
                for track_id, zone_id in assignments.items()
                if zone_id == zone.zone_id
            }
            previous_ids = {
                track_id
                for track_id, zone_id in previous_assignments.items()
                if zone_id == zone.zone_id
            }
            members = [rows[track_id] for track_id in current_ids]
            count = len(members)
            projected_ids = {
                track_id
                for track_id, zone_id in projected_assignments.items()
                if zone_id == zone.zone_id
            }
            inflow = sum(
                track_id in previous_assignments
                and previous_assignments[track_id] != zone.zone_id
                for track_id in current_ids
            )
            outflow = sum(
                track_id in assignments and assignments[track_id] != zone.zone_id
                for track_id in previous_ids
            )
            mean_confidence = (
                sum(float(row["confidence"]) for row in members) / count
                if count
                else 0.0
            )
            mean_speed = (
                sum(float(row["speed_px_s"]) for row in members) / count
                if count
                else 0.0
            )
            predicted_ratio = (
                sum(row["tracking_state"] == "predicted" for row in members) / count
                if count
                else 0.0
            )
            mean_bbox_area = (
                sum(
                    float(row["bbox_width"]) * float(row["bbox_height"])
                    for row in members
                )
                / count
                if count
                else 0.0
            )
            mean_velocity_x = (
                sum(velocities[track_id][0] for track_id in current_ids) / count
                if count
                else 0.0
            )
            mean_velocity_y = (
                sum(velocities[track_id][1] for track_id in current_ids) / count
                if count
                else 0.0
            )
            node_stats.append(
                {
                    "current_count": count,
                    "projected_count": len(projected_ids),
                    "previous_count": len(previous_ids),
                    "count_delta": count - len(previous_ids),
                    "inflow": inflow,
                    "outflow": outflow,
                    "projected_inflow": sum(
                        assignments[track_id] != zone.zone_id
                        for track_id in projected_ids
                    ),
                    "projected_outflow": sum(
                        projected_assignments[track_id] != zone.zone_id
                        for track_id in current_ids
                    ),
                    "new_entries": sum(
                        track_id not in previous_assignments for track_id in current_ids
                    ),
                    "disappeared": sum(
                        track_id not in assignments for track_id in previous_ids
                    ),
                    "mean_confidence": mean_confidence,
                    "mean_speed": mean_speed,
                    "mean_speed_normalized": min(2.0, mean_speed / diagonal),
                    "mean_velocity_x": mean_velocity_x,
                    "mean_velocity_y": mean_velocity_y,
                    "mean_velocity_x_normalized": max(
                        -1.0, min(1.0, mean_velocity_x / config.width)
                    ),
                    "mean_velocity_y_normalized": max(
                        -1.0, min(1.0, mean_velocity_y / config.height)
                    ),
                    "predicted_ratio": predicted_ratio,
                    "mean_bbox_area": mean_bbox_area,
                    "mean_bbox_area_normalized": min(
                        1.0, mean_bbox_area / frame_area
                    ),
                }
            )
        by_time[second] = node_stats
        previous_assignments = assignments
        previous_points = current_points
    return by_time


def split_boundaries(times: Sequence[float]) -> tuple[float, float]:
    train_index = max(1, round(len(times) * 0.60)) - 1
    validation_index = max(train_index + 1, round(len(times) * 0.80)) - 1
    validation_index = min(validation_index, len(times) - 2)
    return times[train_index], times[validation_index]


def split_for_target(target_second: float, train_end: float, validation_end: float) -> str:
    if target_second <= train_end:
        return "train"
    if target_second <= validation_end:
        return "validation"
    return "test"


def feature_tensor(
    node_stats: Sequence[dict[str, float | int]], count_scale: float
) -> torch.Tensor:
    zone_count = len(node_stats)
    features: list[list[float]] = []
    for zone_id, stats in enumerate(node_stats):
        one_hot = [0.0] * zone_count
        one_hot[zone_id] = 1.0
        features.append(
            [
                float(stats["current_count"]) / count_scale,
                float(stats["projected_count"]) / count_scale,
                float(stats["previous_count"]) / count_scale,
                float(stats["count_delta"]) / count_scale,
                float(stats["inflow"]) / count_scale,
                float(stats["outflow"]) / count_scale,
                float(stats["projected_inflow"]) / count_scale,
                float(stats["projected_outflow"]) / count_scale,
                float(stats["new_entries"]) / count_scale,
                float(stats["disappeared"]) / count_scale,
                float(stats["mean_confidence"]),
                float(stats["mean_speed_normalized"]),
                float(stats["mean_velocity_x_normalized"]),
                float(stats["mean_velocity_y_normalized"]),
                float(stats["predicted_ratio"]),
                float(stats["mean_bbox_area_normalized"]),
            ]
            + one_hot
        )
    return torch.tensor(features, dtype=torch.float32)


def make_sequences(
    stats_by_time: dict[float, list[dict[str, float | int]]],
    window: int,
    count_scale: float,
    train_end: float,
    validation_end: float,
) -> list[SequenceSample]:
    times = list(stats_by_time)
    encoded = {
        second: feature_tensor(stats, count_scale)
        for second, stats in stats_by_time.items()
    }
    samples: list[SequenceSample] = []
    for target_index in range(window, len(times)):
        window_times = times[target_index - window : target_index]
        target_second = times[target_index]
        complete_times = window_times + [target_second]
        if any(
            not 0.95 <= later - earlier <= 1.05
            for earlier, later in zip(complete_times, complete_times[1:])
        ):
            continue
        source_second = window_times[-1]
        current_counts = torch.tensor(
            [
                float(item["current_count"])
                for item in stats_by_time[source_second]
            ],
            dtype=torch.float32,
        )
        projected_counts = torch.tensor(
            [
                float(item["projected_count"])
                for item in stats_by_time[source_second]
            ],
            dtype=torch.float32,
        )
        target_counts = torch.tensor(
            [
                float(item["current_count"])
                for item in stats_by_time[target_second]
            ],
            dtype=torch.float32,
        )
        samples.append(
            SequenceSample(
                source_second=source_second,
                target_second=target_second,
                features=torch.stack([encoded[second] for second in window_times]),
                current_counts=current_counts,
                projected_counts=projected_counts,
                target_counts=target_counts,
                split=split_for_target(target_second, train_end, validation_end),
            )
        )
    return samples


def normalized_adjacency(config: RoiConfig) -> torch.Tensor:
    adjacency = torch.zeros((len(config.zones), len(config.zones)), dtype=torch.float32)
    for first, second in config.edges:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stack_samples(
    samples: Sequence[SequenceSample], count_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise RuntimeError("Cannot stack an empty sample set")
    return (
        torch.stack([sample.features for sample in samples]),
        torch.stack([sample.target_counts for sample in samples]) / count_scale,
    )


def model_predictions(
    model: TemporalGraphOccupancyGNN,
    samples: Sequence[SequenceSample],
    adjacency: torch.Tensor,
    count_scale: float,
) -> np.ndarray:
    if not samples:
        return np.empty((0, adjacency.shape[0]), dtype=np.float32)
    features, _ = stack_samples(samples, count_scale)
    model.eval()
    with torch.no_grad():
        predictions = model(features, adjacency) * count_scale
    return predictions.cpu().numpy()


def continuous_mae(
    model: TemporalGraphOccupancyGNN,
    samples: Sequence[SequenceSample],
    adjacency: torch.Tensor,
    count_scale: float,
) -> float:
    predictions = model_predictions(model, samples, adjacency, count_scale)
    targets = np.stack([sample.target_counts.numpy() for sample in samples])
    return float(np.abs(predictions - targets).mean())


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_with_validation(
    trial: TrialConfig,
    samples: Sequence[SequenceSample],
    adjacency: torch.Tensor,
    input_size: int,
    count_scale: float,
    dropout: float,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[TemporalGraphOccupancyGNN, int, float]:
    seed_everything(trial.seed)
    training = [sample for sample in samples if sample.split == "train"]
    validation = [sample for sample in samples if sample.split == "validation"]
    train_x, train_y = stack_samples(training, count_scale)
    model = TemporalGraphOccupancyGNN(input_size, trial.hidden_size, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0005
    )
    generator = torch.Generator().manual_seed(trial.seed)
    batch_size = min(32, len(training))
    best_state = clone_state_dict(model)
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(training), generator=generator)
        for start in range(0, len(training), batch_size):
            indices = order[start : start + batch_size]
            predicted = model(train_x[indices], adjacency)
            target = train_y[indices]
            node_loss = F.smooth_l1_loss(predicted, target, beta=0.08)
            total_loss = F.smooth_l1_loss(
                predicted.sum(dim=1), target.sum(dim=1), beta=0.12
            )
            loss = node_loss + 0.15 * total_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        validation_mae = continuous_mae(
            model, validation, adjacency, count_scale
        )
        if validation_mae + 1e-5 < best_validation:
            best_validation = validation_mae
            best_epoch = epoch
            best_state = clone_state_dict(model)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_validation


def fit_fixed_epochs(
    trial: TrialConfig,
    samples: Sequence[SequenceSample],
    adjacency: torch.Tensor,
    input_size: int,
    count_scale: float,
    dropout: float,
    epochs: int,
    learning_rate: float,
) -> TemporalGraphOccupancyGNN:
    seed_everything(trial.seed)
    train_validation = [sample for sample in samples if sample.split != "test"]
    features, targets = stack_samples(train_validation, count_scale)
    model = TemporalGraphOccupancyGNN(input_size, trial.hidden_size, dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0005
    )
    generator = torch.Generator().manual_seed(trial.seed)
    batch_size = min(32, len(train_validation))
    for _ in range(max(1, epochs)):
        model.train()
        order = torch.randperm(len(train_validation), generator=generator)
        for start in range(0, len(train_validation), batch_size):
            indices = order[start : start + batch_size]
            predicted = model(features[indices], adjacency)
            target = targets[indices]
            node_loss = F.smooth_l1_loss(predicted, target, beta=0.08)
            total_loss = F.smooth_l1_loss(
                predicted.sum(dim=1), target.sum(dim=1), beta=0.12
            )
            loss = node_loss + 0.15 * total_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
    return model


def regression_metrics(
    predictions: np.ndarray,
    samples: Sequence[SequenceSample],
    zone_id: int | None = None,
) -> dict[str, float | int]:
    targets = np.stack([sample.target_counts.numpy() for sample in samples])
    current = np.stack([sample.current_counts.numpy() for sample in samples])
    projected = np.stack([sample.projected_counts.numpy() for sample in samples])
    if zone_id is not None:
        predictions = predictions[:, zone_id : zone_id + 1]
        targets = targets[:, zone_id : zone_id + 1]
        current = current[:, zone_id : zone_id + 1]
        projected = projected[:, zone_id : zone_id + 1]
    errors = predictions - targets
    rounded = np.rint(predictions).clip(min=0)
    rounded_errors = rounded - targets
    baseline_errors = current - targets
    projection_errors = projected - targets
    result: dict[str, float | int] = {
        "samples": int(targets.size),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "rounded_mae": float(np.abs(rounded_errors).mean()),
        "exact_accuracy": float((rounded_errors == 0).mean()),
        "within_one_accuracy": float((np.abs(rounded_errors) <= 1).mean()),
        "persistence_mae": float(np.abs(baseline_errors).mean()),
        "projection_mae": float(np.abs(projection_errors).mean()),
        "persistence_exact_accuracy": float((baseline_errors == 0).mean()),
        "persistence_within_one_accuracy": float(
            (np.abs(baseline_errors) <= 1).mean()
        ),
        "projection_exact_accuracy": float((projection_errors == 0).mean()),
        "projection_within_one_accuracy": float(
            (np.abs(projection_errors) <= 1).mean()
        ),
    }
    if zone_id is None:
        result["total_count_mae"] = float(
            np.abs(predictions.sum(axis=1) - targets.sum(axis=1)).mean()
        )
        result["persistence_total_count_mae"] = float(
            np.abs(current.sum(axis=1) - targets.sum(axis=1)).mean()
        )
        result["projection_total_count_mae"] = float(
            np.abs(projected.sum(axis=1) - targets.sum(axis=1)).mean()
        )
    return result


def write_dataset_csv(
    path: Path,
    stats_by_time: dict[float, list[dict[str, float | int]]],
    config: RoiConfig,
    train_end: float,
    validation_end: float,
) -> None:
    times = list(stats_by_time)
    fields = [
        "sample_second",
        "zone_id",
        "zone",
        *BASE_FEATURE_NAMES,
        "target_second",
        "target_next_count",
        "data_split",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for time_index, second in enumerate(times):
            target_second = times[time_index + 1] if time_index + 1 < len(times) else None
            target_stats = (
                stats_by_time[target_second] if target_second is not None else None
            )
            split = (
                split_for_target(target_second, train_end, validation_end)
                if target_second is not None
                else "forecast"
            )
            for zone in config.zones:
                stats = stats_by_time[second][zone.zone_id]
                writer.writerow(
                    {
                        "sample_second": second,
                        "zone_id": zone.zone_id,
                        "zone": zone.name,
                        **{
                            feature: round(float(stats[feature]), 6)
                            for feature in BASE_FEATURE_NAMES
                        },
                        "target_second": target_second if target_second is not None else "",
                        "target_next_count": (
                            int(target_stats[zone.zone_id]["current_count"])
                            if target_stats is not None
                            else ""
                        ),
                        "data_split": split,
                    }
                )


def prediction_rows(
    model: TemporalGraphOccupancyGNN,
    samples: Sequence[SequenceSample],
    adjacency: torch.Tensor,
    count_scale: float,
    config: RoiConfig,
) -> list[dict[str, float | int | str]]:
    predictions = model_predictions(model, samples, adjacency, count_scale)
    rows: list[dict[str, float | int | str]] = []
    for sample, predicted in zip(samples, predictions):
        for zone in config.zones:
            actual = float(sample.target_counts[zone.zone_id])
            prediction = max(0.0, float(predicted[zone.zone_id]))
            rounded = round(prediction)
            current = int(sample.current_counts[zone.zone_id])
            projected = int(sample.projected_counts[zone.zone_id])
            rows.append(
                {
                    "data_split": sample.split,
                    "source_second": sample.source_second,
                    "target_second": sample.target_second,
                    "zone_id": zone.zone_id,
                    "zone": zone.name,
                    "current_count": current,
                    "projected_count": projected,
                    "actual_next_count": int(actual),
                    "predicted_next_count": round(prediction, 4),
                    "predicted_rounded_count": rounded,
                    "absolute_error": round(abs(prediction - actual), 4),
                    "persistence_absolute_error": abs(current - actual),
                    "projection_absolute_error": abs(projected - actual),
                }
            )
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def boundary_pixels(
    points: Sequence[tuple[float, float]], width: int, height: int
) -> np.ndarray:
    return np.array(
        [[round(x * width), round(y * height)] for x, y in points], dtype=np.int32
    )


def roi_polygons(config: RoiConfig) -> list[np.ndarray]:
    left = boundary_pixels(config.left_boundary, config.width, config.height)
    right = boundary_pixels(config.right_boundary, config.width, config.height)
    if config.depth_boundary is None:
        return [
            np.vstack(([[0, 0]], left, [[0, config.height]])),
            np.vstack((left, right[::-1])),
            np.vstack((right, [[config.width, config.height]], [[config.width, 0]])),
        ]

    y_values = np.arange(config.height, dtype=np.float32) / config.height
    left_by_y = np.interp(
        y_values,
        [point[1] for point in config.left_boundary],
        [point[0] for point in config.left_boundary],
    ) * config.width
    right_by_y = np.interp(
        y_values,
        [point[1] for point in config.right_boundary],
        [point[0] for point in config.right_boundary],
    ) * config.width
    x_pixels = np.arange(config.width, dtype=np.float32)[None, :]
    columns = np.where(
        x_pixels < left_by_y[:, None],
        0,
        np.where(x_pixels <= right_by_y[:, None], 1, 2),
    ).astype(np.uint8)
    x_values = np.arange(config.width, dtype=np.float32) / config.width
    split_by_x = np.interp(
        x_values,
        [point[0] for point in config.depth_boundary],
        [point[1] for point in config.depth_boundary],
    ) * config.height
    depths = (
        np.arange(config.height, dtype=np.float32)[:, None] >= split_by_x[None, :]
    ).astype(np.uint8)
    zone_map = depths * 3 + columns
    polygons: list[np.ndarray] = []
    for zone in config.zones:
        mask = np.where(zone_map == zone.zone_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError(f"ROI Z{zone.zone_id} has no visible polygon")
        polygons.append(max(contours, key=cv2.contourArea).reshape(-1, 2))
    return polygons


def make_roi_review(
    path: Path,
    video_path: Path,
    preview_second: float,
    config: RoiConfig,
) -> None:
    """Draw only the proposed ROI geometry; this does not build data or train."""
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, preview_second * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read ROI review frame at {preview_second}s")
    if (frame.shape[1], frame.shape[0]) != (config.width, config.height):
        raise RuntimeError("Review video resolution does not match ROI configuration")

    polygons = roi_polygons(config)
    overlay = frame.copy()
    for zone, polygon in zip(config.zones, polygons):
        cv2.fillPoly(overlay, [polygon], zone.color_bgr)
    frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0.0)

    review_color = (0, 215, 255)
    left = boundary_pixels(config.left_boundary, config.width, config.height)
    right = boundary_pixels(config.right_boundary, config.width, config.height)
    cv2.polylines(frame, [left], False, review_color, 6, cv2.LINE_AA)
    cv2.polylines(frame, [right], False, review_color, 6, cv2.LINE_AA)
    if config.depth_boundary is not None:
        depth = boundary_pixels(config.depth_boundary, config.width, config.height)
        cv2.polylines(frame, [depth], False, review_color, 6, cv2.LINE_AA)

    for zone, polygon in zip(config.zones, polygons):
        moments = cv2.moments(polygon)
        if moments["m00"] == 0:
            continue
        center_x = round(moments["m10"] / moments["m00"])
        center_y = round(moments["m01"] / moments["m00"])
        label = f"Z{zone.zone_id} {zone.name}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
        )
        origin_x = max(8, min(config.width - text_width - 8, center_x - text_width // 2))
        origin_y = max(
            text_height + 8,
            min(config.height - 8, center_y + text_height // 2),
        )
        cv2.rectangle(
            frame,
            (origin_x - 7, origin_y - text_height - 7),
            (origin_x + text_width + 7, origin_y + 7),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (origin_x, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), frame)


def make_roi_preview(
    path: Path,
    video_path: Path,
    preview_second: float,
    snapshots: dict[float, dict[int, dict[str, str]]],
    config: RoiConfig,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, preview_second * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read preview frame at {preview_second}s")
    if (frame.shape[1], frame.shape[0]) != (config.width, config.height):
        raise RuntimeError("Preview video resolution does not match ROI configuration")
    second = min(snapshots, key=lambda value: abs(value - preview_second))
    rows = snapshots[second]
    assignments = assign_snapshot(rows, config)
    counts = [
        sum(zone_id == index for zone_id in assignments.values())
        for index in range(len(config.zones))
    ]
    overlay = frame.copy()
    polygons = roi_polygons(config)
    for zone, polygon in zip(config.zones, polygons):
        cv2.fillPoly(overlay, [polygon], zone.color_bgr)
    frame = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0.0)
    left = boundary_pixels(config.left_boundary, config.width, config.height)
    right = boundary_pixels(config.right_boundary, config.width, config.height)
    boundary_color = (0, 255, 255)
    cv2.polylines(frame, [left], False, boundary_color, 3, cv2.LINE_AA)
    cv2.polylines(frame, [right], False, boundary_color, 3, cv2.LINE_AA)
    if config.depth_boundary is not None:
        depth = boundary_pixels(config.depth_boundary, config.width, config.height)
        cv2.polylines(frame, [depth], False, boundary_color, 3, cv2.LINE_AA)
    for track_id, row in rows.items():
        zone_id = assignments[track_id]
        color = config.zones[zone_id].color_bgr
        x1 = round(float(row["bbox_x"]))
        y1 = round(float(row["bbox_y"]))
        x2 = round(x1 + float(row["bbox_width"]))
        y2 = round(y1 + float(row["bbox_height"]))
        point_x, point_y = assignment_point(row, config)
        foot = (round(point_x * config.width), round(point_y * config.height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, foot, 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(track_id),
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    rows_per_header = 3
    header_rows = math.ceil(len(config.zones) / rows_per_header)
    header_height = 18 + 35 * header_rows
    cv2.rectangle(frame, (0, 0), (config.width, header_height), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"{second:.0f}s | foot-point ROI assignment",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for row_index in range(header_rows):
        row_zones = config.zones[
            row_index * rows_per_header : (row_index + 1) * rows_per_header
        ]
        labels = " | ".join(
            f"Z{zone.zone_id} {zone.name}: {counts[zone.zone_id]}"
            for zone in row_zones
        )
        cv2.putText(
            frame,
            labels,
            (430, 30 + 35 * row_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), frame)


def make_prediction_chart(
    path: Path,
    rows: Sequence[dict[str, float | int | str]],
    config: RoiConfig,
) -> None:
    test_rows = [row for row in rows if row["data_split"] == "test"]
    if len(config.zones) == 6:
        figure, axes_grid = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        axes = list(axes_grid.flat)
    else:
        figure, axes_grid = plt.subplots(len(config.zones), 1, figsize=(13, 10), sharex=True)
        axes = list(np.atleast_1d(axes_grid).flat)
    for zone, axis in zip(config.zones, axes):
        zone_rows = [row for row in test_rows if row["zone_id"] == zone.zone_id]
        seconds = [float(row["target_second"]) for row in zone_rows]
        actual = [float(row["actual_next_count"]) for row in zone_rows]
        predicted = [float(row["predicted_next_count"]) for row in zone_rows]
        persistence = [float(row["current_count"]) for row in zone_rows]
        axis.plot(seconds, actual, color="black", linewidth=2.2, label="tracked target")
        axis.plot(seconds, predicted, color="#d6279f", linewidth=1.9, label="ST-GNN")
        axis.plot(
            seconds,
            persistence,
            color="#7f7f7f",
            linewidth=1.2,
            linestyle="--",
            label="persistence",
        )
        axis.set_title(f"Z{zone.zone_id} {zone.name}")
        axis.set_ylabel("tracked people")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    for axis in axes:
        axis.set_xlabel("target video time (seconds)")
    figure.suptitle(
        f"{len(config.zones)}-ROI next-second occupancy: held-out final 20%",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def make_count_distribution_chart(
    path: Path,
    stats_by_time: dict[float, list[dict[str, float | int]]],
    config: RoiConfig,
) -> None:
    times = list(stats_by_time)
    figure, axis = plt.subplots(figsize=(13, 5))
    for zone in config.zones:
        counts = [
            int(stats_by_time[second][zone.zone_id]["current_count"])
            for second in times
        ]
        rgb = tuple(channel / 255.0 for channel in zone.color_bgr[::-1])
        axis.plot(times, counts, label=zone.name, color=rgb, linewidth=1.4)
    axis.set_title("Full five-minute tracked occupancy by ROI")
    axis.set_xlabel("video time (seconds)")
    axis.set_ylabel("tracked people")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("A comma-separated list of positive integers is required")
    return parsed


def write_report(
    path: Path,
    summary: dict[str, object],
    metrics_rows: Sequence[dict[str, object]],
) -> None:
    test_all = next(
        row
        for row in metrics_rows
        if row["data_split"] == "test" and row["zone"] == "all"
    )
    improvement = 100.0 * (
        float(test_all["persistence_mae"]) - float(test_all["mae"])
    ) / max(float(test_all["persistence_mae"]), 1e-9)
    lines = [
        f"# Town Centre {summary['zone_count']}-ROI 시공간 GNN 결과",
        "",
        "## 실험 범위",
        "",
        f"- 전체 영상: {summary['video_duration_s']:.1f}초, "
        f"{summary['tracking_snapshots']}개 1초 스냅샷",
        f"- 추적 관측: {summary['tracking_rows']:,}행, "
        f"고유 추적 ID {summary['unique_track_ids']}개",
        f"- 노드({summary['zone_count']}개): "
        + ", ".join(str(name) for name in summary["zone_names"]),
        "- 사람 배정 기준: 경계 박스 아래 중앙점(발 위치)",
        f"- 최종 모델: {summary['selected_window']}초 입력, "
        f"hidden {summary['selected_hidden_size']}, seed {summary['selected_seed']}",
        "- 검증이 끝난 설정으로 앞 80%를 재학습하고 마지막 20%를 테스트",
        "",
        "## 마지막 20% 테스트 결과",
        "",
        "| 범위 | MAE | RMSE | 반올림 정확도 | ±1명 정확도 | 현재값 유지 MAE | 이동 외삽 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        if row["data_split"] != "test":
            continue
        lines.append(
            f"| {row['zone']} | {float(row['mae']):.3f} | "
            f"{float(row['rmse']):.3f} | {100*float(row['exact_accuracy']):.1f}% | "
            f"{100*float(row['within_one_accuracy']):.1f}% | "
            f"{float(row['persistence_mae']):.3f} | "
            f"{float(row['projection_mae']):.3f} |"
        )
    lines.extend(
        [
            "",
            f"전체 노드 MAE는 현재값 유지 기준보다 **{improvement:.1f}%** 개선됐습니다.",
            "",
            "## 전체 노드 기준선 비교",
            "",
            "| 방법 | MAE | 완전일치율 | ±1명 이내 |",
            "|---|---:|---:|---:|",
            f"| ST-GNN | {float(test_all['mae']):.3f} | "
            f"{100*float(test_all['exact_accuracy']):.1f}% | "
            f"{100*float(test_all['within_one_accuracy']):.1f}% |",
            f"| 현재값 유지 | {float(test_all['persistence_mae']):.3f} | "
            f"{100*float(test_all['persistence_exact_accuracy']):.1f}% | "
            f"{100*float(test_all['persistence_within_one_accuracy']):.1f}% |",
            f"| 이동 외삽 | {float(test_all['projection_mae']):.3f} | "
            f"{100*float(test_all['projection_exact_accuracy']):.1f}% | "
            f"{100*float(test_all['projection_within_one_accuracy']):.1f}% |",
            "",
            "> 이 정확도는 사람 검출 자체의 정답 정확도가 아니라, OpenCV/YOLO 추적기가 "
            "만든 인원 시계열의 1초 뒤 값을 얼마나 잘 예측했는지 나타냅니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-csv", type=Path, default=DEFAULT_TRACKING_CSV)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows", type=parse_int_list, default=parse_int_list("4,8,12"))
    parser.add_argument(
        "--hidden-sizes", type=parse_int_list, default=parse_int_list("16,32")
    )
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("17,31"))
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--preview-second", type=float, default=248.0)
    parser.add_argument(
        "--roi-review-only",
        action="store_true",
        help="draw the proposed ROI image without creating a dataset or training",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("--epochs and --patience must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_roi_config(args.config)
    roi_word = {3: "three", 6: "six"}.get(len(config.zones), str(len(config.zones)))
    artifact_prefix = f"{roi_word}_roi"
    if args.roi_review_only:
        review_path = args.output_dir / f"{artifact_prefix}_review.jpg"
        make_roi_review(review_path, args.video, args.preview_second, config)
        print(f"ROI review image: {review_path.resolve()}")
        return 0
    snapshots = read_tracking_snapshots(args.tracking_csv)
    times = list(snapshots)
    stats_by_time = build_node_statistics(snapshots, config)
    train_end, validation_end = split_boundaries(times)
    count_scale = float(
        max(
            10,
            *(
                int(node["current_count"])
                for second, nodes in stats_by_time.items()
                if second <= train_end
                for node in nodes
            ),
        )
    )
    write_dataset_csv(
        args.output_dir / f"{artifact_prefix}_dataset.csv",
        stats_by_time,
        config,
        train_end,
        validation_end,
    )
    adjacency = normalized_adjacency(config)
    input_size = len(BASE_FEATURE_NAMES) + len(config.zones)

    trials: list[dict[str, object]] = []
    best: tuple[
        float,
        TrialConfig,
        int,
        list[SequenceSample],
        TemporalGraphOccupancyGNN,
    ] | None = None
    for window in args.windows:
        samples = make_sequences(
            stats_by_time,
            window,
            count_scale,
            train_end,
            validation_end,
        )
        for hidden_size in args.hidden_sizes:
            for seed in args.seeds:
                trial = TrialConfig(window, hidden_size, seed)
                model, best_epoch, validation_mae = train_with_validation(
                    trial,
                    samples,
                    adjacency,
                    input_size,
                    count_scale,
                    args.dropout,
                    args.epochs,
                    args.patience,
                    args.learning_rate,
                )
                trial_row = {
                    "window": window,
                    "hidden_size": hidden_size,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_mae": round(validation_mae, 6),
                }
                trials.append(trial_row)
                print(
                    f"window={window:2d} hidden={hidden_size:2d} seed={seed:2d} | "
                    f"epoch={best_epoch:3d} validation MAE={validation_mae:.4f}"
                )
                if best is None or validation_mae < best[0]:
                    best = (validation_mae, trial, best_epoch, samples, model)
    assert best is not None
    _, selected_trial, selected_epoch, selected_samples, _ = best
    # Standard final fit: hyperparameters are chosen on validation, then the model is
    # refit on train+validation. The final 20% remains completely unseen.
    final_model = fit_fixed_epochs(
        selected_trial,
        selected_samples,
        adjacency,
        input_size,
        count_scale,
        args.dropout,
        selected_epoch,
        args.learning_rate,
    )

    predictions = prediction_rows(
        final_model, selected_samples, adjacency, count_scale, config
    )
    write_rows(args.output_dir / "predictions.csv", predictions)
    write_rows(args.output_dir / "validation_trials.csv", trials)

    metrics_rows: list[dict[str, object]] = []
    for split in ("train_validation", "test"):
        split_samples = [
            sample
            for sample in selected_samples
            if (sample.split != "test") == (split == "train_validation")
        ]
        split_predictions = model_predictions(
            final_model, split_samples, adjacency, count_scale
        )
        groups = [("all", None)] + [
            (zone.name, zone.zone_id) for zone in config.zones
        ]
        for zone_name, zone_id in groups:
            metrics_rows.append(
                {
                    "data_split": split,
                    "zone": zone_name,
                    **{
                        key: round(value, 6) if isinstance(value, float) else value
                        for key, value in regression_metrics(
                            split_predictions, split_samples, zone_id
                        ).items()
                    },
                }
            )
    write_rows(args.output_dir / "metrics.csv", metrics_rows)

    test_samples = [sample for sample in selected_samples if sample.split == "test"]
    test_predictions = model_predictions(
        final_model, test_samples, adjacency, count_scale
    )
    test_metrics = regression_metrics(test_predictions, test_samples)
    checkpoint = {
        "model_type": "TemporalGraphOccupancyGNN",
        "state_dict": final_model.state_dict(),
        "input_size": input_size,
        "hidden_size": selected_trial.hidden_size,
        "dropout": args.dropout,
        "window_seconds": selected_trial.window,
        "horizon_seconds": 1,
        "count_scale": count_scale,
        "feature_names": list(BASE_FEATURE_NAMES)
        + [f"zone_{index}_one_hot" for index in range(len(config.zones))],
        "roi_config": json.loads(args.config.read_text(encoding="utf-8")),
        "selected_seed": selected_trial.seed,
        "training_epochs": selected_epoch,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, args.output_dir / f"{artifact_prefix}_stgnn.pt")

    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    tracking_rows = sum(len(rows) for rows in snapshots.values())
    summary: dict[str, object] = {
        "video_duration_s": video_frames / fps,
        "video_frames": video_frames,
        "video_fps": fps,
        "tracking_snapshots": len(snapshots),
        "tracking_rows": tracking_rows,
        "unique_track_ids": len(
            {track_id for rows in snapshots.values() for track_id in rows}
        ),
        "zone_count": len(config.zones),
        "zone_names": [zone.name for zone in config.zones],
        "train_end_second": train_end,
        "validation_end_second": validation_end,
        "test_start_second": min(sample.target_second for sample in test_samples),
        "test_end_second": max(sample.target_second for sample in test_samples),
        "selected_window": selected_trial.window,
        "selected_hidden_size": selected_trial.hidden_size,
        "selected_seed": selected_trial.seed,
        "selected_epoch": selected_epoch,
        "count_scale": count_scale,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.output_dir / "REPORT.md", summary, metrics_rows)
    make_roi_preview(
        args.output_dir / f"{artifact_prefix}_preview_248s.jpg",
        args.video,
        args.preview_second,
        snapshots,
        config,
    )
    make_prediction_chart(
        args.output_dir / "test_predictions.png", predictions, config
    )
    make_count_distribution_chart(
        args.output_dir / "full_roi_counts.png", stats_by_time, config
    )

    print("\nSelected model")
    print(
        f"window={selected_trial.window}s hidden={selected_trial.hidden_size} "
        f"seed={selected_trial.seed} epochs={selected_epoch}"
    )
    print(
        f"test MAE={float(test_metrics['mae']):.4f}, "
        f"RMSE={float(test_metrics['rmse']):.4f}, "
        f"exact={100*float(test_metrics['exact_accuracy']):.1f}%, "
        f"within ±1={100*float(test_metrics['within_one_accuracy']):.1f}%"
    )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
