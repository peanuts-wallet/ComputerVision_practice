#!/usr/bin/env python3
"""High-frequency, calibrated, flow-conserving ST-GNN for Town Centre."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/towncentre_flow_gnn_matplotlib")

import cv2
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from towncentre_three_roi_gnn import (
    RoiConfig,
    load_roi_config,
    normalized_adjacency,
    seed_everything,
    zone_for_point,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_CSV = (
    PROJECT_DIR / "results" / "towncentre_six_roi_02s" / "full_tracking_02s.csv"
)
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "towncentre_six_rois.json"
DEFAULT_CALIBRATION = PROJECT_DIR.parent / "archive" / "TownCentre-calibration.ci"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "towncentre_six_roi_02s"
DEFAULT_PREVIOUS_SUMMARY = (
    PROJECT_DIR / "results" / "towncentre_six_roi" / "summary.json"
)

BASE_FEATURE_NAMES = (
    "current_count",
    "pixel_projected_count",
    "ground_projected_count",
    "previous_count",
    "count_delta",
    "inflow",
    "outflow",
    "projected_inflow",
    "projected_outflow",
    "new_entries",
    "disappeared",
    "mean_confidence",
    "mean_ground_speed_m_s",
    "mean_ground_velocity_x_m_s",
    "mean_ground_velocity_y_m_s",
    "predicted_ratio",
    "mean_bbox_area_ratio",
)
COUNT_FEATURE_NAMES = {
    "current_count",
    "pixel_projected_count",
    "ground_projected_count",
    "previous_count",
    "count_delta",
    "inflow",
    "outflow",
    "projected_inflow",
    "projected_outflow",
    "new_entries",
    "disappeared",
}


@dataclass(frozen=True)
class GroundCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    world_to_camera: np.ndarray
    camera_center: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    raw: dict[str, float]


@dataclass
class FrameState:
    second: float
    rows: dict[int, dict[str, str]]
    assignments: dict[int, int]
    ground_points: dict[int, np.ndarray]
    velocities: dict[int, np.ndarray]
    projected_assignments: dict[int, int]
    pixel_projected_assignments: dict[int, int]
    node_stats: list[dict[str, float | int]]


@dataclass(frozen=True)
class FlowSample:
    source_second: float
    target_second: float
    features: torch.Tensor
    current_counts: torch.Tensor
    pixel_projected_counts: torch.Tensor
    ground_projected_counts: torch.Tensor
    target_counts: torch.Tensor
    edge_flows: torch.Tensor
    entries: torch.Tensor
    exits: torch.Tensor
    split: str


@dataclass(frozen=True)
class Candidate:
    learning_rate: float
    dropout: float
    seed: int


class FlowConservingSTGNN(nn.Module):
    """Spatial GNN + temporal GRU + explicit edge and boundary flows."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        directed_edges: Sequence[tuple[int, int]],
        dropout: float,
    ) -> None:
        super().__init__()
        self.directed_edges = tuple(directed_edges)
        self.self_projection = nn.Linear(input_size, hidden_size)
        self.neighbor_projection = nn.Linear(input_size, hidden_size, bias=False)
        self.spatial_norm = nn.LayerNorm(hidden_size)
        self.temporal = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.entry_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.exit_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.gate_head = nn.Linear(hidden_size, 1)
        self.dropout = dropout
        for head in (self.edge_head, self.entry_head, self.exit_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, -3.0)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        current_counts: torch.Tensor,
        projected_counts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        neighbors = torch.einsum("ij,btjf->btif", adjacency, features)
        spatial = self.self_projection(features) + self.neighbor_projection(neighbors)
        spatial = F.relu(self.spatial_norm(spatial))
        spatial = F.dropout(spatial, p=self.dropout, training=self.training)
        batch, time_steps, node_count, hidden_size = spatial.shape
        sequences = spatial.permute(0, 2, 1, 3).reshape(
            batch * node_count, time_steps, hidden_size
        )
        _, hidden = self.temporal(sequences)
        node_hidden = hidden[-1].reshape(batch, node_count, hidden_size)

        edge_flows = []
        for source, target in self.directed_edges:
            pair = torch.cat(
                (node_hidden[:, source], node_hidden[:, target]), dim=-1
            )
            edge_flows.append(F.softplus(self.edge_head(pair).squeeze(-1)))
        edge_flow_tensor = torch.stack(edge_flows, dim=1)
        entries = F.softplus(self.entry_head(node_hidden).squeeze(-1))
        exits = F.softplus(self.exit_head(node_hidden).squeeze(-1))

        physical_counts = current_counts + entries - exits
        for edge_index, (source, target) in enumerate(self.directed_edges):
            flow = edge_flow_tensor[:, edge_index]
            source_adjustment = torch.zeros_like(physical_counts)
            target_adjustment = torch.zeros_like(physical_counts)
            source_adjustment[:, source] = flow
            target_adjustment[:, target] = flow
            physical_counts = physical_counts - source_adjustment + target_adjustment
        physical_counts = torch.relu(physical_counts)
        gate = torch.sigmoid(self.gate_head(node_hidden).squeeze(-1))
        prediction = torch.relu(
            projected_counts + gate * (physical_counts - projected_counts)
        )
        return {
            "prediction": prediction,
            "physical_counts": physical_counts,
            "edge_flows": edge_flow_tensor,
            "entries": entries,
            "exits": exits,
            "gate": gate,
        }


def load_calibration(path: Path) -> GroundCalibration:
    raw: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key.strip()] = float(value.strip())
    required = {
        "FocalLengthX",
        "FocalLengthY",
        "PrincipalPointX",
        "PrincipalPointY",
        "Skew",
        "TranslationX",
        "TranslationY",
        "TranslationZ",
        "RotationX",
        "RotationY",
        "RotationZ",
        "RotationW",
        "DistortionK1",
        "DistortionK2",
        "DistortionP1",
        "DistortionP2",
    }
    missing = required.difference(raw)
    if missing:
        raise RuntimeError(f"Calibration is missing values: {sorted(missing)}")
    camera_matrix = np.array(
        [
            [raw["FocalLengthX"], raw["Skew"], raw["PrincipalPointX"]],
            [0.0, raw["FocalLengthY"], raw["PrincipalPointY"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [
            raw["DistortionK1"],
            raw["DistortionK2"],
            raw["DistortionP1"],
            raw["DistortionP2"],
        ],
        dtype=np.float64,
    )
    x, y, z, w = (
        raw["RotationX"],
        raw["RotationY"],
        raw["RotationZ"],
        raw["RotationW"],
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    world_to_camera = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    camera_center = np.array(
        [raw["TranslationX"], raw["TranslationY"], raw["TranslationZ"]],
        dtype=np.float64,
    )
    rotation_vector, _ = cv2.Rodrigues(world_to_camera)
    translation_vector = -world_to_camera @ camera_center.reshape(3, 1)
    return GroundCalibration(
        camera_matrix=camera_matrix,
        distortion=distortion,
        world_to_camera=world_to_camera,
        camera_center=camera_center,
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
        raw=raw,
    )


def image_to_ground(
    pixels: np.ndarray, calibration: GroundCalibration
) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(
        pixels, calibration.camera_matrix, calibration.distortion
    ).reshape(-1, 2)
    camera_rays = np.column_stack((undistorted, np.ones(len(undistorted))))
    world_rays = camera_rays @ calibration.world_to_camera
    denominator = world_rays[:, 2]
    scales = np.divide(
        -calibration.camera_center[2],
        denominator,
        out=np.full_like(denominator, np.nan),
        where=np.abs(denominator) > 1e-9,
    )
    points = calibration.camera_center[None, :] + scales[:, None] * world_rays
    invalid = (~np.isfinite(points).all(axis=1)) | (scales <= 0)
    points[invalid] = np.nan
    return points[:, :2]


def ground_to_image(
    ground_points: np.ndarray, calibration: GroundCalibration
) -> np.ndarray:
    points = np.asarray(ground_points, dtype=np.float64).reshape(-1, 2)
    world = np.column_stack((points, np.zeros(len(points))))
    image, _ = cv2.projectPoints(
        world,
        calibration.rotation_vector,
        calibration.translation_vector,
        calibration.camera_matrix,
        calibration.distortion,
    )
    return image.reshape(-1, 2)


def read_snapshots(path: Path) -> dict[float, dict[int, dict[str, str]]]:
    snapshots: dict[float, dict[int, dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "sample_second",
            "track_id",
            "confidence",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "tracking_state",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Tracking CSV is missing columns: {sorted(missing)}")
        for row in reader:
            second = round(float(row["sample_second"]), 6)
            track_id = int(row["track_id"])
            existing = snapshots.setdefault(second, {}).get(track_id)
            if existing is None or float(row["confidence"]) > float(
                existing["confidence"]
            ):
                snapshots[second][track_id] = row
    if len(snapshots) < 100:
        raise RuntimeError("At least 100 tracking snapshots are required")
    return dict(sorted(snapshots.items()))


def foot_pixel(row: dict[str, str]) -> tuple[float, float]:
    return (
        float(row["bbox_x"]) + float(row["bbox_width"]) / 2.0,
        float(row["bbox_y"]) + float(row["bbox_height"]),
    )


def infer_step(times: Sequence[float]) -> float:
    differences = np.diff(np.asarray(times, dtype=np.float64))
    step = float(np.median(differences))
    if not 0.05 <= step <= 1.01:
        raise RuntimeError(f"Unexpected tracking interval: {step}")
    if np.max(np.abs(differences - step)) > max(0.02, step * 0.1):
        raise RuntimeError("Tracking snapshots are not regularly spaced")
    return step


def directed_edges(config: RoiConfig) -> tuple[tuple[int, int], ...]:
    return tuple(direction for edge in config.edges for direction in (edge, edge[::-1]))


def build_frame_states(
    snapshots: dict[float, dict[int, dict[str, str]]],
    config: RoiConfig,
    calibration: GroundCalibration,
    max_speed_m_s: float,
) -> list[FrameState]:
    histories: dict[int, deque[tuple[float, np.ndarray]]] = defaultdict(deque)
    pixel_histories: dict[int, deque[tuple[float, np.ndarray]]] = defaultdict(deque)
    previous_assignments: dict[int, int] = {}
    states: list[FrameState] = []
    frame_area = float(config.width * config.height)
    for second, rows in snapshots.items():
        track_ids = list(rows)
        pixels = np.array([foot_pixel(rows[track_id]) for track_id in track_ids])
        ground = image_to_ground(pixels, calibration)
        ground_points = {
            track_id: point
            for track_id, point in zip(track_ids, ground)
            if np.isfinite(point).all() and np.linalg.norm(point) < 200.0
        }
        assignments: dict[int, int] = {}
        velocities: dict[int, np.ndarray] = {}
        projected_assignments: dict[int, int] = {}
        pixel_projected_assignments: dict[int, int] = {}
        projected_ground: list[np.ndarray] = []
        projected_ids: list[int] = []
        for track_id, row in rows.items():
            pixel_x, pixel_y = foot_pixel(row)
            pixel_point = np.array((pixel_x, pixel_y), dtype=np.float64)
            assignments[track_id] = zone_for_point(
                pixel_x / config.width, pixel_y / config.height, config
            )
            pixel_history = pixel_histories[track_id]
            pixel_history.append((second, pixel_point))
            while pixel_history and second - pixel_history[0][0] > 1.01:
                pixel_history.popleft()
            pixel_elapsed = pixel_history[-1][0] - pixel_history[0][0]
            if len(pixel_history) >= 2 and pixel_elapsed >= 0.95:
                pixel_velocity = (
                    pixel_history[-1][1] - pixel_history[0][1]
                ) / pixel_elapsed
            else:
                pixel_velocity = np.zeros(2, dtype=np.float64)
            pixel_projection = np.clip(
                pixel_point + pixel_velocity,
                (0.0, 0.0),
                (float(config.width), float(config.height)),
            )
            pixel_projected_assignments[track_id] = zone_for_point(
                float(pixel_projection[0]) / config.width,
                float(pixel_projection[1]) / config.height,
                config,
            )
            point = ground_points.get(track_id)
            history = histories[track_id]
            if history and second - history[-1][0] > 0.61:
                history.clear()
            if point is None:
                velocity = np.zeros(2, dtype=np.float64)
            else:
                history.append((second, point))
                while history and second - history[0][0] > 1.01:
                    history.popleft()
                if len(history) >= 2 and history[-1][0] > history[0][0]:
                    elapsed = history[-1][0] - history[0][0]
                    velocity = (history[-1][1] - history[0][1]) / elapsed
                else:
                    velocity = np.zeros(2, dtype=np.float64)
                speed = float(np.linalg.norm(velocity))
                if speed > max_speed_m_s:
                    velocity *= max_speed_m_s / speed
                projected_ground.append(point + velocity)
                projected_ids.append(track_id)
            velocities[track_id] = velocity
        if projected_ground:
            projected_pixels = ground_to_image(np.stack(projected_ground), calibration)
            for track_id, pixel in zip(projected_ids, projected_pixels):
                if np.isfinite(pixel).all():
                    projected_assignments[track_id] = zone_for_point(
                        float(pixel[0]) / config.width,
                        float(pixel[1]) / config.height,
                        config,
                    )
        for track_id, zone_id in assignments.items():
            projected_assignments.setdefault(track_id, zone_id)

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
            projected_ids_in_zone = {
                track_id
                for track_id, zone_id in projected_assignments.items()
                if zone_id == zone.zone_id
            }
            pixel_projected_ids_in_zone = {
                track_id
                for track_id, zone_id in pixel_projected_assignments.items()
                if zone_id == zone.zone_id
            }
            members = [rows[track_id] for track_id in current_ids]
            count = len(members)
            velocity_members = [velocities[track_id] for track_id in current_ids]
            node_stats.append(
                {
                    "current_count": count,
                    "pixel_projected_count": len(pixel_projected_ids_in_zone),
                    "ground_projected_count": len(projected_ids_in_zone),
                    "previous_count": len(previous_ids),
                    "count_delta": count - len(previous_ids),
                    "inflow": sum(
                        track_id in previous_assignments
                        and previous_assignments[track_id] != zone.zone_id
                        for track_id in current_ids
                    ),
                    "outflow": sum(
                        track_id in assignments
                        and assignments[track_id] != zone.zone_id
                        for track_id in previous_ids
                    ),
                    "projected_inflow": sum(
                        track_id in assignments
                        and assignments[track_id] != zone.zone_id
                        for track_id in pixel_projected_ids_in_zone
                    ),
                    "projected_outflow": sum(
                        pixel_projected_assignments[track_id] != zone.zone_id
                        for track_id in current_ids
                    ),
                    "new_entries": sum(
                        track_id not in previous_assignments for track_id in current_ids
                    ),
                    "disappeared": sum(
                        track_id not in assignments for track_id in previous_ids
                    ),
                    "mean_confidence": (
                        float(np.mean([float(row["confidence"]) for row in members]))
                        if members
                        else 0.0
                    ),
                    "mean_ground_speed_m_s": (
                        float(np.mean([np.linalg.norm(value) for value in velocity_members]))
                        if velocity_members
                        else 0.0
                    ),
                    "mean_ground_velocity_x_m_s": (
                        float(np.mean([value[0] for value in velocity_members]))
                        if velocity_members
                        else 0.0
                    ),
                    "mean_ground_velocity_y_m_s": (
                        float(np.mean([value[1] for value in velocity_members]))
                        if velocity_members
                        else 0.0
                    ),
                    "predicted_ratio": (
                        float(
                            np.mean(
                                [row["tracking_state"] == "predicted" for row in members]
                            )
                        )
                        if members
                        else 0.0
                    ),
                    "mean_bbox_area_ratio": (
                        float(
                            np.mean(
                                [
                                    float(row["bbox_width"])
                                    * float(row["bbox_height"])
                                    / frame_area
                                    for row in members
                                ]
                            )
                        )
                        if members
                        else 0.0
                    ),
                }
            )
        states.append(
            FrameState(
                second=second,
                rows=rows,
                assignments=assignments,
                ground_points=ground_points,
                velocities=velocities,
                projected_assignments=projected_assignments,
                pixel_projected_assignments=pixel_projected_assignments,
                node_stats=node_stats,
            )
        )
        previous_assignments = assignments
    return states


def split_for_target(second: float) -> str:
    if second < 180.0:
        return "train"
    if second < 240.0:
        return "validation"
    return "test"


def encode_features(state: FrameState, count_scale: float) -> torch.Tensor:
    features: list[list[float]] = []
    zone_count = len(state.node_stats)
    for zone_id, stats in enumerate(state.node_stats):
        encoded = []
        for name in BASE_FEATURE_NAMES:
            value = float(stats[name])
            if name in COUNT_FEATURE_NAMES:
                value /= count_scale
            elif name == "mean_ground_speed_m_s":
                value /= 5.0
            elif name in (
                "mean_ground_velocity_x_m_s",
                "mean_ground_velocity_y_m_s",
            ):
                value = max(-1.0, min(1.0, value / 5.0))
            encoded.append(value)
        one_hot = [0.0] * zone_count
        one_hot[zone_id] = 1.0
        features.append(encoded + one_hot)
    return torch.tensor(features, dtype=torch.float32)


def flow_targets(
    source: FrameState,
    target: FrameState,
    edge_list: Sequence[tuple[int, int]],
    node_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = {edge: index for index, edge in enumerate(edge_list)}
    flows = np.zeros(len(edge_list), dtype=np.float32)
    entries = np.zeros(node_count, dtype=np.float32)
    exits = np.zeros(node_count, dtype=np.float32)
    all_ids = set(source.assignments) | set(target.assignments)
    for track_id in all_ids:
        source_zone = source.assignments.get(track_id)
        target_zone = target.assignments.get(track_id)
        if source_zone is None:
            entries[target_zone] += 1.0
        elif target_zone is None:
            exits[source_zone] += 1.0
        elif source_zone != target_zone:
            index = edge_index.get((source_zone, target_zone))
            if index is None:
                exits[source_zone] += 1.0
                entries[target_zone] += 1.0
            else:
                flows[index] += 1.0
    reconstructed = np.array(
        [item["current_count"] for item in source.node_stats], dtype=np.float32
    )
    reconstructed += entries - exits
    for index, (source_zone, target_zone) in enumerate(edge_list):
        reconstructed[source_zone] -= flows[index]
        reconstructed[target_zone] += flows[index]
    actual = np.array(
        [item["current_count"] for item in target.node_stats], dtype=np.float32
    )
    if not np.array_equal(reconstructed, actual):
        raise RuntimeError("Flow conservation target does not reconstruct node counts")
    return (
        torch.from_numpy(flows),
        torch.from_numpy(entries),
        torch.from_numpy(exits),
    )


def make_samples(
    states: Sequence[FrameState],
    config: RoiConfig,
    count_scale: float,
    window_seconds: float,
    horizon_seconds: float,
) -> tuple[list[FlowSample], float]:
    times = [state.second for state in states]
    step = infer_step(times)
    window_steps = round(window_seconds / step)
    horizon_steps = round(horizon_seconds / step)
    if abs(window_steps * step - window_seconds) > 0.02:
        raise RuntimeError("Window seconds must align with the tracking interval")
    if abs(horizon_steps * step - horizon_seconds) > 0.02:
        raise RuntimeError("Horizon seconds must align with the tracking interval")
    encoded = [encode_features(state, count_scale) for state in states]
    edge_list = directed_edges(config)
    samples: list[FlowSample] = []
    for source_index in range(window_steps - 1, len(states) - horizon_steps):
        target_index = source_index + horizon_steps
        source = states[source_index]
        target = states[target_index]
        window_start = source_index - window_steps + 1
        if any(
            abs(states[index + 1].second - states[index].second - step) > 0.02
            for index in range(window_start, target_index)
        ):
            continue
        flows, entries, exits = flow_targets(
            source, target, edge_list, len(config.zones)
        )
        samples.append(
            FlowSample(
                source_second=source.second,
                target_second=target.second,
                features=torch.stack(encoded[window_start : source_index + 1]),
                current_counts=torch.tensor(
                    [item["current_count"] for item in source.node_stats],
                    dtype=torch.float32,
                ),
                pixel_projected_counts=torch.tensor(
                    [item["pixel_projected_count"] for item in source.node_stats],
                    dtype=torch.float32,
                ),
                ground_projected_counts=torch.tensor(
                    [item["ground_projected_count"] for item in source.node_stats],
                    dtype=torch.float32,
                ),
                target_counts=torch.tensor(
                    [item["current_count"] for item in target.node_stats],
                    dtype=torch.float32,
                ),
                edge_flows=flows,
                entries=entries,
                exits=exits,
                split=split_for_target(target.second),
            )
        )
    return samples, step


def stack_samples(
    samples: Sequence[FlowSample], count_scale: float
) -> tuple[torch.Tensor, ...]:
    if not samples:
        raise RuntimeError("Cannot stack an empty sample set")
    return (
        torch.stack([sample.features for sample in samples]),
        torch.stack([sample.current_counts for sample in samples]) / count_scale,
        torch.stack([sample.pixel_projected_counts for sample in samples]) / count_scale,
        torch.stack([sample.ground_projected_counts for sample in samples]) / count_scale,
        torch.stack([sample.target_counts for sample in samples]) / count_scale,
        torch.stack([sample.edge_flows for sample in samples]) / count_scale,
        torch.stack([sample.entries for sample in samples]) / count_scale,
        torch.stack([sample.exits for sample in samples]) / count_scale,
    )


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def loss_for_batch(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    edge_targets: torch.Tensor,
    entry_targets: torch.Tensor,
    exit_targets: torch.Tensor,
    projected: torch.Tensor,
) -> torch.Tensor:
    prediction = outputs["prediction"]
    node_loss = F.smooth_l1_loss(prediction, targets, beta=0.06)
    total_loss = F.smooth_l1_loss(
        prediction.sum(dim=1), targets.sum(dim=1), beta=0.10
    )
    flow_loss = (
        F.smooth_l1_loss(outputs["edge_flows"], edge_targets, beta=0.04)
        + F.smooth_l1_loss(outputs["entries"], entry_targets, beta=0.04)
        + F.smooth_l1_loss(outputs["exits"], exit_targets, beta=0.04)
    )
    correction_penalty = torch.mean(torch.abs(prediction - projected))
    gate_penalty = outputs["gate"].mean()
    return (
        node_loss
        + 0.15 * total_loss
        + 0.20 * flow_loss
        + 0.015 * correction_penalty
        + 0.005 * gate_penalty
    )


def predict(
    model: FlowConservingSTGNN,
    samples: Sequence[FlowSample],
    adjacency: torch.Tensor,
    count_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.empty((0, adjacency.shape[0])), np.empty((0, adjacency.shape[0]))
    features, current, pixel_projected, *_ = stack_samples(samples, count_scale)
    model.eval()
    with torch.no_grad():
        outputs = model(features, adjacency, current, pixel_projected)
    return (
        (outputs["prediction"] * count_scale).cpu().numpy(),
        outputs["gate"].cpu().numpy(),
    )


def mae(
    model: FlowConservingSTGNN,
    samples: Sequence[FlowSample],
    adjacency: torch.Tensor,
    count_scale: float,
) -> float:
    predictions, _ = predict(model, samples, adjacency, count_scale)
    targets = np.stack([sample.target_counts.numpy() for sample in samples])
    return float(np.abs(predictions - targets).mean())


def select_correction_alphas(
    raw_predictions: np.ndarray,
    samples: Sequence[FlowSample],
    fold_ids: Sequence[int],
) -> tuple[np.ndarray, float, float]:
    """Keep an ROI correction only when it does not hurt any validation fold."""
    targets = np.stack([sample.target_counts.numpy() for sample in samples])
    pixel = np.stack([sample.pixel_projected_counts.numpy() for sample in samples])
    fold_array = np.asarray(fold_ids)
    if len(fold_array) != len(samples):
        raise RuntimeError("Each out-of-fold prediction needs a fold ID")
    alphas = np.zeros(targets.shape[1], dtype=np.float32)
    alpha_grid = np.linspace(0.0, 1.0, 21)
    for zone_id in range(targets.shape[1]):
        baseline_fold_mae = {
            fold_id: float(
                np.abs(
                    pixel[fold_array == fold_id, zone_id]
                    - targets[fold_array == fold_id, zone_id]
                ).mean()
            )
            for fold_id in np.unique(fold_array)
        }
        zone_errors: list[float] = []
        for alpha in alpha_grid:
            blended = pixel[:, zone_id] + alpha * (
                raw_predictions[:, zone_id] - pixel[:, zone_id]
            )
            improves_every_fold = all(
                float(
                    np.abs(
                        blended[fold_array == fold_id]
                        - targets[fold_array == fold_id, zone_id]
                    ).mean()
                )
                <= baseline_fold_mae[fold_id] + 1e-7
                for fold_id in baseline_fold_mae
            )
            zone_errors.append(
                float(np.abs(blended - targets[:, zone_id]).mean())
                if improves_every_fold
                else float("inf")
            )
        alphas[zone_id] = float(alpha_grid[int(np.argmin(zone_errors))])
    calibrated = apply_correction_alphas(raw_predictions, pixel, alphas)
    return (
        alphas,
        float(np.abs(raw_predictions - targets).mean()),
        float(np.abs(calibrated - targets).mean()),
    )


def apply_correction_alphas(
    raw_predictions: np.ndarray,
    pixel_projection: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        0.0,
        pixel_projection + alphas[None, :] * (raw_predictions - pixel_projection),
    )


def train_with_early_stopping(
    training: Sequence[FlowSample],
    validation: Sequence[FlowSample],
    adjacency: torch.Tensor,
    input_size: int,
    hidden_size: int,
    edge_list: Sequence[tuple[int, int]],
    count_scale: float,
    candidate: Candidate,
    max_epochs: int,
    patience: int,
) -> tuple[int, float, FlowConservingSTGNN]:
    seed_everything(candidate.seed)
    model = FlowConservingSTGNN(
        input_size, hidden_size, edge_list, candidate.dropout
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=0.001
    )
    tensors = stack_samples(training, count_scale)
    generator = torch.Generator().manual_seed(candidate.seed)
    batch_size = min(64, len(training))
    best_state = clone_state(model)
    best_epoch = 0
    best_mae = float("inf")
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(training), generator=generator)
        for start in range(0, len(training), batch_size):
            indices = order[start : start + batch_size]
            x, current, pixel_projected, _, target, edge, entry, exit_values = (
                value[indices] for value in tensors
            )
            outputs = model(x, adjacency, current, pixel_projected)
            loss = loss_for_batch(
                outputs, target, edge, entry, exit_values, pixel_projected
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        validation_mae = mae(model, validation, adjacency, count_scale)
        if validation_mae < best_mae - 1e-5:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, best_mae, model


def fit_fixed(
    training: Sequence[FlowSample],
    adjacency: torch.Tensor,
    input_size: int,
    hidden_size: int,
    edge_list: Sequence[tuple[int, int]],
    count_scale: float,
    candidate: Candidate,
    epochs: int,
) -> FlowConservingSTGNN:
    seed_everything(candidate.seed)
    model = FlowConservingSTGNN(
        input_size, hidden_size, edge_list, candidate.dropout
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate.learning_rate, weight_decay=0.001
    )
    tensors = stack_samples(training, count_scale)
    generator = torch.Generator().manual_seed(candidate.seed)
    batch_size = min(64, len(training))
    for _ in range(max(1, epochs)):
        model.train()
        order = torch.randperm(len(training), generator=generator)
        for start in range(0, len(training), batch_size):
            indices = order[start : start + batch_size]
            x, current, pixel_projected, _, target, edge, entry, exit_values = (
                value[indices] for value in tensors
            )
            outputs = model(x, adjacency, current, pixel_projected)
            loss = loss_for_batch(
                outputs, target, edge, entry, exit_values, pixel_projected
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


def metrics(
    predictions: np.ndarray, samples: Sequence[FlowSample]
) -> dict[str, float | int]:
    targets = np.stack([sample.target_counts.numpy() for sample in samples])
    current = np.stack([sample.current_counts.numpy() for sample in samples])
    pixel_projected = np.stack(
        [sample.pixel_projected_counts.numpy() for sample in samples]
    )
    ground_projected = np.stack(
        [sample.ground_projected_counts.numpy() for sample in samples]
    )
    result: dict[str, float | int] = {"samples": int(targets.size)}
    for prefix, values in (
        ("gnn", predictions),
        ("persistence", current),
        ("pixel_projection", pixel_projected),
        ("ground_projection", ground_projected),
    ):
        errors = values - targets
        rounded_errors = np.rint(values).clip(min=0) - targets
        result[f"{prefix}_mae"] = float(np.abs(errors).mean())
        result[f"{prefix}_rmse"] = float(np.sqrt(np.square(errors).mean()))
        result[f"{prefix}_exact_accuracy"] = float((rounded_errors == 0).mean())
        result[f"{prefix}_within_one_accuracy"] = float(
            (np.abs(rounded_errors) <= 1).mean()
        )
        result[f"{prefix}_total_count_mae"] = float(
            np.abs(values.sum(axis=1) - targets.sum(axis=1)).mean()
        )
    return result


def write_enriched_tracking(path: Path, states: Sequence[FrameState]) -> None:
    fields = [
        "sample_second",
        "track_id",
        "zone_id",
        "foot_x_px",
        "foot_y_px",
        "ground_x_m",
        "ground_y_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "speed_m_s",
        "pixel_projected_zone_1s",
        "ground_projected_zone_1s",
        "confidence",
        "tracking_state",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for state in states:
            for track_id, row in state.rows.items():
                pixel_x, pixel_y = foot_pixel(row)
                point = state.ground_points.get(track_id)
                velocity = state.velocities[track_id]
                writer.writerow(
                    {
                        "sample_second": state.second,
                        "track_id": track_id,
                        "zone_id": state.assignments[track_id],
                        "foot_x_px": round(pixel_x, 3),
                        "foot_y_px": round(pixel_y, 3),
                        "ground_x_m": round(float(point[0]), 6) if point is not None else "",
                        "ground_y_m": round(float(point[1]), 6) if point is not None else "",
                        "velocity_x_m_s": round(float(velocity[0]), 6),
                        "velocity_y_m_s": round(float(velocity[1]), 6),
                        "speed_m_s": round(float(np.linalg.norm(velocity)), 6),
                        "pixel_projected_zone_1s": (
                            state.pixel_projected_assignments[track_id]
                        ),
                        "ground_projected_zone_1s": (
                            state.projected_assignments[track_id]
                        ),
                        "confidence": row["confidence"],
                        "tracking_state": row["tracking_state"],
                    }
                )


def write_dataset(
    path: Path,
    states: Sequence[FrameState],
    horizon_steps: int,
    config: RoiConfig,
) -> None:
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
        for index, state in enumerate(states):
            target = states[index + horizon_steps] if index + horizon_steps < len(states) else None
            for zone_id, stats in enumerate(state.node_stats):
                writer.writerow(
                    {
                        "sample_second": state.second,
                        "zone_id": zone_id,
                        "zone": config.zones[zone_id].name,
                        **{name: round(float(stats[name]), 6) for name in BASE_FEATURE_NAMES},
                        "target_second": target.second if target is not None else "",
                        "target_next_count": (
                            target.node_stats[zone_id]["current_count"]
                            if target is not None
                            else ""
                        ),
                        "data_split": split_for_target(target.second) if target else "forecast",
                    }
                )


def write_predictions(
    path: Path,
    predictions: np.ndarray,
    raw_predictions: np.ndarray,
    gates: np.ndarray,
    correction_alphas: np.ndarray,
    samples: Sequence[FlowSample],
    config: RoiConfig,
) -> None:
    fields = [
        "data_split",
        "source_second",
        "target_second",
        "zone_id",
        "zone",
        "current_count",
        "pixel_projected_count",
        "ground_projected_count",
        "actual_next_count",
        "raw_gnn_count",
        "predicted_next_count",
        "predicted_rounded_count",
        "correction_alpha",
        "gate",
        "absolute_error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for sample, prediction, raw_prediction, gate in zip(
            samples, predictions, raw_predictions, gates
        ):
            for zone in config.zones:
                actual = float(sample.target_counts[zone.zone_id])
                value = max(0.0, float(prediction[zone.zone_id]))
                writer.writerow(
                    {
                        "data_split": sample.split,
                        "source_second": sample.source_second,
                        "target_second": sample.target_second,
                        "zone_id": zone.zone_id,
                        "zone": zone.name,
                        "current_count": int(sample.current_counts[zone.zone_id]),
                        "pixel_projected_count": int(
                            sample.pixel_projected_counts[zone.zone_id]
                        ),
                        "ground_projected_count": int(
                            sample.ground_projected_counts[zone.zone_id]
                        ),
                        "actual_next_count": int(actual),
                        "raw_gnn_count": round(
                            max(0.0, float(raw_prediction[zone.zone_id])), 6
                        ),
                        "predicted_next_count": round(value, 6),
                        "predicted_rounded_count": round(value),
                        "correction_alpha": round(
                            float(correction_alphas[zone.zone_id]), 3
                        ),
                        "gate": round(float(gate[zone.zone_id]), 6),
                        "absolute_error": round(abs(value - actual), 6),
                    }
                )


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_chart(
    path: Path,
    predictions: np.ndarray,
    samples: Sequence[FlowSample],
    config: RoiConfig,
) -> None:
    integer_indices = [
        index
        for index, sample in enumerate(samples)
        if sample.split == "test" and abs(sample.target_second - round(sample.target_second)) < 1e-6
    ]
    figure, axes_grid = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = list(axes_grid.flat)
    for zone, axis in zip(config.zones, axes):
        seconds = [samples[index].target_second for index in integer_indices]
        actual = [samples[index].target_counts[zone.zone_id].item() for index in integer_indices]
        predicted = [predictions[index, zone.zone_id] for index in integer_indices]
        projected = [
            samples[index].pixel_projected_counts[zone.zone_id].item()
            for index in integer_indices
        ]
        axis.plot(seconds, actual, color="black", linewidth=2.0, label="tracked target")
        axis.plot(
            seconds,
            predicted,
            color="#d6279f",
            linewidth=1.8,
            label="hybrid GNN",
        )
        axis.plot(
            seconds,
            projected,
            color="#1f77b4",
            linewidth=1.2,
            linestyle="--",
            label="pixel projection",
        )
        axis.set_title(f"Z{zone.zone_id} {zone.name}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_xlabel("target second")
        axis.set_ylabel("people")
    figure.suptitle("0.2s projection-anchored flow-GNN: held-out final 60 seconds")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def calibration_reprojection_error(
    states: Sequence[FrameState], calibration: GroundCalibration
) -> float:
    source_pixels: list[tuple[float, float]] = []
    ground_points: list[np.ndarray] = []
    for state in states[:: max(1, len(states) // 100)]:
        for track_id, point in state.ground_points.items():
            source_pixels.append(foot_pixel(state.rows[track_id]))
            ground_points.append(point)
    if not ground_points:
        return float("nan")
    projected = ground_to_image(np.stack(ground_points), calibration)
    errors = projected - np.asarray(source_pixels)
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-csv", type=Path, default=DEFAULT_TRACKING_CSV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--previous-summary", type=Path, default=DEFAULT_PREVIOUS_SUMMARY
    )
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--max-speed-m-s", type=float, default=5.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_roi_config(args.config)
    if len(config.zones) != 6:
        parser.error("The flow model requires the approved six-ROI configuration")
    calibration = load_calibration(args.calibration)
    snapshots = read_snapshots(args.tracking_csv)
    states = build_frame_states(
        snapshots, config, calibration, args.max_speed_m_s
    )
    times = [state.second for state in states]
    step = infer_step(times)
    count_scale = float(
        max(
            10,
            *(
                int(item["current_count"])
                for state in states
                if state.second < 240.0
                for item in state.node_stats
            ),
        )
    )
    samples, _ = make_samples(
        states,
        config,
        count_scale,
        args.window_seconds,
        args.horizon_seconds,
    )
    adjacency = normalized_adjacency(config)
    edge_list = directed_edges(config)
    input_size = len(BASE_FEATURE_NAMES) + len(config.zones)

    candidates = [
        Candidate(0.001, 0.10, 17),
        Candidate(0.003, 0.10, 17),
        Candidate(0.001, 0.20, 17),
        Candidate(0.003, 0.20, 17),
    ]
    folds = (
        (120.0, 120.0, 160.0),
        (160.0, 160.0, 200.0),
        (200.0, 200.0, 240.0),
    )
    cv_rows: list[dict[str, object]] = []
    candidate_scores: list[
        tuple[float, Candidate, list[int], float, np.ndarray]
    ] = []
    for candidate in candidates:
        fold_maes: list[float] = []
        fold_epochs: list[int] = []
        candidate_oof_samples: list[FlowSample] = []
        candidate_oof_predictions: list[np.ndarray] = []
        candidate_oof_fold_ids: list[int] = []
        for fold_index, (train_end, validation_start, validation_end) in enumerate(folds, 1):
            training = [
                sample for sample in samples if sample.target_second < train_end
            ]
            validation = [
                sample
                for sample in samples
                if validation_start <= sample.target_second < validation_end
            ]
            best_epoch, validation_mae, fold_model = train_with_early_stopping(
                training,
                validation,
                adjacency,
                input_size,
                args.hidden_size,
                edge_list,
                count_scale,
                candidate,
                args.max_epochs,
                args.patience,
            )
            fold_maes.append(validation_mae)
            fold_epochs.append(best_epoch)
            fold_predictions, _ = predict(
                fold_model, validation, adjacency, count_scale
            )
            candidate_oof_samples.extend(validation)
            candidate_oof_predictions.append(fold_predictions)
            candidate_oof_fold_ids.extend([fold_index] * len(validation))
            cv_rows.append(
                {
                    "learning_rate": candidate.learning_rate,
                    "dropout": candidate.dropout,
                    "seed": candidate.seed,
                    "fold": fold_index,
                    "train_end_exclusive_second": train_end,
                    "validation_start_second": validation_start,
                    "validation_end_exclusive_second": validation_end,
                    "best_epoch": best_epoch,
                    "validation_mae": round(validation_mae, 6),
                }
            )
            print(
                f"lr={candidate.learning_rate:.3f} dropout={candidate.dropout:.2f} "
                f"fold={fold_index} epoch={best_epoch} MAE={validation_mae:.4f}"
            )
        oof_predictions = np.concatenate(candidate_oof_predictions, axis=0)
        candidate_alphas, raw_oof_mae, calibrated_oof_mae = (
            select_correction_alphas(
                oof_predictions,
                candidate_oof_samples,
                candidate_oof_fold_ids,
            )
        )
        print(
            f"candidate rolling OOF: raw={raw_oof_mae:.4f} "
            f"hybrid={calibrated_oof_mae:.4f} alphas={candidate_alphas.tolist()}"
        )
        candidate_scores.append(
            (
                calibrated_oof_mae,
                candidate,
                fold_epochs,
                raw_oof_mae,
                candidate_alphas,
            )
        )
    (
        mean_cv_mae,
        selected,
        selected_epochs,
        raw_cv_mae,
        correction_alphas,
    ) = min(candidate_scores, key=lambda item: item[0])
    final_epochs = max(1, round(median(selected_epochs)))
    training = [sample for sample in samples if sample.target_second < 240.0]
    test = [sample for sample in samples if sample.target_second >= 240.0]
    final_model = fit_fixed(
        training,
        adjacency,
        input_size,
        args.hidden_size,
        edge_list,
        count_scale,
        selected,
        final_epochs,
    )
    raw_all_predictions, all_gates = predict(
        final_model, samples, adjacency, count_scale
    )
    all_pixel_projection = np.stack(
        [sample.pixel_projected_counts.numpy() for sample in samples]
    )
    all_predictions = apply_correction_alphas(
        raw_all_predictions, all_pixel_projection, correction_alphas
    )
    test_indices = [index for index, sample in enumerate(samples) if sample.target_second >= 240.0]
    test_predictions = all_predictions[test_indices]
    raw_test_predictions = raw_all_predictions[test_indices]
    test_samples = [samples[index] for index in test_indices]
    integer_indices = [
        index
        for index, sample in enumerate(test_samples)
        if abs(sample.target_second - round(sample.target_second)) < 1e-6
    ]
    integer_predictions = test_predictions[integer_indices]
    raw_integer_predictions = raw_test_predictions[integer_indices]
    integer_samples = [test_samples[index] for index in integer_indices]
    test_metrics = metrics(test_predictions, test_samples)
    integer_metrics = metrics(integer_predictions, integer_samples)
    raw_test_metrics = metrics(raw_test_predictions, test_samples)
    raw_integer_metrics = metrics(raw_integer_predictions, integer_samples)

    write_enriched_tracking(args.output_dir / "ground_tracking_02s.csv", states)
    write_dataset(
        args.output_dir / "flow_dataset_02s.csv",
        states,
        round(args.horizon_seconds / step),
        config,
    )
    write_predictions(
        args.output_dir / "flow_predictions.csv",
        all_predictions,
        raw_all_predictions,
        all_gates,
        correction_alphas,
        samples,
        config,
    )
    write_rows(args.output_dir / "rolling_cv_trials.csv", cv_rows)
    metric_rows = [
        {"scope": "test_every_0.2s", **{key: round(value, 6) if isinstance(value, float) else value for key, value in test_metrics.items()}},
        {"scope": "test_integer_seconds", **{key: round(value, 6) if isinstance(value, float) else value for key, value in integer_metrics.items()}},
    ]
    write_rows(args.output_dir / "flow_metrics.csv", metric_rows)
    make_chart(
        args.output_dir / "flow_test_predictions.png",
        all_predictions,
        samples,
        config,
    )

    reprojection_rmse = calibration_reprojection_error(states, calibration)
    previous_metrics = None
    if args.previous_summary.is_file():
        previous_metrics = json.loads(
            args.previous_summary.read_text(encoding="utf-8")
        ).get("test_metrics")
    summary = {
        "tracking_interval_seconds": step,
        "tracking_snapshots": len(states),
        "tracking_rows": sum(len(state.rows) for state in states),
        "dataset_rows": len(states) * len(config.zones),
        "sequence_samples": len(samples),
        "window_seconds": args.window_seconds,
        "horizon_seconds": args.horizon_seconds,
        "hidden_size": args.hidden_size,
        "selected_learning_rate": selected.learning_rate,
        "selected_dropout": selected.dropout,
        "selected_seed": selected.seed,
        "selected_epochs": final_epochs,
        "rolling_cv_mean_mae": mean_cv_mae,
        "rolling_cv_raw_gnn_mae": raw_cv_mae,
        "correction_alpha_by_zone": {
            f"Z{zone.zone_id}_{zone.name}": float(correction_alphas[zone.zone_id])
            for zone in config.zones
        },
        "correction_selection_rule": (
            "A zone correction must not increase MAE in any rolling validation fold"
        ),
        "test_start_second": min(sample.target_second for sample in test),
        "test_end_second": max(sample.target_second for sample in test),
        "calibration_camera_height_m": float(calibration.camera_center[2]),
        "calibration_reprojection_rmse_px": reprojection_rmse,
        "test_every_0_2s": test_metrics,
        "test_integer_seconds": integer_metrics,
        "raw_gnn_test_every_0_2s": raw_test_metrics,
        "raw_gnn_test_integer_seconds": raw_integer_metrics,
        "previous_1s_stgnn_test": previous_metrics,
    }
    (args.output_dir / "flow_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checkpoint = {
        "model_type": "ProjectionAnchoredFlowConservingSTGNN",
        "state_dict": final_model.state_dict(),
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "directed_edges": edge_list,
        "dropout": selected.dropout,
        "window_seconds": args.window_seconds,
        "horizon_seconds": args.horizon_seconds,
        "tracking_interval_seconds": step,
        "count_scale": count_scale,
        "correction_alpha_by_zone": correction_alphas,
        "correction_selection_rule": (
            "non-increasing MAE in every rolling validation fold"
        ),
        "feature_names": list(BASE_FEATURE_NAMES)
        + [f"zone_{index}_one_hot" for index in range(len(config.zones))],
        "roi_config": json.loads(args.config.read_text(encoding="utf-8")),
        "calibration": calibration.raw,
        "test_metrics": test_metrics,
        "raw_gnn_test_metrics": raw_test_metrics,
        "integer_second_test_metrics": integer_metrics,
        "raw_gnn_integer_second_test_metrics": raw_integer_metrics,
    }
    torch.save(checkpoint, args.output_dir / "flow_stgnn.pt")

    report = [
        "# Town Centre 0.2초 보정·흐름 보존 GNN 결과",
        "",
        f"- 추적 간격: {step:.1f}초, {len(states):,}개 스냅샷",
        f"- 데이터셋: {len(states) * len(config.zones):,}행",
        f"- 입력/예측: 최근 {args.window_seconds:.0f}초로 {args.horizon_seconds:.0f}초 뒤 예측",
        f"- 지면 보정 재투영 RMSE: {reprojection_rmse:.4f}px",
        f"- 롤링 검증 원시 GNN MAE: {raw_cv_mae:.4f}명",
        f"- 롤링 검증 하이브리드 MAE: {mean_cv_mae:.4f}명",
        f"- 최종 설정: lr={selected.learning_rate}, dropout={selected.dropout}, epochs={final_epochs}",
        "- ROI별 GNN 보정 계수: "
        + ", ".join(
            f"Z{zone.zone_id}={float(correction_alphas[zone.zone_id]):.2f}"
            for zone in config.zones
        ),
        "- 보정 채택 조건: 세 롤링 검증 구간 모두에서 MAE가 나빠지지 않을 것",
        "",
        "## 마지막 60초 결과",
        "",
        "| 평가 간격 | 방법 | MAE | 완전일치율 | ±1명 정확도 | 전체 인원 MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope, result, raw_result in (
        ("0.2초", test_metrics, raw_test_metrics),
        ("정수 초", integer_metrics, raw_integer_metrics),
    ):
        report.append(
            f"| {scope} | 원시 흐름 GNN | "
            f"{float(raw_result['gnn_mae']):.3f} | "
            f"{100*float(raw_result['gnn_exact_accuracy']):.1f}% | "
            f"{100*float(raw_result['gnn_within_one_accuracy']):.1f}% | "
            f"{float(raw_result['gnn_total_count_mae']):.3f} |"
        )
        for prefix, name in (
            ("gnn", "검증 보정 하이브리드 GNN"),
            ("persistence", "현재값 유지"),
            ("pixel_projection", "픽셀 이동 외삽"),
            ("ground_projection", "지면 이동 외삽"),
        ):
            report.append(
                f"| {scope} | {name} | {float(result[prefix + '_mae']):.3f} | "
                f"{100*float(result[prefix + '_exact_accuracy']):.1f}% | "
                f"{100*float(result[prefix + '_within_one_accuracy']):.1f}% | "
                f"{float(result[prefix + '_total_count_mae']):.3f} |"
            )
    report.extend(
        [
            "",
            "> 마지막 60초는 모든 롤링 검증과 최종 학습에서 제외했습니다. 이 평가는 사람 검출 정답 정확도가 아니라 추적 시계열의 미래 예측 성능입니다.",
            "",
        ]
    )
    if previous_metrics:
        report.extend(
            [
                "## 기존 1초 모델과 같은 정수 초 비교",
                "",
                "| 모델 | MAE | 완전일치율 | ±1명 정확도 |",
                "|---|---:|---:|---:|",
                f"| 기존 1초 ST-GNN | {float(previous_metrics['mae']):.3f} | "
                f"{100*float(previous_metrics['exact_accuracy']):.1f}% | "
                f"{100*float(previous_metrics['within_one_accuracy']):.1f}% |",
                f"| 새 0.2초 하이브리드 GNN | {float(integer_metrics['gnn_mae']):.3f} | "
                f"{100*float(integer_metrics['gnn_exact_accuracy']):.1f}% | "
                f"{100*float(integer_metrics['gnn_within_one_accuracy']):.1f}% |",
                "",
            ]
        )
    (args.output_dir / "FLOW_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\nSelected flow model")
    print(
        f"lr={selected.learning_rate} dropout={selected.dropout} "
        f"epochs={final_epochs} rolling-CV raw={raw_cv_mae:.4f} "
        f"hybrid={mean_cv_mae:.4f}"
    )
    print(
        f"test 0.2s MAE={float(test_metrics['gnn_mae']):.4f}, "
        f"pixel={float(test_metrics['pixel_projection_mae']):.4f}, "
        f"ground={float(test_metrics['ground_projection_mae']):.4f}, "
        f"raw={float(raw_test_metrics['gnn_mae']):.4f}"
    )
    print(
        f"test integer-second MAE={float(integer_metrics['gnn_mae']):.4f}, "
        f"pixel={float(integer_metrics['pixel_projection_mae']):.4f}, "
        f"ground={float(integer_metrics['ground_projection_mae']):.4f}, "
        f"raw={float(raw_integer_metrics['gnn_mae']):.4f}"
    )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
