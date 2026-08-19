#!/usr/bin/env python3
"""Predict next-second occupancy with regions as graph nodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import cv2
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "필요한 패키지가 없습니다. 'python3 -m pip install -r requirements.txt'를 "
        "먼저 실행하세요."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ZONES = PROJECT_DIR / "configs" / "shopping_zones.json"
BASE_FEATURE_NAMES = (
    "current_count",
    "previous_count",
    "count_delta",
    "inflow",
    "outflow",
    "new_entries",
    "disappeared",
    "mean_confidence",
    "mean_speed",
    "predicted_ratio",
    "mean_bbox_area",
    "zone_center_x",
    "zone_center_y",
    "zone_width",
    "zone_height",
)


@dataclass(frozen=True)
class Zone:
    zone_id: int
    name: str
    rect: tuple[float, float, float, float]
    color_bgr: tuple[int, int, int]

    def contains(self, x: float, y: float) -> bool:
        left, top, right, bottom = self.rect
        return left <= x <= right and top <= y <= bottom


@dataclass
class RegionGraphSample:
    source_second: float
    target_second: float
    features: torch.Tensor
    adjacency: torch.Tensor
    current_counts: torch.Tensor
    target_counts: torch.Tensor | None
    zone_stats: list[dict[str, float | int]]


class GraphConv(nn.Module):
    """One message-passing layer over the fixed region graph."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.self_layer = nn.Linear(input_size, output_size)
        self.neighbor_layer = nn.Linear(input_size, output_size, bias=False)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_features = adjacency @ features
        return self.self_layer(features) + self.neighbor_layer(neighbor_features)


class RegionOccupancyGNN(nn.Module):
    """Forecast each region's count while exchanging adjacent-region messages."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GraphConv(input_size, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.dropout = dropout
        # 학습 시작점은 "다음에도 현재 인원과 같다"는 안전한 기준선이다.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(features, adjacency))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        hidden = F.relu(self.conv2(hidden, adjacency))
        predicted_delta = self.output(hidden).squeeze(-1)
        return torch.relu(features[:, 0] + predicted_delta)


def load_graph_config(
    path: Path,
) -> tuple[int, int, list[Zone], list[tuple[int, int]]]:
    if not path.is_file():
        raise FileNotFoundError(f"영역 설정을 찾을 수 없습니다: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    width = int(config.get("frame_width", 0))
    height = int(config.get("frame_height", 0))
    raw_zones = config.get("zones")
    raw_edges = config.get("graph_edges")
    if (
        width <= 0
        or height <= 0
        or not isinstance(raw_zones, list)
        or not isinstance(raw_edges, list)
    ):
        raise RuntimeError(
            "영역 설정에 frame_width, frame_height, zones, graph_edges가 필요합니다."
        )

    zones: list[Zone] = []
    for index, item in enumerate(raw_zones):
        if not isinstance(item, dict):
            raise RuntimeError(f"zones[{index}] 형식이 올바르지 않습니다.")
        rect = tuple(float(value) for value in item.get("rect", ()))
        color = tuple(int(value) for value in item.get("color_bgr", ()))
        if len(rect) != 4 or len(color) != 3:
            raise RuntimeError(f"zones[{index}]의 rect 또는 color_bgr가 잘못됐습니다.")
        if not all(0.0 <= value <= 1.0 for value in rect):
            raise RuntimeError(f"zones[{index}]의 rect는 0~1 비율이어야 합니다.")
        left, top, right, bottom = rect
        if left >= right or top >= bottom:
            raise RuntimeError(f"zones[{index}]의 rect 범위가 올바르지 않습니다.")
        zones.append(
            Zone(
                zone_id=int(item.get("id", index)),
                name=str(item.get("name", f"zone_{index}")),
                rect=(left, top, right, bottom),
                color_bgr=color,
            )
        )
    zones.sort(key=lambda zone: zone.zone_id)
    expected_ids = list(range(len(zones)))
    if len(zones) < 2 or [zone.zone_id for zone in zones] != expected_ids:
        raise RuntimeError("영역 ID는 0부터 연속되어야 하며 영역은 2개 이상이어야 합니다.")

    edges: list[tuple[int, int]] = []
    for index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise RuntimeError(f"graph_edges[{index}] 형식이 올바르지 않습니다.")
        first, second = (int(value) for value in raw_edge)
        if first == second or first not in expected_ids or second not in expected_ids:
            raise RuntimeError(f"graph_edges[{index}]에 잘못된 영역 ID가 있습니다.")
        edge = (min(first, second), max(first, second))
        if edge not in edges:
            edges.append(edge)
    if not edges:
        raise RuntimeError("지역 GNN에는 graph_edges가 한 개 이상 필요합니다.")
    return width, height, zones, edges


def read_snapshots(path: Path) -> dict[float, dict[int, dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"추적 CSV를 찾을 수 없습니다: {path}")
    required = {
        "sample_second",
        "track_id",
        "confidence",
        "bbox_width",
        "bbox_height",
        "center_x",
        "center_y",
        "speed_px_s",
        "tracking_state",
    }
    snapshots: dict[float, dict[int, dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"추적 CSV에 필요한 열이 없습니다: {sorted(missing)}")
        for row in reader:
            second = float(row["sample_second"])
            track_id = int(row["track_id"])
            by_id = snapshots.setdefault(second, {})
            previous = by_id.get(track_id)
            if previous is None or float(row["confidence"]) > float(previous["confidence"]):
                by_id[track_id] = row
    if len(snapshots) < 3:
        raise RuntimeError("GNN 학습에는 사람이 기록된 시각이 최소 3개 필요합니다.")
    return dict(sorted(snapshots.items()))


def zone_for_row(
    row: dict[str, str], width: int, height: int, zones: Sequence[Zone]
) -> int:
    x = min(1.0, max(0.0, float(row["center_x"]) / width))
    y = min(1.0, max(0.0, float(row["center_y"]) / height))
    for zone in zones:
        if zone.contains(x, y):
            return zone.zone_id
    # 설정에 작은 틈이 있으면 가장 가까운 영역 중심으로 배정한다.
    return min(
        zones,
        key=lambda zone: math.hypot(
            x - (zone.rect[0] + zone.rect[2]) / 2.0,
            y - (zone.rect[1] + zone.rect[3]) / 2.0,
        ),
    ).zone_id


def assign_people(
    rows: dict[int, dict[str, str]],
    width: int,
    height: int,
    zones: Sequence[Zone],
) -> dict[int, int]:
    return {
        track_id: zone_for_row(row, width, height, zones)
        for track_id, row in rows.items()
    }


def make_adjacency(
    zone_count: int, edges: Sequence[tuple[int, int]]
) -> torch.Tensor:
    adjacency = torch.zeros((zone_count, zone_count), dtype=torch.float32)
    for first, second in edges:
        adjacency[first, second] = 1.0
        adjacency[second, first] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def zone_counts(assignments: dict[int, int], zone_count: int) -> list[int]:
    counts = [0] * zone_count
    for zone_id in assignments.values():
        counts[zone_id] += 1
    return counts


def calculate_count_scale(
    snapshots: dict[float, dict[int, dict[str, str]]],
    width: int,
    height: int,
    zones: Sequence[Zone],
) -> float:
    maximum = 0
    for rows in snapshots.values():
        assignments = assign_people(rows, width, height, zones)
        maximum = max(maximum, *zone_counts(assignments, len(zones)))
    return float(max(10, maximum))


def build_zone_stats(
    rows: dict[int, dict[str, str]],
    assignments: dict[int, int],
    previous_assignments: dict[int, int],
    zones: Sequence[Zone],
    width: int,
    height: int,
) -> list[dict[str, float | int]]:
    frame_area = float(width * height)
    diagonal = math.hypot(width, height)
    stats: list[dict[str, float | int]] = []
    for zone in zones:
        members = [
            rows[track_id]
            for track_id, zone_id in assignments.items()
            if zone_id == zone.zone_id
        ]
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
        inflow = sum(
            track_id in previous_assignments
            and previous_assignments[track_id] != zone.zone_id
            for track_id in current_ids
        )
        outflow = sum(
            track_id in assignments and assignments[track_id] != zone.zone_id
            for track_id in previous_ids
        )
        new_entries = sum(track_id not in previous_assignments for track_id in current_ids)
        disappeared = sum(track_id not in assignments for track_id in previous_ids)
        count = len(members)
        mean_confidence = (
            sum(float(row["confidence"]) for row in members) / count if count else 0.0
        )
        mean_speed = (
            sum(float(row["speed_px_s"]) for row in members) / count if count else 0.0
        )
        predicted_ratio = (
            sum(row["tracking_state"].strip() == "predicted" for row in members) / count
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
        left, top, right, bottom = zone.rect
        stats.append(
            {
                "current_count": count,
                "previous_count": len(previous_ids),
                "count_delta": count - len(previous_ids),
                "inflow": inflow,
                "outflow": outflow,
                "new_entries": new_entries,
                "disappeared": disappeared,
                "mean_confidence": mean_confidence,
                "mean_speed": mean_speed,
                "mean_speed_normalized": min(2.0, mean_speed / diagonal),
                "predicted_ratio": predicted_ratio,
                "mean_bbox_area": mean_bbox_area,
                "mean_bbox_area_normalized": min(1.0, mean_bbox_area / frame_area),
                "zone_center_x": (left + right) / 2.0,
                "zone_center_y": (top + bottom) / 2.0,
                "zone_width": right - left,
                "zone_height": bottom - top,
            }
        )
    return stats


def stats_to_features(
    stats: Sequence[dict[str, float | int]], count_scale: float
) -> torch.Tensor:
    zone_count = len(stats)
    features: list[list[float]] = []
    for zone_id, item in enumerate(stats):
        one_hot = [0.0] * zone_count
        one_hot[zone_id] = 1.0
        features.append(
            [
                float(item["current_count"]) / count_scale,
                float(item["previous_count"]) / count_scale,
                float(item["count_delta"]) / count_scale,
                float(item["inflow"]) / count_scale,
                float(item["outflow"]) / count_scale,
                float(item["new_entries"]) / count_scale,
                float(item["disappeared"]) / count_scale,
                float(item["mean_confidence"]),
                float(item["mean_speed_normalized"]),
                float(item["predicted_ratio"]),
                float(item["mean_bbox_area_normalized"]),
                float(item["zone_center_x"]),
                float(item["zone_center_y"]),
                float(item["zone_width"]),
                float(item["zone_height"]),
            ]
            + one_hot
        )
    return torch.tensor(features, dtype=torch.float32)


def build_region_graphs(
    snapshots: dict[float, dict[int, dict[str, str]]],
    width: int,
    height: int,
    zones: Sequence[Zone],
    edges: Sequence[tuple[int, int]],
    horizon_steps: int,
) -> tuple[list[RegionGraphSample], float]:
    times = list(snapshots)
    typical_interval = float(np.median(np.diff(times)))
    count_scale = calculate_count_scale(snapshots, width, height, zones)
    adjacency = make_adjacency(len(zones), edges)
    graphs: list[RegionGraphSample] = []
    previous_assignments: dict[int, int] = {}

    for time_index, source_second in enumerate(times):
        rows = snapshots[source_second]
        assignments = assign_people(rows, width, height, zones)
        stats = build_zone_stats(
            rows,
            assignments,
            previous_assignments,
            zones,
            width,
            height,
        )
        current_counts = torch.tensor(
            zone_counts(assignments, len(zones)), dtype=torch.float32
        )
        future_index = time_index + horizon_steps
        if future_index < len(times):
            future_rows = snapshots[times[future_index]]
            future_assignments = assign_people(future_rows, width, height, zones)
            target_counts = torch.tensor(
                zone_counts(future_assignments, len(zones)), dtype=torch.float32
            )
            target_second = times[future_index]
        else:
            target_counts = None
            target_second = source_second + typical_interval * horizon_steps
        graphs.append(
            RegionGraphSample(
                source_second=source_second,
                target_second=target_second,
                features=stats_to_features(stats, count_scale),
                adjacency=adjacency,
                current_counts=current_counts,
                target_counts=target_counts,
                zone_stats=stats,
            )
        )
        previous_assignments = assignments
    return graphs, count_scale


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_counts(
    model: RegionOccupancyGNN,
    graph: RegionGraphSample,
    count_scale: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        normalized = model(
            graph.features.to(device), graph.adjacency.to(device)
        ).cpu()
    return normalized * count_scale


def evaluate(
    model: RegionOccupancyGNN,
    graphs: Sequence[RegionGraphSample],
    count_scale: float,
    device: torch.device,
) -> dict[str, float | int]:
    absolute_errors: list[torch.Tensor] = []
    squared_errors: list[torch.Tensor] = []
    baseline_errors: list[torch.Tensor] = []
    for graph in graphs:
        if graph.target_counts is None:
            continue
        predicted = predict_counts(model, graph, count_scale, device)
        error = predicted - graph.target_counts
        absolute_errors.append(error.abs())
        squared_errors.append(error.square())
        baseline_errors.append((graph.current_counts - graph.target_counts).abs())
    if not absolute_errors:
        return {"nodes": 0, "mae": 0.0, "rmse": 0.0, "baseline_mae": 0.0}
    absolute = torch.cat(absolute_errors)
    squared = torch.cat(squared_errors)
    baseline = torch.cat(baseline_errors)
    return {
        "nodes": absolute.numel(),
        "mae": float(absolute.mean()),
        "rmse": float(torch.sqrt(squared.mean())),
        "baseline_mae": float(baseline.mean()),
    }


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_model(
    model: RegionOccupancyGNN,
    training: Sequence[RegionGraphSample],
    validation: Sequence[RegionGraphSample],
    count_scale: float,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> tuple[int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    report_every = max(1, epochs // 5)
    best_state = clone_state_dict(model)
    best_validation_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[torch.Tensor] = []
        for graph in training:
            if graph.target_counts is None:
                continue
            predicted = model(
                graph.features.to(device), graph.adjacency.to(device)
            )
            target = graph.target_counts.to(device) / count_scale
            losses.append(F.smooth_l1_loss(predicted, target, beta=0.05))
        if not losses:
            raise RuntimeError("학습 가능한 다음 시각 데이터가 없습니다.")
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        validation_metrics = evaluate(model, validation, count_scale, device)
        validation_mae = float(validation_metrics["mae"])
        if validation_mae + 1e-5 < best_validation_mae:
            best_validation_mae = validation_mae
            best_state = clone_state_dict(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch == epochs or epoch % report_every == 0:
            print(
                f"epoch {epoch:4d}/{epochs} | loss {loss.item():.5f} | "
                f"validation MAE {validation_mae:.3f}명"
            )
        if stale_epochs >= patience:
            print(f"early stopping: {epoch} epoch")
            break

    model.load_state_dict(best_state)
    return best_epoch, best_validation_mae


def prediction_rows(
    model: RegionOccupancyGNN,
    graphs: Sequence[RegionGraphSample],
    zones: Sequence[Zone],
    count_scale: float,
    device: torch.device,
    train_count: int,
) -> dict[float, list[dict[str, float | int | str]]]:
    rows_by_second: dict[float, list[dict[str, float | int | str]]] = {}
    for graph_index, graph in enumerate(graphs):
        predicted = predict_counts(model, graph, count_scale, device)
        split = (
            "forecast"
            if graph.target_counts is None
            else "train"
            if graph_index < train_count
            else "validation"
        )
        rows: list[dict[str, float | int | str]] = []
        for zone in zones:
            zone_id = zone.zone_id
            stats = graph.zone_stats[zone_id]
            prediction = max(0.0, float(predicted[zone_id]))
            actual: float | str = (
                float(graph.target_counts[zone_id])
                if graph.target_counts is not None
                else ""
            )
            rows.append(
                {
                    "source_second": round(graph.source_second, 3),
                    "target_second": round(graph.target_second, 3),
                    "zone_id": zone_id,
                    "zone": zone.name,
                    "current_people_count": int(graph.current_counts[zone_id]),
                    "previous_people_count": int(stats["previous_count"]),
                    "inflow": int(stats["inflow"]),
                    "outflow": int(stats["outflow"]),
                    "new_entries": int(stats["new_entries"]),
                    "disappeared": int(stats["disappeared"]),
                    "mean_speed_px_s": round(float(stats["mean_speed"]), 2),
                    "predicted_next_count": round(prediction, 2),
                    "predicted_rounded_count": round(prediction),
                    "actual_next_count": int(actual) if actual != "" else "",
                    "absolute_error": (
                        round(abs(prediction - float(actual)), 2)
                        if actual != ""
                        else ""
                    ),
                    "data_split": split,
                }
            )
        rows_by_second[graph.source_second] = rows
    return rows_by_second


def write_predictions(
    path: Path,
    rows_by_second: dict[float, list[dict[str, float | int | str]]],
) -> None:
    fields = [
        "source_second",
        "target_second",
        "zone_id",
        "zone",
        "current_people_count",
        "previous_people_count",
        "inflow",
        "outflow",
        "new_entries",
        "disappeared",
        "mean_speed_px_s",
        "predicted_next_count",
        "predicted_rounded_count",
        "actual_next_count",
        "absolute_error",
        "data_split",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rows in rows_by_second.values():
            writer.writerows(rows)


def draw_preview(
    video_path: Path,
    output_path: Path,
    preview_second: float,
    rows_by_second: dict[float, list[dict[str, float | int | str]]],
    zones: Sequence[Zone],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"미리보기 영상을 열 수 없습니다: {video_path}")
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, preview_second) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"{preview_second}초 프레임을 읽을 수 없습니다.")

    sample_second = min(rows_by_second, key=lambda value: abs(value - preview_second))
    rows = {int(row["zone_id"]): row for row in rows_by_second[sample_second]}
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for zone in zones:
        left, top, right, bottom = zone.rect
        x1, y1 = round(left * width), round(top * height)
        x2, y2 = round(right * width), round(bottom * height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), zone.color_bgr, -1)
    frame = cv2.addWeighted(overlay, 0.13, frame, 0.87, 0.0)

    for zone in zones:
        left, top, right, bottom = zone.rect
        x1, y1 = round(left * width), round(top * height)
        x2, y2 = round(right * width), round(bottom * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), zone.color_bgr, 5)
        row = rows[zone.zone_id]
        label = (
            f"Z{zone.zone_id} {zone.name} | now {row['current_people_count']} "
            f"-> next {float(row['predicted_next_count']):.1f}"
        )
        cv2.putText(
            frame,
            label,
            (x1 + 18, min(height - 20, y1 + 48)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            zone.color_bgr,
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Region-node GNN: {sample_second:.1f}s -> next second",
        (20, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"미리보기 이미지를 저장하지 못했습니다: {output_path}")


def save_checkpoint(
    path: Path,
    model: RegionOccupancyGNN,
    zones_path: Path,
    width: int,
    height: int,
    hidden_size: int,
    dropout: float,
    count_scale: float,
    edges: Sequence[tuple[int, int]],
    zone_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "region_node_occupancy_gnn_v1",
            "state_dict": model.state_dict(),
            "input_size": len(BASE_FEATURE_NAMES) + zone_count,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "count_scale": count_scale,
            "frame_width": width,
            "frame_height": height,
            "zone_count": zone_count,
            "graph_edges": list(edges),
            "zones_path": str(zones_path.resolve()),
            "feature_names": BASE_FEATURE_NAMES,
        },
        path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="각 영역을 GNN 노드로 구성해 영역별 다음 인원수를 예측합니다."
    )
    parser.add_argument("csv", help="object_tracker.py가 생성한 추적 CSV")
    parser.add_argument("--zones", default=str(DEFAULT_ZONES), help="지역 그래프 JSON")
    parser.add_argument(
        "--model-output", default="models/shopping_zone_gnn.pt", help="모델 저장 경로"
    )
    parser.add_argument("--load-model", help="기존 지역 GNN 체크포인트")
    parser.add_argument(
        "--predictions",
        default="results/shopping_region_gnn_predictions.csv",
        help="영역별 다음 인원 예측 CSV",
    )
    parser.add_argument("--video", help="영역 미리보기에 사용할 원본 영상")
    parser.add_argument(
        "--preview-output",
        default="results/shopping_region_gnn_zones_preview.jpg",
        help="영역별 인원 예측 미리보기 이미지",
    )
    parser.add_argument("--preview-second", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda 등")
    args = parser.parse_args(argv)
    if args.epochs < 0:
        parser.error("--epochs는 0 이상이어야 합니다.")
    if args.epochs == 0 and not args.load_model:
        parser.error("--epochs 0에는 --load-model이 필요합니다.")
    if args.hidden_size <= 0 or args.horizon_steps <= 0 or args.patience <= 0:
        parser.error("hidden-size, horizon-steps, patience는 0보다 커야 합니다.")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout은 0 이상 1 미만이어야 합니다.")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("--train-ratio는 0보다 크고 1보다 작아야 합니다.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        zones_path = Path(args.zones).expanduser()
        width, height, zones, edges = load_graph_config(zones_path)
        snapshots = read_snapshots(Path(args.csv).expanduser())
        graphs, count_scale = build_region_graphs(
            snapshots,
            width,
            height,
            zones,
            edges,
            args.horizon_steps,
        )
        labeled_graphs = [graph for graph in graphs if graph.target_counts is not None]
        if len(labeled_graphs) < 2:
            raise RuntimeError("학습/검증 분할에 필요한 연속 시각 데이터가 부족합니다.")
        train_count = max(
            1,
            min(
                len(labeled_graphs) - 1,
                round(len(labeled_graphs) * args.train_ratio),
            ),
        )
        training = labeled_graphs[:train_count]
        validation = labeled_graphs[train_count:]
        device = choose_device(args.device)
        input_size = len(BASE_FEATURE_NAMES) + len(zones)
        hidden_size = args.hidden_size
        dropout = args.dropout

        checkpoint = None
        if args.load_model:
            checkpoint = torch.load(
                Path(args.load_model).expanduser(), map_location="cpu", weights_only=True
            )
            if checkpoint.get("model_type") != "region_node_occupancy_gnn_v1":
                raise RuntimeError("사람 노드 모델이 아닌 지역 노드 GNN 체크포인트가 필요합니다.")
            if (
                int(checkpoint["input_size"]) != input_size
                or int(checkpoint["zone_count"]) != len(zones)
            ):
                raise RuntimeError("체크포인트와 현재 지역 그래프 구조가 다릅니다.")
            hidden_size = int(checkpoint["hidden_size"])
            dropout = float(checkpoint["dropout"])
            count_scale = float(checkpoint["count_scale"])

        model = RegionOccupancyGNN(input_size, hidden_size, dropout).to(device)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["state_dict"])

        print(
            f"지역 노드 {len(zones)}개 | 엣지 {len(edges)}개 | "
            f"학습 시점 {len(training)}개 | 검증 시점 {len(validation)}개 | 장치 {device}"
        )
        if args.epochs:
            best_epoch, best_mae = train_model(
                model,
                training,
                validation,
                count_scale,
                device,
                args.epochs,
                args.learning_rate,
                args.weight_decay,
                args.patience,
            )
            print(f"선택된 epoch {best_epoch} | 검증 MAE {best_mae:.3f}명")

        train_metrics = evaluate(model, training, count_scale, device)
        validation_metrics = evaluate(model, validation, count_scale, device)
        model_path = Path(args.model_output).expanduser()
        save_checkpoint(
            model_path,
            model,
            zones_path,
            width,
            height,
            hidden_size,
            dropout,
            count_scale,
            edges,
            len(zones),
        )
        rows_by_second = prediction_rows(
            model, graphs, zones, count_scale, device, train_count
        )
        predictions_path = Path(args.predictions).expanduser()
        write_predictions(predictions_path, rows_by_second)
        if args.video:
            draw_preview(
                Path(args.video).expanduser(),
                Path(args.preview_output).expanduser(),
                args.preview_second,
                rows_by_second,
                zones,
            )

        print(
            f"학습 MAE {float(train_metrics['mae']):.3f}명 | "
            f"현재 인원 유지 기준 {float(train_metrics['baseline_mae']):.3f}명"
        )
        print(
            f"검증 MAE {float(validation_metrics['mae']):.3f}명 | "
            f"RMSE {float(validation_metrics['rmse']):.3f}명 | "
            f"현재 인원 유지 기준 {float(validation_metrics['baseline_mae']):.3f}명"
        )
        print(f"모델: {model_path.resolve()}")
        print(f"지역별 예측: {predictions_path.resolve()}")
        if args.video:
            print(f"지역 미리보기: {Path(args.preview_output).expanduser().resolve()}")
        if len(labeled_graphs) < 20:
            print(
                "주의: 학습 시점이 20개 미만입니다. 구조 검증용 결과이며 실제 예측에는 "
                "더 긴 추적 CSV가 필요합니다.",
                file=sys.stderr,
            )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
