#!/usr/bin/env python3
"""Run a reproducible Town Centre ground-truth/tracker GNN comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/towncentre_matplotlib")

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from gnn_zone_predictor import (
    BASE_FEATURE_NAMES,
    RegionGraphSample,
    RegionOccupancyGNN,
    Zone,
    assign_people,
    build_zone_stats,
    evaluate,
    load_graph_config,
    make_adjacency,
    predict_counts,
    read_snapshots,
    stats_to_features,
    train_model,
    zone_counts,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_TOP = PROJECT_DIR.parent / "archive" / "TownCentre-groundtruth.top"
DEFAULT_VIDEO = PROJECT_DIR.parent / "archive" / "TownCentreXVID.mp4"
DEFAULT_ZONES = PROJECT_DIR / "configs" / "towncentre_zones.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "towncentre_experiment"

TRACKING_FIELDS = (
    "sample_second",
    "video_time_s",
    "frame_index",
    "track_id",
    "label",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "center_x",
    "center_y",
    "area_px",
    "displacement_px",
    "speed_px_s",
    "track_age_s",
    "tracking_state",
    "missed_frames",
)


@dataclass(frozen=True)
class TopRecord:
    person_id: int
    frame_index: int
    head_valid: bool
    body_valid: bool
    head_box: tuple[float, float, float, float]
    body_box: tuple[float, float, float, float]


def parse_top(path: Path) -> tuple[list[TopRecord], dict[str, int]]:
    records: list[TopRecord] = []
    malformed = 0
    invalid = 0
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 12:
                malformed += 1
                continue
            try:
                values = [float(value) for value in parts]
            except ValueError:
                malformed += 1
                continue
            record = TopRecord(
                person_id=int(values[0]),
                frame_index=int(values[1]),
                head_valid=bool(int(values[2])),
                body_valid=bool(int(values[3])),
                head_box=tuple(values[4:8]),
                body_box=tuple(values[8:12]),
            )
            if not record.body_valid:
                invalid += 1
                continue
            records.append(record)
    if not records:
        raise RuntimeError(f"No valid records found in {path}")
    return records, {"malformed_rows": malformed, "invalid_body_rows": invalid}


def video_metadata(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return fps, width, height, frame_count


def clipped_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    left, top, right, bottom = box
    left = min(float(width), max(0.0, left))
    right = min(float(width), max(0.0, right))
    top = min(float(height), max(0.0, top))
    bottom = min(float(height), max(0.0, bottom))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def write_ground_truth_tracking_csv(
    records: Sequence[TopRecord],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    sample_interval: float,
) -> dict[str, int | float]:
    by_frame: dict[int, list[TopRecord]] = defaultdict(list)
    first_frame_by_id: dict[int, int] = {}
    for record in records:
        by_frame[record.frame_index].append(record)
        first_frame_by_id[record.person_id] = min(
            record.frame_index, first_frame_by_id.get(record.person_id, record.frame_index)
        )

    maximum_frame = max(by_frame)
    step = max(1, round(fps * sample_interval))
    annotation_fps = step / sample_interval
    sample_frames = range(0, maximum_frame + 1, step)
    previous_by_id: dict[int, tuple[float, float, float]] = {}
    rows_written = 0
    snapshots_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TRACKING_FIELDS, lineterminator="\n")
        writer.writeheader()
        for frame_index in sample_frames:
            frame_records = by_frame.get(frame_index, [])
            if not frame_records:
                continue
            # The MP4 container reports an average rate near 25.003 fps, while the
            # Town Centre annotation frame numbers use the nominal 25 fps clock.
            sample_second = (frame_index // step) * sample_interval
            snapshots_written += 1
            for record in frame_records:
                box = clipped_box(record.body_box, width, height)
                if box is None:
                    continue
                left, top, right, bottom = box
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                previous = previous_by_id.get(record.person_id)
                displacement = 0.0
                speed = 0.0
                if previous is not None:
                    previous_x, previous_y, previous_time = previous
                    elapsed = sample_second - previous_time
                    if elapsed > 0:
                        displacement = math.hypot(
                            center_x - previous_x, center_y - previous_y
                        )
                        speed = displacement / elapsed
                previous_by_id[record.person_id] = (
                    center_x,
                    center_y,
                    sample_second,
                )
                box_width = right - left
                box_height = bottom - top
                writer.writerow(
                    {
                        "sample_second": round(sample_second, 3),
                        "video_time_s": round(sample_second, 3),
                        "frame_index": frame_index,
                        "track_id": record.person_id,
                        "label": "person",
                        "confidence": 1.0,
                        "bbox_x": round(left, 3),
                        "bbox_y": round(top, 3),
                        "bbox_width": round(box_width, 3),
                        "bbox_height": round(box_height, 3),
                        "center_x": round(center_x, 3),
                        "center_y": round(center_y, 3),
                        "area_px": round(box_width * box_height, 3),
                        "displacement_px": round(displacement, 3),
                        "speed_px_s": round(speed, 3),
                        "track_age_s": round(
                            (frame_index - first_frame_by_id[record.person_id])
                            / annotation_fps,
                            3,
                        ),
                        "tracking_state": "detected",
                        "missed_frames": 0,
                    }
                )
                rows_written += 1
    return {
        "sample_interval_s": sample_interval,
        "sample_step_frames": step,
        "annotation_fps": annotation_fps,
        "snapshots": snapshots_written,
        "rows": rows_written,
        "maximum_annotated_frame": maximum_frame,
        "maximum_annotated_second": maximum_frame / annotation_fps,
    }


def target_at(
    target_snapshots: dict[float, dict[int, dict[str, str]]], requested: float
) -> tuple[float, dict[int, dict[str, str]]] | None:
    if requested in target_snapshots:
        return requested, target_snapshots[requested]
    nearest = min(target_snapshots, key=lambda value: abs(value - requested))
    if abs(nearest - requested) <= 0.05:
        return nearest, target_snapshots[nearest]
    return None


def build_supervised_graphs(
    input_snapshots: dict[float, dict[int, dict[str, str]]],
    target_snapshots: dict[float, dict[int, dict[str, str]]],
    width: int,
    height: int,
    zones: Sequence[Zone],
    edges: Sequence[tuple[int, int]],
    horizon_seconds: float,
    count_scale: float,
) -> list[RegionGraphSample]:
    adjacency = make_adjacency(len(zones), edges)
    previous_assignments: dict[int, int] = {}
    graphs: list[RegionGraphSample] = []
    for source_second, rows in sorted(input_snapshots.items()):
        assignments = assign_people(rows, width, height, zones)
        stats = build_zone_stats(
            rows, assignments, previous_assignments, zones, width, height
        )
        target = target_at(target_snapshots, source_second + horizon_seconds)
        if target is not None:
            target_second, target_rows = target
            target_assignments = assign_people(target_rows, width, height, zones)
            graphs.append(
                RegionGraphSample(
                    source_second=source_second,
                    target_second=target_second,
                    features=stats_to_features(stats, count_scale),
                    adjacency=adjacency,
                    current_counts=torch.tensor(
                        zone_counts(assignments, len(zones)), dtype=torch.float32
                    ),
                    target_counts=torch.tensor(
                        zone_counts(target_assignments, len(zones)), dtype=torch.float32
                    ),
                    zone_stats=stats,
                )
            )
        previous_assignments = assignments
    return graphs


def split_times(times: Sequence[float]) -> tuple[set[float], set[float], set[float]]:
    if len(times) < 15:
        raise RuntimeError("At least 15 aligned time samples are required")
    train_end = max(2, round(len(times) * 0.60))
    validation_end = max(train_end + 1, round(len(times) * 0.80))
    validation_end = min(validation_end, len(times) - 1)
    return (
        set(times[:train_end]),
        set(times[train_end:validation_end]),
        set(times[validation_end:]),
    )


def select_graphs(
    graphs: Sequence[RegionGraphSample], times: set[float]
) -> list[RegionGraphSample]:
    return [graph for graph in graphs if graph.source_second in times]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_model(
    mode: str,
    graphs: Sequence[RegionGraphSample],
    train_times: set[float],
    validation_times: set[float],
    count_scale: float,
    zone_count: int,
    output_dir: Path,
    epochs: int,
    seed: int,
) -> tuple[RegionOccupancyGNN, dict[str, list[RegionGraphSample]]]:
    seed_everything(seed)
    split_graphs = {
        "train": select_graphs(graphs, train_times),
        "validation": select_graphs(graphs, validation_times),
    }
    model = RegionOccupancyGNN(
        input_size=len(BASE_FEATURE_NAMES) + zone_count,
        hidden_size=32,
        dropout=0.10,
    )
    print(f"\n[{mode}] training")
    best_epoch, best_mae = train_model(
        model=model,
        training=split_graphs["train"],
        validation=split_graphs["validation"],
        count_scale=count_scale,
        device=torch.device("cpu"),
        epochs=epochs,
        learning_rate=0.01,
        weight_decay=0.0005,
        patience=max(40, epochs // 6),
    )
    torch.save(
        {
            "mode": mode,
            "state_dict": model.state_dict(),
            "input_size": len(BASE_FEATURE_NAMES) + zone_count,
            "hidden_size": 32,
            "dropout": 0.10,
            "count_scale": count_scale,
            "best_epoch": best_epoch,
            "best_validation_mae": best_mae,
        },
        output_dir / f"{mode}_gnn.pt",
    )
    return model, split_graphs


def collect_predictions(
    mode: str,
    model: RegionOccupancyGNN,
    split: str,
    graphs: Sequence[RegionGraphSample],
    zones: Sequence[Zone],
    count_scale: float,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for graph in graphs:
        predicted = predict_counts(
            model, graph, count_scale, torch.device("cpu")
        )
        assert graph.target_counts is not None
        for zone in zones:
            zone_id = zone.zone_id
            actual = float(graph.target_counts[zone_id])
            prediction = float(predicted[zone_id])
            baseline = float(graph.current_counts[zone_id])
            rows.append(
                {
                    "input_mode": mode,
                    "data_split": split,
                    "source_second": graph.source_second,
                    "target_second": graph.target_second,
                    "zone_id": zone_id,
                    "zone": zone.name,
                    "current_input_count": int(baseline),
                    "actual_next_count": int(actual),
                    "predicted_next_count": round(prediction, 4),
                    "model_absolute_error": round(abs(prediction - actual), 4),
                    "persistence_absolute_error": round(abs(baseline - actual), 4),
                }
            )
    return rows


def metrics_for_rows(
    rows: Sequence[dict[str, str | int | float]], zones: Sequence[Zone]
) -> list[dict[str, str | int | float]]:
    results: list[dict[str, str | int | float]] = []
    groups = [("all", rows)] + [
        (
            zone.name,
            [row for row in rows if int(row["zone_id"]) == zone.zone_id],
        )
        for zone in zones
    ]
    for zone_name, group in groups:
        if not group:
            continue
        errors = np.array(
            [float(row["model_absolute_error"]) for row in group], dtype=float
        )
        squared = np.array(
            [
                (float(row["predicted_next_count"]) - float(row["actual_next_count"]))
                ** 2
                for row in group
            ],
            dtype=float,
        )
        baseline = np.array(
            [float(row["persistence_absolute_error"]) for row in group], dtype=float
        )
        results.append(
            {
                "input_mode": str(group[0]["input_mode"]),
                "data_split": str(group[0]["data_split"]),
                "zone": zone_name,
                "samples": len(group),
                "model_mae": round(float(errors.mean()), 4),
                "model_rmse": round(float(np.sqrt(squared.mean())), 4),
                "persistence_mae": round(float(baseline.mean()), 4),
                "mae_improvement": round(float(baseline.mean() - errors.mean()), 4),
            }
        )
    return results


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_prediction_chart(
    path: Path,
    prediction_rows: Sequence[dict[str, str | int | float]],
    zones: Sequence[Zone],
) -> None:
    test_rows = [row for row in prediction_rows if row["data_split"] == "test"]
    modes = sorted({str(row["input_mode"]) for row in test_rows})
    figure, axes = plt.subplots(len(zones), 1, figsize=(12, 14), sharex=True)
    for zone, axis in zip(zones, axes):
        by_mode = {
            mode: sorted(
                [
                    row
                    for row in test_rows
                    if row["input_mode"] == mode and row["zone_id"] == zone.zone_id
                ],
                key=lambda row: float(row["target_second"]),
            )
            for mode in modes
        }
        available = next((rows for rows in by_mode.values() if rows), [])
        axis.plot(
            [float(row["target_second"]) for row in available],
            [float(row["actual_next_count"]) for row in available],
            color="black",
            linewidth=2.2,
            marker="o",
            markersize=3,
            label="ground truth",
        )
        colors = {"ground_truth_input": "#2ca02c", "tracker_input": "#d6279f"}
        for mode, rows in by_mode.items():
            if not rows:
                continue
            axis.plot(
                [float(row["target_second"]) for row in rows],
                [float(row["predicted_next_count"]) for row in rows],
                linewidth=1.8,
                color=colors.get(mode),
                label=mode,
            )
        axis.set_title(f"Zone {zone.zone_id}: {zone.name}")
        axis.set_ylabel("people")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("target video time (seconds)")
    figure.suptitle("Town Centre: next-second occupancy on held-out final 20%", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def draw_zones(frame: np.ndarray, zones: Sequence[Zone]) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = frame.copy()
    for zone in zones:
        left, top, right, bottom = zone.rect
        first = (round(left * width), round(top * height))
        second = (round(right * width), round(bottom * height))
        cv2.rectangle(overlay, first, second, zone.color_bgr, -1)
    output = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0.0)
    for zone in zones:
        left, top, right, bottom = zone.rect
        y = max(28, round(top * height) + 28)
        cv2.line(
            output,
            (round(left * width), round(top * height)),
            (round(right * width), round(top * height)),
            zone.color_bgr,
            3,
        )
        cv2.putText(
            output,
            f"Z{zone.zone_id} {zone.name}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            zone.color_bgr,
            2,
            cv2.LINE_AA,
        )
    return output


def draw_snapshot(
    frame: np.ndarray,
    rows: dict[int, dict[str, str]],
    zones: Sequence[Zone],
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    output = draw_zones(frame, zones)
    assignments = assign_people(rows, output.shape[1], output.shape[0], zones)
    counts = zone_counts(assignments, len(zones))
    for track_id, row in rows.items():
        left = round(float(row["bbox_x"]))
        top = round(float(row["bbox_y"]))
        right = round(left + float(row["bbox_width"]))
        bottom = round(top + float(row["bbox_height"]))
        cv2.rectangle(output, (left, top), (right, bottom), color, 2)
        cv2.putText(
            output,
            str(track_id),
            (left, max(12, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(output, (0, 0), (output.shape[1], 52), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"{title} | total {len(rows)} | zone counts {counts}",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def make_detection_preview(
    path: Path,
    video_path: Path,
    preview_second: float,
    ground_truth: dict[float, dict[int, dict[str, str]]],
    tracker: dict[float, dict[int, dict[str, str]]] | None,
    zones: Sequence[Zone],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, preview_second * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read preview frame at {preview_second}s")
    gt_time = min(ground_truth, key=lambda value: abs(value - preview_second))
    left = draw_snapshot(
        frame.copy(), ground_truth[gt_time], zones, "Ground truth", (40, 230, 40)
    )
    if tracker:
        tracker_time = min(tracker, key=lambda value: abs(value - preview_second))
        right = draw_snapshot(
            frame.copy(), tracker[tracker_time], zones, "YOLO + BoT-SORT", (230, 60, 220)
        )
    else:
        right = frame.copy()
        cv2.putText(
            right,
            "Tracker CSV not supplied",
            (80, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    target_height = 540
    scale = target_height / left.shape[0]
    target_width = round(left.shape[1] * scale)
    left = cv2.resize(left, (target_width, target_height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), np.hstack((left, right)))


def current_count_mae(
    tracker_graphs: Sequence[RegionGraphSample],
    ground_truth_snapshots: dict[float, dict[int, dict[str, str]]],
    width: int,
    height: int,
    zones: Sequence[Zone],
) -> float:
    errors: list[float] = []
    for graph in tracker_graphs:
        target = target_at(ground_truth_snapshots, graph.source_second)
        if target is None:
            continue
        _, rows = target
        assignments = assign_people(rows, width, height, zones)
        gt_counts = torch.tensor(zone_counts(assignments, len(zones)))
        errors.extend((graph.current_counts - gt_counts).abs().tolist())
    return float(np.mean(errors)) if errors else float("nan")


def write_report(
    path: Path,
    summary: dict[str, object],
    overall_metrics: Sequence[dict[str, str | int | float]],
) -> None:
    lines = [
        "# Town Centre 예측 실험 결과",
        "",
        "## 데이터",
        "",
        f"- 영상: {summary['video_width']}×{summary['video_height']}, "
        f"{summary['video_fps']:.3f}fps, {summary['video_frames']}프레임",
        f"- 유효 `.top` 행: {summary['valid_top_rows']:,}개",
        f"- 잘못된 `.top` 행: {summary['malformed_top_rows']}개",
        f"- 정답 범위: 0~{summary['maximum_annotated_second']:.2f}초",
        f"- 정답 사람 ID: {summary['unique_people']}개",
        f"- 학습/검증/테스트 시점: {summary['train_times']}/"
        f"{summary['validation_times']}/{summary['test_times']}",
        "",
        "## 전체 영역 결과",
        "",
        "| 입력 | 분할 | 모델 MAE | 모델 RMSE | 현재값 유지 MAE | MAE 개선 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall_metrics:
        lines.append(
            f"| {row['input_mode']} | {row['data_split']} | "
            f"{float(row['model_mae']):.3f} | {float(row['model_rmse']):.3f} | "
            f"{float(row['persistence_mae']):.3f} | {float(row['mae_improvement']):+.3f} |"
        )
    if "tracker_current_count_mae" in summary:
        lines.extend(
            [
                "",
                "## 추적 입력 품질",
                "",
                f"- 같은 시각의 영역별 YOLO 추적 인원과 `.top` 정답 사이 MAE: "
                f"{float(summary['tracker_current_count_mae']):.3f}명",
            ]
        )
    lines.extend(
        [
            "",
            "시간 순서대로 앞 60%/다음 20%/마지막 20%를 나눴으며, "
            "테스트 구간은 학습과 조기 종료에 사용하지 않았습니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--tracker-csv", type=Path)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preview-second", type=float, default=60.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, top_stats = parse_top(args.top)
    fps, width, height, frame_count = video_metadata(args.video)
    configured_width, configured_height, zones, edges = load_graph_config(args.zones)
    if (configured_width, configured_height) != (width, height):
        raise RuntimeError(
            f"Zone resolution {configured_width}x{configured_height} does not match "
            f"video {width}x{height}"
        )

    ground_truth_csv = args.output_dir / "ground_truth_tracking_1s.csv"
    conversion_stats = write_ground_truth_tracking_csv(
        records,
        ground_truth_csv,
        fps,
        width,
        height,
        args.sample_interval,
    )
    ground_truth_snapshots = read_snapshots(ground_truth_csv)
    tracker_snapshots = read_snapshots(args.tracker_csv) if args.tracker_csv else None

    inputs: dict[str, dict[float, dict[int, dict[str, str]]]] = {
        "ground_truth_input": ground_truth_snapshots
    }
    if tracker_snapshots is not None:
        inputs["tracker_input"] = tracker_snapshots

    possible_times = [
        time
        for time in sorted(ground_truth_snapshots)
        if target_at(ground_truth_snapshots, time + args.horizon_seconds) is not None
    ]
    for snapshots in inputs.values():
        possible_times = [time for time in possible_times if time in snapshots]
    train_times, validation_times, test_times = split_times(possible_times)

    maximum_count = 0
    for snapshots in inputs.values():
        for rows in snapshots.values():
            assignments = assign_people(rows, width, height, zones)
            maximum_count = max(maximum_count, *zone_counts(assignments, len(zones)))
    for rows in ground_truth_snapshots.values():
        assignments = assign_people(rows, width, height, zones)
        maximum_count = max(maximum_count, *zone_counts(assignments, len(zones)))
    count_scale = float(max(10, maximum_count))

    all_predictions: list[dict[str, str | int | float]] = []
    all_metrics: list[dict[str, str | int | float]] = []
    models: dict[str, RegionOccupancyGNN] = {}
    graphs_by_mode: dict[str, list[RegionGraphSample]] = {}
    for mode, snapshots in inputs.items():
        graphs = build_supervised_graphs(
            snapshots,
            ground_truth_snapshots,
            width,
            height,
            zones,
            edges,
            args.horizon_seconds,
            count_scale,
        )
        graphs = select_graphs(graphs, set(possible_times))
        graphs_by_mode[mode] = graphs
        model, _ = train_one_model(
            mode,
            graphs,
            train_times,
            validation_times,
            count_scale,
            len(zones),
            args.output_dir,
            args.epochs,
            args.seed,
        )
        models[mode] = model
        for split, times in (
            ("train", train_times),
            ("validation", validation_times),
            ("test", test_times),
        ):
            predictions = collect_predictions(
                mode,
                model,
                split,
                select_graphs(graphs, times),
                zones,
                count_scale,
            )
            all_predictions.extend(predictions)
            all_metrics.extend(metrics_for_rows(predictions, zones))

    write_csv(args.output_dir / "predictions.csv", all_predictions)
    write_csv(args.output_dir / "metrics.csv", all_metrics)
    make_prediction_chart(
        args.output_dir / "test_predictions.png", all_predictions, zones
    )
    make_detection_preview(
        args.output_dir / "detection_comparison_60s.jpg",
        args.video,
        args.preview_second,
        ground_truth_snapshots,
        tracker_snapshots,
        zones,
    )

    summary: dict[str, object] = {
        "video_fps": fps,
        "video_width": width,
        "video_height": height,
        "video_frames": frame_count,
        "valid_top_rows": len(records),
        "malformed_top_rows": top_stats["malformed_rows"],
        "invalid_body_rows": top_stats["invalid_body_rows"],
        "unique_people": len({record.person_id for record in records}),
        **conversion_stats,
        "aligned_source_times": len(possible_times),
        "train_times": len(train_times),
        "validation_times": len(validation_times),
        "test_times": len(test_times),
        "count_scale": count_scale,
    }
    if "tracker_input" in graphs_by_mode:
        summary["tracker_current_count_mae"] = current_count_mae(
            select_graphs(graphs_by_mode["tracker_input"], test_times),
            ground_truth_snapshots,
            width,
            height,
            zones,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    overall_metrics = [row for row in all_metrics if row["zone"] == "all"]
    write_report(args.output_dir / "REPORT.md", summary, overall_metrics)

    print("\nOverall metrics")
    for row in overall_metrics:
        print(
            f"{row['input_mode']:>20} {row['data_split']:>10} | "
            f"MAE {float(row['model_mae']):.3f} | "
            f"persistence {float(row['persistence_mae']):.3f}"
        )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
