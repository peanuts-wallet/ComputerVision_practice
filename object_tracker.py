#!/usr/bin/env python3
"""Assign IDs to objects in a video and export one CSV snapshot per interval."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on the local environment
    raise SystemExit(
        "OpenCV가 설치되어 있지 않습니다. "
        "'python3 -m pip install -r requirements.txt'를 먼저 실행하세요."
    ) from exc


BBox = tuple[int, int, int, int]
Point = tuple[float, float]
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL = PROJECT_DIR / "models" / "yolo26s.pt"
DEFAULT_TRACKER_CONFIG = PROJECT_DIR / "configs" / "botsort_persistent.yaml"


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    label: str = "object"
    confidence: float = 1.0
    external_id: int | None = None

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    label: str
    confidence: float
    first_frame: int
    last_detected_frame: int
    hits: int = 1
    missed: int = 0
    velocity: Point = (0.0, 0.0)
    peak_confidence: float = 0.0
    last_sample_center: Point | None = None
    last_sample_time: float | None = None

    def __post_init__(self) -> None:
        self.peak_confidence = max(self.peak_confidence, self.confidence)

    @property
    def center(self) -> Point:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0

    @property
    def predicted_center(self) -> Point:
        cx, cy = self.center
        vx, vy = self.velocity
        return cx + vx, cy + vy

    def update(
        self,
        detection: Detection,
        frame_index: int,
        bbox_smoothing: float = 0.78,
    ) -> None:
        old_cx, old_cy = self.center
        new_cx, new_cy = detection.center
        measured_velocity = (new_cx - old_cx, new_cy - old_cy)
        # 속도를 부드럽게 하여 짧은 검출 누락 때 ID가 덜 바뀌도록 한다.
        self.velocity = (
            0.65 * self.velocity[0] + 0.35 * measured_velocity[0],
            0.65 * self.velocity[1] + 0.35 * measured_velocity[1],
        )
        self.bbox = tuple(
            round(bbox_smoothing * new + (1.0 - bbox_smoothing) * old)
            for old, new in zip(self.bbox, detection.bbox)
        )
        self.confidence = detection.confidence
        self.peak_confidence = max(self.peak_confidence, detection.confidence)
        self.last_detected_frame = frame_index
        self.hits += 1
        self.missed = 0

    def mark_missed(self) -> None:
        self.missed += 1
        x, y, w, h = self.bbox
        vx, vy = self.velocity
        self.bbox = (round(x + vx), round(y + vy), w, h)
        # 검출이 오래 끊겼을 때 박스가 화면 밖으로 계속 날아가지 않게 감쇠한다.
        self.velocity = (vx * 0.94, vy * 0.94)


class MotionDetector:
    def __init__(
        self,
        min_area: int,
        max_area_ratio: float,
        history: int,
        var_threshold: float,
    ) -> None:
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    def detect(self, frame) -> list[Detection]:
        mask = self.subtractor.apply(frame)
        # MOG2 그림자 값(127)을 제거하고 실제 전경(255)만 사용한다.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)
        mask = cv2.dilate(mask, self.kernel_open, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = frame.shape[0] * frame.shape[1]
        max_area = frame_area * self.max_area_ratio
        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= max_area:
                detections.append(Detection(cv2.boundingRect(contour)))
        return detections


class PersonDetector:
    """OpenCV에 포함된 HOG 보행자 검출기(추가 모델 파일 불필요)."""

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame) -> list[Detection]:
        boxes, weights = self.hog.detectMultiScale(
            frame,
            hitThreshold=0.0,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        candidates: list[BBox] = []
        scores: list[float] = []
        for box, weight in zip(boxes, weights):
            score = float(weight)
            if score >= self.confidence:
                candidates.append(tuple(int(value) for value in box))
                scores.append(score)

        if not candidates:
            return []
        kept = cv2.dnn.NMSBoxes(candidates, scores, self.confidence, 0.35)
        indices = [int(index) for index in kept]
        return [
            Detection(candidates[index], "person", scores[index]) for index in indices
        ]


class YoloDetector:
    """Ultralytics YOLO + BoT-SORT detector that supplies persistent IDs."""

    def __init__(
        self,
        model_path: str,
        tracker_config: str,
        classes: Sequence[int] | None,
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO 모드에는 ultralytics가 필요합니다. "
                "requirements.txt를 설치하세요."
            ) from exc

        model = Path(model_path).expanduser()
        if not model.is_file():
            raise FileNotFoundError(f"YOLO 모델을 찾을 수 없습니다: {model}")
        tracker_path = Path(tracker_config).expanduser()
        # botsort.yaml/byetrack.yaml처럼 Ultralytics 내장 설정 이름도 허용한다.
        self._tracker_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.tracker_config = self._prepare_tracker_config(
            tracker_path, tracker_config
        )
        self.model = YOLO(str(model.resolve()))
        self.classes = list(classes) if classes is not None else None
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device

    def _prepare_tracker_config(
        self, tracker_path: Path, tracker_config: str
    ) -> str:
        """Resolve a custom YAML's relative ReID model path against the YAML."""
        if not tracker_path.is_file():
            return tracker_config
        try:
            import yaml
        except ImportError:
            return str(tracker_path.resolve())

        config = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise RuntimeError(f"트래커 YAML 형식이 올바르지 않습니다: {tracker_path}")
        reid_model = config.get("model")
        if not isinstance(reid_model, str) or reid_model == "auto":
            return str(tracker_path.resolve())

        reid_path = Path(reid_model).expanduser()
        if reid_path.is_absolute():
            return str(tracker_path.resolve())
        resolved_reid = (tracker_path.parent / reid_path).resolve()
        if not resolved_reid.is_file():
            raise FileNotFoundError(
                "트래커 YAML의 ReID 모델을 찾을 수 없습니다: "
                f"{resolved_reid}"
            )

        config["model"] = str(resolved_reid)
        self._tracker_temp_dir = tempfile.TemporaryDirectory(
            prefix="opencv_tracker_"
        )
        runtime_config = Path(self._tracker_temp_dir.name) / tracker_path.name
        runtime_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        return str(runtime_config)

    def detect(self, frame) -> list[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            classes=self.classes,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        track_ids = boxes.id.int().cpu().tolist()
        detections: list[Detection] = []
        for xyxy, confidence, class_id, track_id in zip(
            coordinates, confidences, class_ids, track_ids
        ):
            x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
            bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            label = str(result.names.get(class_id, class_id))
            detections.append(
                Detection(bbox, label, float(confidence), int(track_id))
            )
        return detections


class MultiObjectTracker:
    def __init__(
        self,
        max_distance: float,
        min_iou: float,
        max_missed: int,
        min_hits: int,
        max_relink_missed: int = 15,
        bbox_smoothing: float = 0.78,
    ) -> None:
        self.max_distance = max_distance
        self.min_iou = min_iou
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.max_relink_missed = max_relink_missed
        self.bbox_smoothing = bbox_smoothing
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.external_to_track: dict[int, int] = {}

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        honor_external_ids: bool = False,
    ) -> list[Track]:
        if honor_external_ids:
            return self._update_external_ids(detections, frame_index)

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            predicted = track.predicted_center
            for detection_index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                distance = point_distance(predicted, detection.center)
                overlap = bbox_iou(track.bbox, detection.bbox)
                if distance <= self.max_distance or overlap >= self.min_iou:
                    # 작은 값이 우선: 가까운 거리와 큰 IoU를 함께 선호한다.
                    cost = distance / self.max_distance - 0.7 * overlap
                    candidates.append((cost, track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self.tracks[track_id].update(
                detections[detection_index], frame_index, self.bbox_smoothing
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track.mark_missed()
            if track.missed > self.max_missed:
                del self.tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = Track(
                track_id=self.next_id,
                bbox=detection.bbox,
                label=detection.label,
                confidence=detection.confidence,
                first_frame=frame_index,
                last_detected_frame=frame_index,
            )
            self.tracks[self.next_id] = track
            self.next_id += 1

        return self.confirmed_tracks(visible_only=False)

    def _update_external_ids(
        self, detections: Sequence[Detection], frame_index: int
    ) -> list[Track]:
        """Stabilize BoT-SORT IDs and reconnect brief detector dropouts."""
        seen_track_ids: set[int] = set()
        pending: list[Detection] = []

        # 먼저 이미 알려진 외부 ID를 내부의 안정된 ID에 연결한다.
        for detection in detections:
            if detection.external_id is None:
                continue
            stable_id = self.external_to_track.get(detection.external_id)
            if (
                stable_id is not None
                and stable_id in self.tracks
                and stable_id not in seen_track_ids
            ):
                self.tracks[stable_id].update(
                    detection, frame_index, self.bbox_smoothing
                )
                seen_track_ids.add(stable_id)
            else:
                pending.append(detection)

        # BoT-SORT가 새 ID를 냈더라도 직전 박스와 가까우면 기존 ID로 재연결한다.
        candidates: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(pending):
            for stable_id, track in self.tracks.items():
                if stable_id in seen_track_ids:
                    continue
                if track.label != detection.label:
                    continue
                if track.missed > self.max_relink_missed:
                    continue
                distance = point_distance(track.predicted_center, detection.center)
                overlap = bbox_iou(track.bbox, detection.bbox)
                old_area = max(1, track.bbox[2] * track.bbox[3])
                new_area = max(1, detection.bbox[2] * detection.bbox[3])
                area_ratio = min(old_area, new_area) / max(old_area, new_area)
                distance_gate = self.max_distance * (
                    1.0 + min(track.missed, self.max_relink_missed) * 0.08
                )
                if area_ratio >= 0.35 and (
                    distance <= distance_gate or overlap >= self.min_iou
                ):
                    cost = (
                        distance / distance_gate
                        - 0.8 * overlap
                        + 0.2 * (1.0 - area_ratio)
                        + 0.01 * track.missed
                    )
                    candidates.append((cost, stable_id, detection_index))

        matched_pending: set[int] = set()
        for _, stable_id, detection_index in sorted(candidates):
            if stable_id in seen_track_ids or detection_index in matched_pending:
                continue
            detection = pending[detection_index]
            self.tracks[stable_id].update(
                detection, frame_index, self.bbox_smoothing
            )
            self._bind_external_id(detection.external_id, stable_id)
            seen_track_ids.add(stable_id)
            matched_pending.add(detection_index)

        # 재연결할 대상이 없는 검출만 새 안정 ID로 만든다.
        for detection_index, detection in enumerate(pending):
            if detection_index in matched_pending or detection.external_id is None:
                continue
            stable_id = self._new_stable_id(detection.external_id)
            self.tracks[stable_id] = Track(
                track_id=stable_id,
                bbox=detection.bbox,
                label=detection.label,
                confidence=detection.confidence,
                first_frame=frame_index,
                last_detected_frame=frame_index,
                # 몇 프레임 유지된 트랙만 표시하여 순간적인 오검출을 거른다.
                hits=1,
            )
            self._bind_external_id(detection.external_id, stable_id)
            seen_track_ids.add(stable_id)

        for track_id, track in list(self.tracks.items()):
            if track_id not in seen_track_ids:
                track.mark_missed()
            if track.missed > self.max_missed:
                del self.tracks[track_id]
                self.external_to_track = {
                    external_id: stable_id
                    for external_id, stable_id in self.external_to_track.items()
                    if stable_id != track_id
                }

        return self.confirmed_tracks(visible_only=False)

    def _new_stable_id(self, preferred_id: int) -> int:
        # 삭제된 과거 ID를 다시 사용하지 않아 CSV에서 서로 다른 사람이 합쳐지는
        # 일을 막는다. 아직 한 번도 사용하지 않은 외부 ID만 그대로 채택한다.
        if preferred_id >= self.next_id and preferred_id not in self.tracks:
            self.next_id = max(self.next_id, preferred_id + 1)
            return preferred_id
        while self.next_id in self.tracks:
            self.next_id += 1
        stable_id = self.next_id
        self.next_id += 1
        return stable_id

    def _bind_external_id(self, external_id: int | None, stable_id: int) -> None:
        if external_id is None:
            return
        self.external_to_track = {
            known_external: known_stable
            for known_external, known_stable in self.external_to_track.items()
            if known_stable != stable_id
        }
        self.external_to_track[external_id] = stable_id

    def confirmed_tracks(self, visible_only: bool = True) -> list[Track]:
        tracks = [
            track
            for track in self.tracks.values()
            if track.hits >= self.min_hits and (not visible_only or track.missed == 0)
        ]
        return sorted(tracks, key=lambda item: item.track_id)


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def id_color(track_id: int) -> tuple[int, int, int]:
    """같은 ID에 항상 같은, 눈에 잘 띄는 BGR 색상을 돌려준다."""
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


def draw_tracks(frame, tracks: Iterable[Track]) -> None:
    frame_height, frame_width = frame.shape[:2]
    for track in tracks:
        x, y, w, h = clip_bbox(track.bbox, frame_width, frame_height)
        if w <= 1 or h <= 1:
            continue
        color = id_color(track.track_id)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"ID {track.track_id} | {track.label}"
        if track.missed:
            label += " | predicted"
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def clip_bbox(bbox: BBox, frame_width: int, frame_height: int) -> BBox:
    x, y, w, h = bbox
    left = max(0, min(frame_width - 1, x))
    top = max(0, min(frame_height - 1, y))
    right = max(0, min(frame_width, x + w))
    bottom = max(0, min(frame_height, y + h))
    return left, top, max(0, right - left), max(0, bottom - top)


def select_output_tracks(
    tracks: Sequence[Track],
    prediction_frames: int,
    prediction_min_hits: int = 6,
    prediction_min_confidence: float = 0.20,
    detection_overlap_threshold: float = 0.85,
    prediction_overlap_threshold: float = 0.15,
) -> list[Track]:
    """Keep reliable detections and only short, well-established predictions."""
    detected_candidates = sorted(
        (track for track in tracks if track.missed == 0),
        key=lambda track: (track.first_frame, -track.confidence, track.track_id),
    )
    selected: list[Track] = []
    occupied_boxes: list[BBox] = []
    for track in detected_candidates:
        if any(
            bbox_iou(track.bbox, occupied) >= detection_overlap_threshold
            for occupied in occupied_boxes
        ):
            continue
        selected.append(track)
        occupied_boxes.append(track.bbox)

    predicted = sorted(
        (
            track
            for track in tracks
            if 0 < track.missed <= prediction_frames
            and track.hits >= prediction_min_hits
            and track.peak_confidence >= prediction_min_confidence
        ),
        key=lambda track: (track.missed, track.track_id),
    )
    for track in predicted:
        if any(
            bbox_iou(track.bbox, occupied) >= prediction_overlap_threshold
            for occupied in occupied_boxes
        ):
            continue
        selected.append(track)
        occupied_boxes.append(track.bbox)
    return sorted(selected, key=lambda track: track.track_id)


CSV_FIELDS = [
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
]


def write_track_samples(
    writer: csv.DictWriter,
    tracks: Iterable[Track],
    sample_time: float,
    video_time: float,
    frame_index: int,
    fps: float,
) -> None:
    for track in tracks:
        center = track.center
        if track.last_sample_center is None or track.last_sample_time is None:
            displacement = 0.0
            speed = 0.0
        else:
            displacement = point_distance(center, track.last_sample_center)
            elapsed = sample_time - track.last_sample_time
            speed = displacement / elapsed if elapsed > 0 else 0.0

        x, y, w, h = track.bbox
        writer.writerow(
            {
                "sample_second": round(sample_time, 3),
                "video_time_s": round(video_time, 3),
                "frame_index": frame_index,
                "track_id": track.track_id,
                "label": track.label,
                "confidence": round(track.confidence, 4),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
                "center_x": round(center[0], 2),
                "center_y": round(center[1], 2),
                "area_px": w * h,
                "displacement_px": round(displacement, 2),
                "speed_px_s": round(speed, 2),
                "track_age_s": round((frame_index - track.first_frame) / fps, 3),
                "tracking_state": "predicted" if track.missed else "detected",
                "missed_frames": track.missed,
            }
        )
        track.last_sample_center = center
        track.last_sample_time = sample_time


def make_detector(args: argparse.Namespace):
    if args.detector == "person":
        return PersonDetector(args.person_confidence)
    if args.detector == "yolo":
        return YoloDetector(
            model_path=args.model,
            tracker_config=args.tracker_config,
            classes=args.classes,
            confidence=args.yolo_confidence,
            iou=args.yolo_iou,
            image_size=args.imgsz,
            device=args.device,
        )
    return MotionDetector(
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        history=args.background_history,
        var_threshold=args.var_threshold,
    )


def process_video(args: argparse.Namespace) -> tuple[int, int]:
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {input_path}")

    csv_path = Path(args.csv).expanduser()
    video_path = Path(args.output).expanduser()
    resolved_paths = {
        "input": input_path.resolve(),
        "csv": csv_path.resolve(),
        "output": video_path.resolve(),
    }
    if len(set(resolved_paths.values())) != len(resolved_paths):
        raise RuntimeError(
            "입력 영상, CSV, 출력 영상 경로는 서로 달라야 합니다: "
            f"{resolved_paths}"
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
        print("경고: 영상 FPS를 읽지 못해 30 FPS로 처리합니다.", file=sys.stderr)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("영상의 프레임 크기를 읽을 수 없습니다.")

    if (
        args.detector == "yolo"
        and args.auto_resolution
        and max(width, height) >= 3000
        and args.imgsz < 1920
    ):
        print(
            f"고해상도 영상({width}x{height})을 감지하여 "
            f"YOLO 추론 크기를 {args.imgsz}에서 1920으로 자동 조정합니다."
        )
        args.imgsz = 1920

    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(f"출력 영상을 만들 수 없습니다: {video_path}")

    detector = make_detector(args)
    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        min_iou=args.min_iou,
        max_missed=max(1, round(args.max_lost_seconds * fps)),
        min_hits=args.min_hits,
        max_relink_missed=max(1, round(args.relink_seconds * fps)),
        bbox_smoothing=args.bbox_smoothing,
    )
    prediction_frames = max(1, round(args.prediction_seconds * fps))

    frame_index = 0
    next_sample_time = 0.0
    samples_written = 0
    stopped_by_user = False

    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                video_time = frame_index / fps
                detections = detector.detect(frame)
                tracks = tracker.update(
                    detections,
                    frame_index,
                    honor_external_ids=args.detector == "yolo",
                )
                active_tracks = tracker.confirmed_tracks(visible_only=False)
                output_tracks = select_output_tracks(
                    active_tracks,
                    prediction_frames,
                    prediction_min_hits=args.prediction_min_hits,
                    prediction_min_confidence=args.prediction_min_confidence,
                )
                detected_count = sum(track.missed == 0 for track in output_tracks)

                # 샘플 시각을 지난 첫 프레임에서 기록하므로 가변 FPS/소수 FPS도 처리한다.
                while video_time + (0.5 / fps) >= next_sample_time:
                    write_track_samples(
                        csv_writer,
                        output_tracks,
                        next_sample_time,
                        video_time,
                        frame_index,
                        fps,
                    )
                    samples_written += len(output_tracks)
                    csv_file.flush()
                    next_sample_time += args.sample_interval

                draw_tracks(frame, output_tracks)
                cv2.putText(
                    frame,
                    f"time {video_time:.1f}s | tracks {len(output_tracks)} | "
                    f"detected {detected_count}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                video_writer.write(frame)
                frame_index += 1

                if args.show:
                    cv2.imshow("OpenCV Object Tracker - q: quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped_by_user = True
                        break
    finally:
        capture.release()
        video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if stopped_by_user:
        print("사용자 요청으로 처리를 일찍 종료했습니다.")
    return frame_index, samples_written


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("0보다 큰 값이어야 합니다.")
    return number


def ratio(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("0보다 크고 1 이하여야 합니다.")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상 객체에 ID를 붙여 추적하고 초 단위 CSV와 표시 영상을 생성합니다."
    )
    parser.add_argument("input", help="입력 영상 경로")
    parser.add_argument("--csv", default="tracking.csv", help="CSV 출력 경로")
    parser.add_argument(
        "--output", default="tracked_output.mp4", help="ID가 표시된 MP4 출력 경로"
    )
    parser.add_argument(
        "--detector",
        choices=("motion", "person", "yolo"),
        default="yolo",
        help="motion: 움직이는 객체, person: HOG 사람, yolo: YOLO+BoT-SORT",
    )
    parser.add_argument(
        "--sample-interval",
        type=positive_float,
        default=1.0,
        help="CSV 기록 간격(초, 기본 1.0)",
    )
    parser.add_argument(
        "--min-area", type=int, default=900, help="motion 최소 객체 면적(px²)"
    )
    parser.add_argument(
        "--max-area-ratio",
        type=ratio,
        default=0.65,
        help="motion 최대 객체 면적/전체 화면 비율",
    )
    parser.add_argument(
        "--background-history", type=int, default=300, help="motion 배경 학습 프레임 수"
    )
    parser.add_argument(
        "--var-threshold", type=positive_float, default=24.0, help="motion 민감도 기준"
    )
    parser.add_argument(
        "--person-confidence",
        type=float,
        default=0.5,
        help="person 최소 신뢰도",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_YOLO_MODEL),
        help="yolo 모드의 .pt 모델 경로",
    )
    parser.add_argument(
        "--tracker-config",
        default=str(DEFAULT_TRACKER_CONFIG),
        help="yolo 트래커 YAML 경로 또는 Ultralytics 내장 설정 이름",
    )
    parser.add_argument(
        "--classes",
        default="0",
        help="yolo 클래스 번호(쉼표 구분, 기본 0=person, all=전체)",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.02,
        help="yolo 최소 검출 신뢰도",
    )
    parser.add_argument(
        "--yolo-iou", type=float, default=0.7, help="yolo NMS IoU 기준"
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="yolo 추론 이미지 크기"
    )
    parser.add_argument(
        "--auto-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="고해상도 영상의 yolo 추론 크기를 1920까지 자동 상향",
    )
    parser.add_argument(
        "--device", default="cpu", help="yolo 실행 장치(cpu, mps, 0 등)"
    )
    parser.add_argument(
        "--max-distance",
        type=positive_float,
        default=100.0,
        help="ID 연결을 허용할 최대 중심 거리(px)",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.05,
        help="거리가 멀 때 ID 연결을 허용할 최소 IoU",
    )
    parser.add_argument(
        "--max-lost-seconds",
        type=positive_float,
        default=2.0,
        help="검출이 끊겨도 ID를 보존할 시간(초)",
    )
    parser.add_argument(
        "--relink-seconds",
        type=positive_float,
        default=0.75,
        help="새 외부 ID를 기존 ID에 재연결할 최대 누락 시간(초)",
    )
    parser.add_argument(
        "--prediction-seconds",
        type=positive_float,
        default=0.2,
        help="실제 검출 누락 중 예측 박스를 표시·기록할 시간(초)",
    )
    parser.add_argument(
        "--prediction-min-hits",
        type=int,
        default=6,
        help="예측 박스를 허용하기 전에 필요한 실제 검출 횟수",
    )
    parser.add_argument(
        "--prediction-min-confidence",
        type=float,
        default=0.20,
        help="예측 박스를 허용할 트랙의 누적 최고 신뢰도",
    )
    parser.add_argument(
        "--bbox-smoothing",
        type=float,
        default=0.78,
        help="새 검출 박스 반영 비율(0 초과 1 이하)",
    )
    parser.add_argument(
        "--min-hits", type=int, default=3, help="실제 객체로 확정할 연속 검출 수"
    )
    parser.add_argument("--show", action="store_true", help="처리 화면을 실시간 표시")
    args = parser.parse_args(argv)

    if args.min_area <= 0:
        parser.error("--min-area는 0보다 커야 합니다.")
    if args.background_history <= 0:
        parser.error("--background-history는 0보다 커야 합니다.")
    if args.min_hits <= 0:
        parser.error("--min-hits는 0보다 커야 합니다.")
    if args.prediction_min_hits <= 0:
        parser.error("--prediction-min-hits는 0보다 커야 합니다.")
    if not 0 <= args.min_iou <= 1:
        parser.error("--min-iou는 0 이상 1 이하여야 합니다.")
    if not 0 <= args.person_confidence <= 1:
        parser.error("--person-confidence는 0 이상 1 이하여야 합니다.")
    if not 0 <= args.yolo_confidence <= 1:
        parser.error("--yolo-confidence는 0 이상 1 이하여야 합니다.")
    if not 0 <= args.yolo_iou <= 1:
        parser.error("--yolo-iou는 0 이상 1 이하여야 합니다.")
    if not 0 <= args.prediction_min_confidence <= 1:
        parser.error("--prediction-min-confidence는 0 이상 1 이하여야 합니다.")
    if args.imgsz <= 0:
        parser.error("--imgsz는 0보다 커야 합니다.")
    if not 0 < args.bbox_smoothing <= 1:
        parser.error("--bbox-smoothing은 0보다 크고 1 이하여야 합니다.")
    if args.classes.strip().lower() == "all":
        args.classes = None
    else:
        try:
            args.classes = [
                int(item.strip()) for item in args.classes.split(",") if item.strip()
            ]
        except ValueError:
            parser.error("--classes는 쉼표로 구분한 정수 또는 all이어야 합니다.")
        if not args.classes or any(item < 0 for item in args.classes):
            parser.error("--classes에는 0 이상의 클래스 번호가 필요합니다.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frames, rows = process_video(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    print(f"완료: {frames}개 프레임 처리, CSV {rows}개 행 기록")
    print(f"CSV: {Path(args.csv).expanduser().resolve()}")
    print(f"추적 영상: {Path(args.output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
