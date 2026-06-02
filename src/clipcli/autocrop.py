from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class CropAnchor:
    x: float
    detections: int


def detect_speaker_crop_x(
    source: Path,
    *,
    start: float,
    duration: float,
    samples: int = 7,
) -> CropAnchor | None:
    try:
        import cv2
    except ImportError:
        return None

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return None

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        return None

    cascades = _load_cascades(cv2)
    if not cascades:
        capture.release()
        return None

    sample_count = max(1, samples)
    sample_offsets = _sample_offsets(duration, sample_count)
    frame_shape: tuple[int, int] | None = None
    for offset in sample_offsets:
        at = start + offset
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, at) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        frame_shape = (width, height)
        centers = _detect_frame_centers(cv2, cascades, frame)
        if centers:
            center_x = _weighted_median_center(centers)
            capture.release()
            return CropAnchor(x=_center_to_crop_anchor(center_x, width, height), detections=len(centers))
    capture.release()

    if frame_shape is None:
        return None
    return None


def _sample_offsets(duration: float, samples: int) -> list[float]:
    initial = [0.0, min(0.25, duration / 4), min(0.5, duration / 3)]
    if samples <= 1:
        return [0.0]
    early = min(duration * 0.08, 1.5)
    span = max(0.0, duration * 0.35 - early)
    later = [early + span * (index / (samples - 1)) for index in range(samples)]
    offsets = []
    for offset in initial + later:
        if offset not in offsets:
            offsets.append(offset)
    return offsets


def _weighted_median_center(centers: list[tuple[float, float]]) -> float:
    expanded: list[float] = []
    for center_x, weight in sorted(centers, key=lambda item: item[0]):
        expanded.extend([center_x] * max(1, int(weight)))
    return float(median(expanded or [item[0] for item in centers]))


def _load_cascades(cv2):
    names = [
        "haarcascade_profileface.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",
    ]
    cascades = []
    for name in names:
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        if not cascade.empty():
            cascades.append(cascade)
    return cascades


def _detect_frame_centers(cv2, cascades, frame) -> list[tuple[float, float]]:
    height, width = frame.shape[:2]
    max_width = 640
    scale = min(1.0, max_width / width)
    if scale < 1:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    small_height, small_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detections: list[tuple[float, float]] = []
    min_size = max(24, int(small_width * 0.035))
    for cascade in cascades:
        boxes = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        for x, y, w, h in boxes:
            center_y = y + h / 2
            if center_y > small_height * 0.62:
                continue
            center_x_original = (x + w / 2) / scale
            area_weight = (w * h) / max(1, small_width * small_height)
            detections.append((center_x_original, max(1.0, area_weight * 1000)))
    return detections


def _center_to_crop_anchor(center_x: float, width: int, height: int) -> float:
    scaled_width = width * (1920 / height)
    scaled_center = center_x * (1920 / height)
    max_crop_x = max(1.0, scaled_width - 1080)
    crop_left = min(max_crop_x, max(0.0, scaled_center - 540))
    return min(1.0, max(0.0, crop_left / max_crop_x))
