from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app_config import DEFAULT_CONFIG


def _odd_kernel(value: int, minimum: int = 1) -> int:
    value = max(int(value), minimum)
    if value % 2 == 0:
        value += 1
    return value


@dataclass
class DetectionResult:
    found: bool
    targets: list[dict[str, Any]]


class GreenLightDetector:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = {}
        self.debug_mask: np.ndarray | None = None
        self.debug_result: np.ndarray | None = None
        self.last_metrics: dict[str, Any] = {}
        self.set_config(config or DEFAULT_CONFIG.get("detection", {}))

    def set_config(self, config: dict[str, Any]) -> None:
        self.config = config or {}

    def current_hsv_range(self) -> tuple[np.ndarray, np.ndarray]:
        hsv_config = self.config.get("hsv", {})
        lower = np.array(hsv_config.get("lower", [24, 31, 123]), dtype=np.uint8)
        upper = np.array(hsv_config.get("upper", [83, 253, 255]), dtype=np.uint8)
        return lower, upper

    def detect_green_light(self, frame_rgb: np.ndarray) -> DetectionResult:
        lower_hsv, upper_hsv = self.current_hsv_range()
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

        green_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        value_channel = hsv[:, :, 2]
        min_v = int(self.config.get("brightness", {}).get("min_v", 180))
        bright_mask = cv2.threshold(value_channel, min_v, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(green_mask, bright_mask)
        mask = self._refine_mask(mask)
        self.debug_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            candidate = self._evaluate_candidate(contour, hsv, value_channel, frame_rgb.shape[:2])
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            self.last_metrics = {
                "candidate_count": len(contours),
                "lower_hsv": lower_hsv.tolist(),
                "upper_hsv": upper_hsv.tolist(),
                "reason": "no_valid_candidate",
            }
            self.debug_result = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            return DetectionResult(False, [])

        contour_config = self.config.get("contour", {})
        far_min_area = float(contour_config.get("far_min_area", contour_config.get("min_area", 60)))
        near_min_area = float(contour_config.get("near_min_area", 300))

        near_candidates = []
        far_candidates = []
        for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
            if candidate["area"] >= near_min_area:
                near_candidates.append(candidate)
            elif candidate["area"] >= far_min_area:
                far_candidates.append(candidate)

        selected_candidates = []
        if near_candidates:
            selected_candidates.append(("near", near_candidates[0]))
        if far_candidates:
            selected_candidates.append(("far", far_candidates[0]))

        if not selected_candidates:
            self.last_metrics = {
                "candidate_count": len(contours),
                "reason": "no_candidate_matched_near_far_threshold",
                "near_min_area": near_min_area,
                "far_min_area": far_min_area,
            }
            self.debug_result = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            return DetectionResult(False, [])

        frame_h, frame_w = frame_rgb.shape[:2]
        center_x = frame_w // 2
        center_y = frame_h // 2
        targets: list[dict[str, Any]] = []
        for index, (role, candidate) in enumerate(selected_candidates):
            target = {
                "contour": candidate["contour"],
                "center": candidate["center"],
                "area": candidate["area"],
                "bbox": candidate["bbox"],
                "circle": candidate["circle"],
                "score": candidate["score"],
                "metrics": candidate["metrics"],
                "offset": (
                    int(candidate["center"][0] - center_x),
                    int(candidate["center"][1] - center_y),
                ),
                "role": role,
                "index": index,
            }
            targets.append(target)

        self.last_metrics = {
            "target_count": len(targets),
            "targets": [target["metrics"] for target in targets],
        }
        self.debug_result = self._build_debug_result(frame_rgb, targets, candidates)
        return DetectionResult(bool(targets), targets)

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        morphology = self.config.get("morphology", {})
        blur_kernel = _odd_kernel(morphology.get("blur_kernel", 5))
        open_kernel = max(int(morphology.get("open_kernel", 3)), 1)
        close_kernel = max(int(morphology.get("close_kernel", 5)), 1)
        erode_iterations = max(int(morphology.get("erode_iterations", 0)), 0)
        dilate_iterations = max(int(morphology.get("dilate_iterations", 1)), 0)

        refined = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        open_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        close_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, open_element)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_element)
        if erode_iterations:
            refined = cv2.erode(refined, close_element, iterations=erode_iterations)
        if dilate_iterations:
            refined = cv2.dilate(refined, close_element, iterations=dilate_iterations)
        return refined

    def _evaluate_candidate(
        self,
        contour: np.ndarray,
        hsv: np.ndarray,
        value_channel: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> dict[str, Any] | None:
        contour_config = self.config.get("contour", {})
        brightness_config = self.config.get("brightness", {})
        circle_config = self.config.get("circle", {})
        scoring = self.config.get("scoring", {})

        area = float(cv2.contourArea(contour))
        if area < float(contour_config.get("min_area", 60)):
            return None
        if area > float(contour_config.get("max_area", 20000)):
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < float(contour_config.get("min_circularity", 0.55)):
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None
        aspect_ratio = float(min(w, h) / max(w, h))
        if aspect_ratio < float(contour_config.get("min_aspect_ratio", 0.55)):
            return None

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        min_radius = float(circle_config.get("min_radius", 4))
        max_radius = float(circle_config.get("max_radius", 120))
        if radius < min_radius or radius > max_radius:
            return None

        enclosing_area = np.pi * radius * radius
        fill_ratio = float(area / enclosing_area) if enclosing_area > 0 else 0.0
        if fill_ratio < float(contour_config.get("min_fill_ratio", 0.55)):
            return None

        contour_mask = np.zeros(value_channel.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        contour_values = value_channel[contour_mask == 255]
        contour_hsv = hsv[contour_mask == 255]
        if contour_values.size == 0 or contour_hsv.size == 0:
            return None

        mean_brightness = float(np.mean(contour_values))
        brightness_std = float(np.std(contour_values))
        if mean_brightness < float(brightness_config.get("min_v", 180)):
            return None
        if brightness_std > float(brightness_config.get("max_std", 90)):
            return None

        mean_h = float(np.mean(contour_hsv[:, 0]))
        mean_s = float(np.mean(contour_hsv[:, 1]))

        frame_h, frame_w = frame_shape
        center_distance = float(
            np.hypot(circle_x - frame_w / 2.0, circle_y - frame_h / 2.0)
        )
        max_distance = max(np.hypot(frame_w / 2.0, frame_h / 2.0), 1.0)
        center_score = max(0.0, 1.0 - center_distance / max_distance)

        score = (
            circularity * float(scoring.get("circularity_weight", 1.2))
            + fill_ratio * float(scoring.get("fill_ratio_weight", 1.0))
            + (mean_brightness / 255.0) * float(scoring.get("brightness_weight", 1.4))
            + center_score * float(scoring.get("center_weight", 0.35))
        )

        metrics = {
            "area": round(area, 2),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "fill_ratio": round(fill_ratio, 3),
            "radius": round(float(radius), 2),
            "mean_brightness": round(mean_brightness, 2),
            "brightness_std": round(brightness_std, 2),
            "mean_h": round(mean_h, 2),
            "mean_s": round(mean_s, 2),
            "center_score": round(center_score, 3),
            "score": round(score, 3),
        }

        return {
            "contour": contour,
            "bbox": (x, y, w, h),
            "center": (int(circle_x), int(circle_y)),
            "circle": (int(circle_x), int(circle_y), int(radius)),
            "area": area,
            "score": score,
            "metrics": metrics,
        }

    def _build_debug_result(
        self,
        frame_rgb: np.ndarray,
        selected_targets: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> np.ndarray:
        result = frame_rgb.copy()
        draw_rejected = bool(self.config.get("debug", {}).get("draw_rejected", False))

        if draw_rejected:
            for candidate in candidates:
                x, y, w, h = candidate["bbox"]
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 165, 0), 1)

        colors = [(0, 255, 0), (255, 165, 0)]
        for target in selected_targets:
            x, y, w, h = target["bbox"]
            center_x, center_y, radius = target["circle"]
            color = colors[min(target["index"], len(colors) - 1)]
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.drawContours(result, [target["contour"]], -1, color, 2)
            cv2.circle(result, (center_x, center_y), radius, (255, 255, 0), 2)
            cv2.circle(result, (center_x, center_y), 4, (255, 0, 0), -1)
            cv2.putText(
                result,
                "NEAR" if target["role"] == "near" else "FAR",
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return result


detector = GreenLightDetector()


def set_detection_config(config: dict[str, Any]) -> None:
    detector.set_config(config)


def get_trackbar_values() -> tuple[np.ndarray, np.ndarray]:
    return detector.current_hsv_range()


def create_trackbars() -> None:
    return None


def detect_green_light_and_offset(
    frame: np.ndarray,
    lower_hsv: np.ndarray | None = None,
    upper_hsv: np.ndarray | None = None,
) -> tuple[bool, tuple[int, int], dict[str, Any] | None]:
    if lower_hsv is not None and upper_hsv is not None:
        detector.set_config(
            {
                **detector.config,
                "hsv": {
                    "lower": lower_hsv.tolist(),
                    "upper": upper_hsv.tolist(),
                },
            }
        )

    result = detector.detect_green_light(frame)
    primary_target = result.targets[0] if result.targets else None
    primary_offset = primary_target["offset"] if primary_target is not None else (0, 0)
    return result.found, primary_offset, primary_target


def detect_green_targets(
    frame: np.ndarray,
    lower_hsv: np.ndarray | None = None,
    upper_hsv: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if lower_hsv is not None and upper_hsv is not None:
        detector.set_config(
            {
                **detector.config,
                "hsv": {
                    "lower": lower_hsv.tolist(),
                    "upper": upper_hsv.tolist(),
                },
            }
        )
    return detector.detect_green_light(frame).targets


def get_debug_images() -> tuple[np.ndarray | None, np.ndarray | None]:
    return detector.debug_mask, detector.debug_result
