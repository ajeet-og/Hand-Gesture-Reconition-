"""
Webcam hand-gesture recognizer with MediaPipe Hands.

Features:
- Learns gesture references from images placed in the `gestures/` folder.
- Compares live hand pose to reference poses; “close enough” matches count.
- Shows webcam feed with detected label and displays the matched reference image.
- Easy to extend: drop more reference images; filenames become labels.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

GESTURES_DIR = "gestures"


@dataclass
class ReferenceGesture:
    label: str
    landmarks: np.ndarray  # shape (21, 3), normalized
    image: np.ndarray      # original image as loaded


@dataclass
class PendingGesture:
    label: str
    image: np.ndarray      # original image; landmarks to be captured live


def normalize_landmarks(landmarks) -> np.ndarray:
    """Normalize landmark array to be translation and scale invariant."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]  # use wrist as origin
    scale = np.linalg.norm(pts, axis=1).max()
    scale = scale if scale > 1e-6 else 1.0
    return pts / scale


def load_reference_gestures(
    gestures_dir: str, hands_static
) -> Tuple[List[ReferenceGesture], List[PendingGesture]]:
    """Scan gestures_dir for images, extract landmarks, build reference set, and collect skipped ones."""
    references: List[ReferenceGesture] = []
    pending: List[PendingGesture] = []
    if not os.path.isdir(gestures_dir):
        print(f"[warn] gestures directory '{gestures_dir}' not found.")
        return references, pending

    image_files = [
        f
        for f in os.listdir(gestures_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    if not image_files:
        print(f"[warn] No reference images found in '{gestures_dir}'.")
        return references, pending

    for fname in image_files:
        path = os.path.join(gestures_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[warn] Could not read {path}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands_static.process(rgb)
        label = os.path.splitext(fname)[0]
        if not res.multi_hand_landmarks:
            print(f"[warn] No hand detected in {fname}; will capture live. Label='{label}'")
            pending.append(PendingGesture(label=label, image=img))
            continue
        norm = normalize_landmarks(res.multi_hand_landmarks[0].landmark)
        references.append(ReferenceGesture(label=label, landmarks=norm, image=img))
        print(f"[info] Loaded reference: {label}")
    return references, pending


def match_gesture(
    current_landmarks: np.ndarray,
    references: List[ReferenceGesture],
    threshold: float = 2.0,
) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
    """
    Compare current normalized landmarks to references.
    Returns (label, similarity_score, matched_image) or (None, 0, None).
    Lower distance means closer; similarity = 1 - distance.
    """
    if not references:
        return None, 0.0, None

    best_ref = None
    best_dist = 1e9
    for ref in references:
        dist = np.linalg.norm(current_landmarks - ref.landmarks)
        if dist < best_dist:
            best_dist = dist
            best_ref = ref

    if best_ref is not None and best_dist <= threshold:
        similarity = max(0.0, 1.0 - best_dist)
        return best_ref.label, similarity, best_ref.image
    return None, 0.0, None


def main() -> None:
    mp_hands = mp.solutions.hands
    hands_static = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
    )
    references, pending = load_reference_gestures(GESTURES_DIR, hands_static)
    if not references:
        print("No reference gestures loaded. Add images to the 'gestures' folder.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    last_label: Optional[str] = None
    last_image: Optional[np.ndarray] = None

    # If any reference images were skipped (no landmarks), capture live landmarks for them first.
    if pending:
        print(f"[info] {len(pending)} reference image(s) need live capture.")
        print("For each skipped image, a side-by-side view will show: left=webcam, right=original image.")
        print("Hold the matching gesture and press SPACE to capture landmarks for that image.")
        print("Press 's' to skip an item, or 'q' to exit capture mode early.")
        try:
            while pending:
                pending_item = pending[0]

                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                # Build side-by-side view: left = webcam, right = pending original image letterboxed.
                target_h, target_w = frame.shape[:2]
                box_w = max(200, int(target_h * 0.75))
                right_box = np.zeros((target_h, box_w, 3), dtype=np.uint8)
                pmh, pmw = pending_item.image.shape[:2]
                pscale = min(box_w / max(1, pmw), target_h / max(1, pmh))
                pnw, pnh = max(1, int(pmw * pscale)), max(1, int(pmh * pscale))
                presized = cv2.resize(pending_item.image, (pnw, pnh))
                p_pad_x = (box_w - pnw) // 2
                p_pad_y = (target_h - pnh) // 2
                right_box[p_pad_y : p_pad_y + pnh, p_pad_x : p_pad_x + pnw] = presized

                prompt = f"Pending: '{pending_item.label}'  (SPACE=capture, s=skip, q=quit)"
                cv2.putText(frame, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if result.multi_hand_landmarks:
                    cv2.putText(frame, "Hand detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

                composite_cap = np.concatenate([frame, right_box], axis=1)
                cv2.imshow("Capture Pending References", composite_cap)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[info] Capture mode exited early by user.")
                    break
                if key == ord("s"):
                    print(f"[info] Skipped '{pending_item.label}'.")
                    pending.pop(0)
                    continue
                if key == ord(" "):  # spacebar to capture
                    if result.multi_hand_landmarks:
                        norm_current = normalize_landmarks(result.multi_hand_landmarks[0].landmark)
                        references.append(
                            ReferenceGesture(
                                label=pending_item.label,
                                landmarks=norm_current,
                                image=pending_item.image,
                            )
                        )
                        pending.pop(0)
                        print(f"[info] Captured live landmarks for '{pending_item.label}'. Remaining: {len(pending)}")
                    else:
                        print("[warn] No hand detected on capture attempt; try again.")
        finally:
            cv2.destroyWindow("Capture Pending References")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror for user-friendly view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w, _ = frame.shape
            label_text = "No gesture detected."
            matched_img = None

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    norm_current = normalize_landmarks(hand_landmarks.landmark)

                    # If we have pending references (images where no hand was detected), capture the current hand
                    # landmarks for the first pending item to create a usable reference.
                    if pending:
                        pending_item = pending.pop(0)
                        references.append(
                            ReferenceGesture(
                                label=pending_item.label,
                                landmarks=norm_current,
                                image=pending_item.image,
                            )
                        )
                        print(f"[info] Captured live landmarks for '{pending_item.label}' (was skipped).")

                    label, sim, matched_img = match_gesture(
                        norm_current, references, threshold=2.0
                    )
                    if label:
                        label_text = f"{label} (sim {sim:.2f})"
                        last_label = label
                        last_image = matched_img
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                    break  # use first detected hand

            # If nothing detected this frame but we have a previous match, keep showing it.
            if matched_img is None and last_image is not None and last_label is not None:
                matched_img = last_image
                label_text = f"{last_label} (last match)"

            cv2.putText(
                frame,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Build a single composite view with two boxes: webcam (left) and matched image (right).
            target_h = frame.shape[0]
            target_w = frame.shape[1]
            # Fixed-ratio box on the right (4:3 based on webcam height), so size stays constant across images.
            box_w = max(200, int(target_h * 0.75))

            # Build right box with aspect-preserving fit (letterbox, no cropping).
            box = np.zeros((target_h, box_w, 3), dtype=np.uint8)
            if matched_img is not None:
                mh, mw = matched_img.shape[:2]
                scale = min(box_w / max(1, mw), target_h / max(1, mh))
                new_w, new_h = max(1, int(mw * scale)), max(1, int(mh * scale))
                resized = cv2.resize(matched_img, (new_w, new_h))

                pad_x = (box_w - new_w) // 2
                pad_y = (target_h - new_h) // 2

                # If padding is negative (shouldn't happen with min scaling), crop safely.
                if pad_x < 0 or pad_y < 0:
                    x_start = max(0, -pad_x)
                    y_start = max(0, -pad_y)
                    resized = resized[y_start : y_start + min(target_h, resized.shape[0]),
                                      x_start : x_start + min(box_w, resized.shape[1])]
                    new_h, new_w = resized.shape[:2]
                    pad_x = max(0, (box_w - new_w) // 2)
                    pad_y = max(0, (target_h - new_h) // 2)

                box[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            else:
                cv2.putText(
                    box,
                    "No image",
                    (20, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            composite = np.concatenate([frame, box], axis=1)
            cv2.imshow("Gesture Viewer", composite)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

