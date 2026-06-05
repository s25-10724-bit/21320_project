import cv2
import numpy as np
import mediapipe as mp

from config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    FEATURES_PER_FRAME,
)

mp_hands = mp.solutions.hands


def empty_hand():
    return np.zeros((21, 3), dtype=np.float32)


def landmark_to_array(hand_landmarks):
    coords = []

    for lm in hand_landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords, dtype=np.float32)


def normalize_hand(hand):
    if np.allclose(hand, 0):
        return hand

    wrist = hand[0].copy()
    hand = hand - wrist

    scale = np.linalg.norm(hand[9])

    if scale < 1e-6:
        scale = 1.0

    return hand / scale


def extract_frame_landmarks(image_bgr, hands):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    left = empty_hand()
    right = empty_hand()

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            label = handedness.classification[0].label
            hand_array = landmark_to_array(hand_landmarks)
            hand_array = normalize_hand(hand_array)

            if label == "Left":
                left = hand_array
            else:
                right = hand_array

    frame_features = np.concatenate([
        left.reshape(-1),
        right.reshape(-1)
    ])

    return frame_features.astype(np.float32)


def resample_or_pad(sequence, max_frames):
    if len(sequence) == 0:
        return np.zeros((max_frames, FEATURES_PER_FRAME), dtype=np.float32)

    sequence = np.asarray(sequence, dtype=np.float32)

    if len(sequence) >= max_frames:
        indices = np.linspace(0, len(sequence) - 1, max_frames).astype(int)
        return sequence[indices]

    padded = np.zeros((max_frames, FEATURES_PER_FRAME), dtype=np.float32)
    padded[:len(sequence)] = sequence

    return padded


def extract_video_landmarks(video_path, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            success, frame = cap.read()

            if not success:
                break

            feature = extract_frame_landmarks(frame, hands)
            frames.append(feature)

    cap.release()

    return resample_or_pad(np.array(frames, dtype=np.float32), max_frames)
