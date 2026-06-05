# 학습된 TensorFlow Lite 모델을 사용한 실시간 웹캠 수어 인식 파일
# 인식 결과의 화면 텍스트 표시와 TTS 음성 출력

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
import mediapipe as mp

from config import (
    DEFAULT_MAX_FRAMES,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    CONFIDENCE_THRESHOLD,
    STABLE_COUNT,
)

from extract_landmarks import extract_frame_landmarks, resample_or_pad


def load_labels(label_path):
    # labels.txt 파일의 라벨 이름 읽기

    label_path = Path(label_path)

    labels = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return labels


def load_scaler(scaler_path):
    # 학습 때 저장한 평균과 표준편차 불러오기

    data = np.load(scaler_path)

    mean = data["mean"].astype(np.float32)
    scale = data["scale"].astype(np.float32)

    # 0으로 나누는 오류 방지
    scale[scale == 0] = 1.0

    return mean, scale


def preprocess_sequence(sequence, max_frames, mean, scale):
    # 실시간 수집 프레임의 모델 입력 형태 변환

    # 프레임 수를 학습 때와 동일하게 보정
    sequence = resample_or_pad(
        np.asarray(sequence, dtype=np.float32),
        max_frames=max_frames
    )

    # 2차원 데이터를 1차원 입력 벡터로 변환
    x = sequence.reshape(1, -1)

    # 학습 때 사용한 평균과 표준편차 기준 표준화
    x = (x - mean) / scale

    return x.astype(np.float32)


class TFLiteClassifier:
    # TensorFlow Lite 모델 실행 클래스

    def __init__(self, model_path):
        # TFLite 모델 불러오기
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))

        # 모델 실행 텐서 준비
        self.interpreter.allocate_tensors()

        # 입력과 출력 정보 저장
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, x):
        # 입력 데이터 모델 전달
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            x
        )

        # 모델 실행
        self.interpreter.invoke()

        # 모델 출력값 반환
        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        return output[0]


def speak(tts_engine, text):
    # 인식 단어 음성 출력

    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    # 명령어 옵션 처리 객체
    parser = argparse.ArgumentParser()

    # 학습된 TFLite 모델 경로
    parser.add_argument("--model", default="models/ksl_model.tflite")

    # 라벨 파일 경로
    parser.add_argument("--labels", default="models/labels.txt")

    # 표준화 정보 파일 경로
    parser.add_argument("--scaler", default="models/scaler.npz")

    # 사용할 프레임 수
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)

    # 사용할 웹캠 번호
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    # 라벨 목록 불러오기
    labels = load_labels(args.labels)

    # 표준화 정보 불러오기
    mean, scale = load_scaler(args.scaler)

    # TFLite 모델 불러오기
    classifier = TFLiteClassifier(args.model)

    # TTS 엔진 준비
    tts_engine = pyttsx3.init()

    # 음성 속도 설정
    tts_engine.setProperty("rate", 165)

    # 웹캠 열기
    cap = cv2.VideoCapture(args.camera)

    # 웹캠 열기 실패 예외 처리
    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    # 최근 프레임 랜드마크 저장 리스트
    sequence = []

    # 이전 인식 라벨
    last_label = None

    # 같은 라벨 연속 인식 횟수
    stable_counter = 0

    # 마지막 음성 출력 라벨
    spoken_label = None

    # 마지막 음성 출력 시간
    last_speak_time = 0

    # MediaPipe Hands와 그리기 도구
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    print("프로그램 실행 중")
    print("종료하려면 q를 누르세요.")

    # MediaPipe Hands 객체 생성
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            # 웹캠 프레임 읽기
            success, frame = cap.read()

            # 프레임 읽기 실패 시 종료
            if not success:
                break

            # 사용자 확인용 좌우 반전
            frame = cv2.flip(frame, 1)

            # 손 관절 표시용 RGB 변환
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 손 감지
            draw_result = hands.process(rgb)

            # 감지된 손 관절 화면 표시
            if draw_result.multi_hand_landmarks:
                for hand_landmarks in draw_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # 현재 프레임의 손 랜드마크 특징 추출
            feature = extract_frame_landmarks(frame, hands)

            # 최근 프레임 리스트 추가
            sequence.append(feature)

            # 오래된 프레임 삭제
            if len(sequence) > args.max_frames:
                sequence = sequence[-args.max_frames:]

            # 기본 출력 문구
            predicted_text = "인식 대기 중"

            # 예측 확률 초기값
            confidence = 0.0

            # 일정 수 이상의 프레임 수집 후 예측 시작
            if len(sequence) >= max(5, args.max_frames // 3):
                # 모델 입력 형태 전처리
                x = preprocess_sequence(
                    sequence,
                    args.max_frames,
                    mean,
                    scale
                )

                # 모델 예측 수행
                probabilities = classifier.predict(x)

                # 가장 높은 확률의 라벨 번호
                predicted_index = int(np.argmax(probabilities))

                # 가장 높은 예측 확률
                confidence = float(probabilities[predicted_index])

                # 라벨 번호를 실제 단어로 변환
                predicted_label = labels[predicted_index]

                # 기준 확률 이상일 때의 인식 처리
                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_text = predicted_label

                    # 같은 라벨 연속 인식 횟수 계산
                    if predicted_label == last_label:
                        stable_counter += 1
                    else:
                        stable_counter = 1
                        last_label = predicted_label

                    current_time = time.time()

                    # 안정적 인식 후 음성 출력 조건
                    if (
                        stable_counter >= STABLE_COUNT
                        and predicted_label != spoken_label
                        and current_time - last_speak_time > 2.0
                    ):
                        spoken_label = predicted_label
                        last_speak_time = current_time

                        speak(tts_engine, predicted_label)

                else:
                    # 기준 확률 미만일 때의 인식 보류
                    predicted_text = "확신 부족"
                    stable_counter = 0
                    last_label = None

            # 텍스트 가독성 확보용 상단 검은 배경
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 80),
                (0, 0, 0),
                -1
            )

            # 예측 결과와 확률 화면 표시
            cv2.putText(
                frame,
                f"{predicted_text} / confidence: {confidence:.2f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # 최종 화면 출력
            cv2.imshow("KSL Sign Translator", frame)

            # 종료 키 입력 확인
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    # 웹캠과 창 종료
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
