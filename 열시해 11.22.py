import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe 설정
mp_face = mp.solutions.face_detection


def get_face_roll_angle(detection, w, h):
    """
    오른쪽 눈~왼쪽 눈을 잇는 선의 기울기로 얼굴 roll angle 계산
    반환: 각도(deg), -180 ~ 180
    """
    keypoints = detection.location_data.relative_keypoints

    # 0: 오른쪽 눈, 1: 왼쪽 눈 (FaceDetection 기본 모델 기준)
    right_eye = keypoints[0]
    left_eye = keypoints[1]

    x1, y1 = right_eye.x * w, right_eye.y * h
    x2, y2 = left_eye.x * w, left_eye.y * h

    dx = x2 - x1
    dy = y2 - y1

    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    return angle_deg, (int(x1), int(y1)), (int(x2), int(y2))


def main():
    st.title("얼굴 기울기(roll) 측정 데모")

    st.write("얼굴이 나온 사진을 업로드하면 눈 기준으로 기울기를 계산해 줄게.")

    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요 (jpg / jpeg / png)",
        type=["jpg", "jpeg", "png"]
    )

    # 아직 파일이 없으면 그냥 안내만 띄우고 종료
    if uploaded_file is None:
        st.info("왼쪽에서 이미지를 선택하면 결과가 여기에 표시됩니다.")
        return

    # ---- 1) 이미지 읽기 ----
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("이미지를 읽지 못했습니다.")
        return

    h, w, _ = img_bgr.shape

    # ---- 2) 얼굴 검출 + 각도 계산 ----
    with mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
    ) as face_detector:

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_detector.process(img_rgb)

        if not results.detections:
            st.error("얼굴이 감지되지 않았습니다. 다른 사진으로 시도해 주세요.")
            st.image(img_bgr, channels="BGR")
            return

        # 가장 첫 번째 얼굴만 사용
        detection = results.detections[0]

        angle_deg, right_eye_pt, left_eye_pt = get_face_roll_angle(
            detection, w, h
        )

    # ---- 3) 결과 그리기 ----
    result_img = img_bgr.copy()

    # 눈을 잇는 선
    cv2.line(result_img, right_eye_pt, left_eye_pt, (0, 255, 0), 2)

    # 각도 텍스트
    text = f"roll: {angle_deg:.1f} deg"
    cv2.putText(
        result_img, text, (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
    )

    # ---- 4) Streamlit에 표시 ----
    st.image(result_img, channels="BGR")
    st.write(f"**얼굴 기울기 (roll angle)**: `{angle_deg:.1f}°`")


if __name__ == "__main__":
    main()

