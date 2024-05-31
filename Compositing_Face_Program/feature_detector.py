import cv2
import os
from tkinter import messagebox

class FeatureDetector:
    def __init__(self, eye_cascade_path, nose_cascade_path, mouth_cascade_path):
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        self.nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
        os.makedirs('images/eyes/left', exist_ok=True)
        os.makedirs('images/eyes/right', exist_ok=True)
        os.makedirs('images/noses', exist_ok=True)
        os.makedirs('images/mouths', exist_ok=True)

    def detect_and_save_features(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", "이미지 로드에 실패하였습니다.")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_center = image.shape[1] // 2

        eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(25, 25))
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            eye_img = image[ey:ey + eh, ex:ex + ew]
            if ex + ew / 2 < image_center:
                eye_file_path = os.path.join('images/eyes/right', f'eye_{i}.jpg')  # 사진 기준으로 오른쪽 눈
            else:
                eye_file_path = os.path.join('images/eyes/left', f'eye_{i}.jpg')  # 사진 기준으로 왼쪽 눈
            cv2.imwrite(eye_file_path, eye_img)

        noses = self.nose_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(20, 35))
        if len(noses) > 0:
            largest_nose = max(noses, key=lambda rect: rect[2] * rect[3])
            nx, ny, nw, nh = largest_nose
            nose_img = image[ny:ny + nh, nx:nx + nw]
            nose_file_path = os.path.join('images/noses', 'nose_0.jpg')
            cv2.imwrite(nose_file_path, nose_img)

        mouths = self.mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(30, 30))
        if len(mouths) > 0:
            largest_mouth = max(mouths, key=lambda rect: rect[2] * rect[3])
            mx, my, mw, mh = largest_mouth
            mouth_img = image[my:my + mh, mx:mx + mw]
            mouth_file_path = os.path.join('images/mouths', 'mouth_0.jpg')
            cv2.imwrite(mouth_file_path, mouth_img)
