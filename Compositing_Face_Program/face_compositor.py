import cv2
import os
import random
import numpy as np
from tkinter import messagebox

class FaceCompositor:
    def __init__(self, eye_cascade_path, nose_cascade_path, mouth_cascade_path):
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        self.nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

    def composite_faces(self, image2_path):
        image = cv2.imread(image2_path)
        if image is None:
            messagebox.showerror("Error", "이미지 로드에 실패하였습니다.")
            return None, None

        original_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = self.blur_features(image, gray)
        self.overlay_features(image)
        return original_image, image

    def blur_features(self, image, gray):
        def blur_feature(image, features, ksize=(23, 23)):
            for (x, y, w, h) in features:
                roi = image[y:y + h, x:x + w]
                blurred_roi = cv2.GaussianBlur(roi, ksize, 0)
                image[y:y + h, x:x + w] = blurred_roi
            return image

        image = blur_feature(image, self.eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(25, 25)))
        image = blur_feature(image, self.nose_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(25, 35)))
        image = blur_feature(image, self.mouth_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30)))
        return image

    def overlay_features(self, image):
        left_eye_file = random.choice(os.listdir('images/eyes/left'))
        right_eye_file = random.choice(os.listdir('images/eyes/right'))
        nose_file = random.choice(os.listdir('images/noses'))
        mouth_file = random.choice(os.listdir('images/mouths'))

        left_eye_img = cv2.imread(os.path.join('images/eyes/left', left_eye_file), cv2.IMREAD_UNCHANGED)
        right_eye_img = cv2.imread(os.path.join('images/eyes/right', right_eye_file), cv2.IMREAD_UNCHANGED)
        nose_img = cv2.imread(os.path.join('images/noses', nose_file), cv2.IMREAD_UNCHANGED)
        mouth_img = cv2.imread(os.path.join('images/mouths', mouth_file), cv2.IMREAD_UNCHANGED)

        def color_match(source, target):
            source_mean, source_std = cv2.meanStdDev(source)
            target_mean, target_std = cv2.meanStdDev(target)
            source_mean = source_mean.reshape(1, 1, 3)
            source_std = source_std.reshape(1, 1, 3)
            target_mean = target_mean.reshape(1, 1, 3)
            target_std = target_std.reshape(1, 1, 3)
            matched = (source - source_mean) / source_std * target_std + target_mean
            matched = np.clip(matched, 0, 255).astype(np.uint8)
            return matched

        def resize_feature(image, positions):
            resized_images = []
            for (x, y, w, h) in positions:
                resized = cv2.resize(image, (w, h))
                resized_images.append((resized, (x, y, w, h)))
            return resized_images

        def blend_images(background, overlay, x, y, alpha=0.7):
            ol_h, ol_w = overlay.shape[:2]
            for c in range(0, 3):
                background[y:y + ol_h, x:x + ol_w, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[y:y + ol_h, x:x + ol_w, c]).astype(np.uint8)
            return background

        eye_positions = self.eye_cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=6, minSize=(25, 25))
        if len(eye_positions) >= 2:
            if eye_positions[0][0] < eye_positions[1][0]:
                left_eye_resized = resize_feature(left_eye_img, [eye_positions[1]])[0]
                right_eye_resized = resize_feature(right_eye_img, [eye_positions[0]])[0]
            else:
                left_eye_resized = resize_feature(left_eye_img, [eye_positions[0]])[0]
                right_eye_resized = resize_feature(right_eye_img, [eye_positions[1]])[0]
        else:
            messagebox.showerror("Error", "눈이 인식에 실패하였습니다.")
            return

        nose_positions = self.nose_cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=6, minSize=(25, 35))
        if len(nose_positions) > 0:
            largest_nose_position = max(nose_positions, key=lambda rect: rect[2] * rect[3])
            nose_resized = resize_feature(nose_img, [largest_nose_position])[0]
        else:
            nose_resized = None

        mouth_positions = self.mouth_cascade.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scaleFactor=1.3, minNeighbors=6, minSize=(30, 30))
        if len(mouth_positions) > 0:
            mouth_resized = resize_feature(mouth_img, [mouth_positions[0]])[0]
        else:
            mouth_resized = None

        def adjust_and_blend_feature(image, feature_img, position):
            x, y, w, h = position
            target_region = image[y:y+h, x:x+w]
            matched_feature = color_match(feature_img, target_region)
            return blend_images(image, matched_feature, x, y)

        image = adjust_and_blend_feature(image, left_eye_resized[0], left_eye_resized[1])
        image = adjust_and_blend_feature(image, right_eye_resized[0], right_eye_resized[1])
        if nose_resized is not None:
            image = adjust_and_blend_feature(image, nose_resized[0], largest_nose_position)
        if mouth_resized is not None:
            image = adjust_and_blend_feature(image, mouth_resized[0], mouth_positions[0])

        return image
