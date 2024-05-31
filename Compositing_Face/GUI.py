import cv2
import os
import random
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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
            messagebox.showerror("Error", "업로드에 실패하였습니다.확인 후 다시 시도해보기 바랍니다.")
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

class FaceCompositor:
    def __init__(self, eye_cascade_path, nose_cascade_path, mouth_cascade_path):
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        self.nose_cascade = cv2.CascadeClassifier(nose_cascade_path)
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

    def composite_faces(self, image2_path):
        image = cv2.imread(image2_path)
        if image is None:
            messagebox.showerror("Error", "업로드에 실패하였습니다. 확인 후 다시 시도해보기 바랍니다.")
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
            messagebox.showerror("Error", "인식에 실패하였습니다.")
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

class GUI:
    @staticmethod
    def display_result(original_image, composite_image):
        b1, g1, r1 = cv2.split(original_image)
        img1 = cv2.merge((r1, g1, b1))
        im1 = Image.fromarray(img1)
        imgtk1 = ImageTk.PhotoImage(image=im1)
        original_label.config(image=imgtk1)
        original_label.image = imgtk1

        b2, g2, r2 = cv2.split(composite_image)
        img2 = cv2.merge((r2, g2, b2))
        im2 = Image.fromarray(img2)
        imgtk2 = ImageTk.PhotoImage(image=im2)
        composite_label.config(image=imgtk2)
        composite_label.image = imgtk2

    def __init__(self, root):
        self.root = root
        self.root.title("얼굴 합성")
        frame = tk.Frame(root)
        frame.pack(pady=20)

        self.detector = FeatureDetector(eye_cascade_path, nose_cascade_path, mouth_cascade_path)
        self.compositor = FaceCompositor(eye_cascade_path, nose_cascade_path, mouth_cascade_path)

        btn_first_image = tk.Button(frame, text="저장할 첫번째 얼굴", command=self.upload_first_image)
        btn_first_image.grid(row=0, column=0, padx=10)

        btn_second_image = tk.Button(frame, text="합성할 두번째 얼굴", command=self.upload_second_image)
        btn_second_image.grid(row=0, column=1, padx=10)

        global original_label
        original_label = tk.Label(root)
        original_label.pack(side=tk.LEFT, padx=10)

        global composite_label
        composite_label = tk.Label(root)
        composite_label.pack(side=tk.RIGHT, padx=10)

        original_text = tk.Label(root, text="전")
        original_text.pack(side=tk.LEFT)

        composite_text = tk.Label(root, text="후")
        composite_text.pack(side=tk.RIGHT)

    def upload_first_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.detector.detect_and_save_features(file_path)
            messagebox.showinfo("Info", "특징이 감지되었고 성공적으로 저장되었습니다.")

    def upload_second_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            original_image, composite_image = self.compositor.composite_faces(file_path)
            if original_image is not None and composite_image is not None:
                self.display_result(original_image, composite_image)

def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Haar Cascade 파일 경로
    eye_cascade_path = 'haarcascade/haarcascade_eye.xml'
    nose_cascade_path = 'haarcascade/haarcascade_mcs_nose.xml'
    mouth_cascade_path = 'haarcascade/haarcascade_mcs_mouth.xml'
    
    main()
