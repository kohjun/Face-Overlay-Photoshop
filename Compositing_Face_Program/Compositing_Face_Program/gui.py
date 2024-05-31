import tkinter as tk
import cv2
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from feature_detector import FeatureDetector
from face_compositor import FaceCompositor
from image_adjuster import ImageAdjuster

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
        self.root.title("얼굴 합성 포토샵")
        frame = tk.Frame(root)
        frame.pack(pady=20)

        # Define Haar Cascade file paths
        eye_cascade_path = 'haarcascade/haarcascade_eye.xml'
        nose_cascade_path = 'haarcascade/haarcascade_mcs_nose.xml'
        mouth_cascade_path = 'haarcascade/haarcascade_mcs_mouth.xml'

        self.detector = FeatureDetector(eye_cascade_path, nose_cascade_path, mouth_cascade_path)
        self.compositor = FaceCompositor(eye_cascade_path, nose_cascade_path, mouth_cascade_path)
        self.image_adjuster = None

        btn_first_image = tk.Button(frame, text="Upload First Image", command=self.upload_first_image)
        btn_first_image.grid(row=0, column=0, padx=10)

        btn_second_image = tk.Button(frame, text="Upload Second Image", command=self.upload_second_image)
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

        self.brightness_scale = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, label="밝기", command=self.update_brightness)
        self.brightness_scale.pack()

        self.contrast_scale = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, label="명암", command=self.update_contrast)
        self.contrast_scale.pack()

        self.saturation_scale = tk.Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, label="채도", command=self.update_saturation)
        self.saturation_scale.pack()

        save_button = tk.Button(root, text="저장하기", command=self.save_image)
        save_button.pack(pady=20)

    def upload_first_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.detector.detect_and_save_features(file_path)
            messagebox.showinfo("알림", "이미지가 업로드되었습니다.")

    def upload_second_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            original_image, composite_image = self.compositor.composite_faces(file_path)
            if original_image is not None and composite_image is not None:
                self.image_adjuster = ImageAdjuster(composite_image)
                self.display_result(original_image, composite_image)

    def update_brightness(self, value):
        if self.image_adjuster:
            adjusted_image = self.image_adjuster.adjust_brightness(value)
            self.update_composite_label(adjusted_image)

    def update_contrast(self, value):
        if self.image_adjuster:
            adjusted_image = self.image_adjuster.adjust_contrast(value)
            self.update_composite_label(adjusted_image)

    def update_saturation(self, value):
        if self.image_adjuster:
            adjusted_image = self.image_adjuster.adjust_saturation(value)
            self.update_composite_label(adjusted_image)

    def update_composite_label(self, image):
        b, g, r = cv2.split(image)
        img = cv2.merge((r, g, b))
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        composite_label.config(image=imgtk)
        composite_label.image = imgtk

    def save_image(self):
        if self.image_adjuster:
            output_path = 'images/outputs'
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, 'adjusted_image.jpg')
            cv2.imwrite(save_path, self.image_adjuster.adjusted_image)
            messagebox.showinfo("알림", "이미지가 저장되었습니다.")
