import tkinter as tk
import cv2
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from feature_detector import FeatureDetector
from face_compositor import FaceCompositor

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

        # Define Haar Cascade file paths
        eye_cascade_path = 'haarcascade/haarcascade_eye.xml'
        nose_cascade_path = 'haarcascade/haarcascade_mcs_nose.xml'
        mouth_cascade_path = 'haarcascade/haarcascade_mcs_mouth.xml'

        self.detector = FeatureDetector(eye_cascade_path, nose_cascade_path, mouth_cascade_path)
        self.compositor = FaceCompositor(eye_cascade_path, nose_cascade_path, mouth_cascade_path)

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

    def upload_first_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.detector.detect_and_save_features(file_path)
            messagebox.showinfo("Info", "특징이 검출되어 성공적으로 저장되었습니다.")

    def upload_second_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            original_image, composite_image = self.compositor.composite_faces(file_path)
            if original_image is not None and composite_image is not None:
                self.display_result(original_image, composite_image)
