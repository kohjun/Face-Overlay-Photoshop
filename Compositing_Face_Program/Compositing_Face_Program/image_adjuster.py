import cv2
import numpy as np

class ImageAdjuster:
    def __init__(self, image):
        self.original_image = image
        self.adjusted_image = image.copy()
    
    def adjust_brightness(self, value):
        value = int(value)
        self.adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=1, beta=value)
        return self.adjusted_image
    
    def adjust_contrast(self, value):
        value = int(value)
        f = 131 * (value + 127) / (127 * (131 - value))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        self.adjusted_image = cv2.addWeighted(self.original_image, alpha_c, self.original_image, 0, gamma_c)
        return self.adjusted_image

    def adjust_saturation(self, value):
        value = int(value)
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * (value / 127.0)
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        self.adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return self.adjusted_image
