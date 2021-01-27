import numpy as np
import dlib
import cv2
from pkg_resources import resource_filename


class Centralize:
    """
    Centralize an image using simple heuristic of centralizing nose.
    Initialize on an image.
    """
    def __init__(self, img, landmarks_model=None, nose_start=28, nose_end=30):
        self.img = img
        self.h, self.w, _ = self.img.shape
        self.center_y = self.h//2
        self.center_x = self.w//2

        self.nose_start = nose_start
        self.nose_end = nose_end

        self.detector = dlib.get_frontal_face_detector()

        included_model_path = resource_filename('centralize', 'resources/shape_predictor_68_face_landmarks.dat')
        model_path = landmarks_model if landmarks_model is not None else included_model_path
        self.landmark_predictor = dlib.shape_predictor(model_path)

    def get_nose(self):
        """
        gets mean x and y coords of nose points
        """
        face = self.detector(self.img)[0]
        landmarks = self.landmark_predictor(image=self.img, box=face)

        nose_coords = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(self.nose_start, self.nose_end+1)]

        return np.mean([coord[0] for coord in nose_coords]), np.mean([coord[1] for coord in nose_coords])

    def centralize(self):
        nose_x, nose_y = self.get_nose()

        x_shift = self.center_x - nose_x
        y_shift = self.center_y - nose_y

        m = np.float32([[1, 0, x_shift], [0, 1, y_shift]])

        return cv2.warpAffine(self.img, m, (self.w, self.h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=[255, 255, 255])
