"""
Auto-transform portrait image to conform to some ideal

Instantiate model with:
model = PortraitAutoTransform(<TRAINING IMAGE>)

Fit image with:
fitted_image = model.fir(<INPUT_IMAGE>)

"""

import cv2
import dlib
import warnings
import numpy as np
from typing import List, Optional, Tuple, Union
from shapely.geometry import LineString, Polygon

FACIAL_LANDMARK_MODEL_PATH = "/content/drive/MyDrive/transform_conform/shape_predictor_68_face_landmarks.dat"
OVERLAY_OFFSET_X = 2
OVERLAY_OFFSET_Y = 2
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_THICKNESS = 1
OVERLAY_SCALE = 0.25
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_COLOR_2 = (255, 0, 0)
LANDMARKS_PICKED = [
    # 28, 29, 30, 31,  # nose line
    37, 38, 39, 40, 41, 42,  # left eye
    43, 44, 45, 46, 47, 48,  # right eye
    49, 50, 52, 54, 55, 56, 58, 60, # mouth
]
CANVAS_SCALING_WARN_THRESHOLD = 1.1


def make_transform_matrix(
        scale_factor: float,
        ccw_rotate_rads: float,
        x_translate: float,
        y_translate: float
):
    """
    Image image transform matrix
    :param scale_factor: scale factor (centered on (0, 0) i.e. top left)
    :param ccw_rotate_rads: amount to rotate counter-clockwise in radians (centered on (0, 0) i.e. top left)
    :param x_translate: shifts image to the right
    :param y_translate: shifts image upwards
    :return:
    """
    rotate_scale_matrix = np.vstack((
        cv2.getRotationMatrix2D(
            center=(0, 0),
            angle=180 * ccw_rotate_rads / np.pi,
            scale=scale_factor
        ),
        np.array([0, 0, 1])
    ))
    translate_matrix = np.array([
        [1, 0, x_translate],  # again, cv origin is top left, so y
        [0, 1, y_translate],
        [0, 0, 1]
    ])

    return np.dot(translate_matrix, rotate_scale_matrix)


def scale_rotate_translate_img(
        img: np.array,
        m: np.array,
        crop: Optional[Tuple[int, int]] = None
):
    """
    Applies scaling, rotation, and translation to an image in this strict order
    :param img: input image
    :param m: transform matrix
    :param crop: crop size from top left, use original size if None
    :return: transformed image
    """
    (height, width) = img.shape[:2]
    transformed_image = cv2.warpPerspective(
        img, m, (width, height) if crop is None else crop
    )

    return transformed_image


def scale_rotate_translate_coords(
        coords: np.array,
        m: np.array
):
    """
    Applies scaling, rotation, and translation to coordinates
    :param coords: n x 2 array in order of (x, y)
    :param m: transform matrix
    :return:
    """
    return np.dot(
        m, np.vstack((coords.transpose(), np.ones(len(coords))))
    ).transpose()[:, :2]


def overlay_landmarks(
        img: np.array,
        landmarks: np.array,
        labels: List = LANDMARKS_PICKED,
        color: Tuple = OVERLAY_COLOR):
    """
    :param img: image
    :param landmarks: n x 2 array of landmarks
    :param labels: if specified, a list of corresponding labels to the landmarks
    :param color: if specified, BGR tuple of colour
    :return: image with landmarks overlaid
    """
    overlay_img = img.copy()
    for n, [x, y] in enumerate(landmarks):
        overlay_img = cv2.circle(
            img=overlay_img,
            center=(int(x), int(y)),
            radius=1,
            color=color,
            thickness=-1
        )
        overlay_img = cv2.putText(
            img=overlay_img,
            text=str(labels[n]) if labels is not None else str(n),
            org=(int(x) + OVERLAY_OFFSET_X, int(y) + OVERLAY_OFFSET_Y),
            fontFace=OVERLAY_FONT,
            fontScale=OVERLAY_SCALE,
            color=color,
            thickness=OVERLAY_FONT_THICKNESS
        )

    return overlay_img


class PortraitAutoTransform:
    def __init__(self, train_image: Union[str, np.array]):
        """
        :param train_image: path of ideal portrait or as a image array
        """
        if isinstance(train_image, str):
            self.train_image = cv2.imread(train_image)
        else:
            self.train_image = train_image

        # init models
        self.face_detector = dlib.get_frontal_face_detector()
        self.facial_landmark_predictor = dlib.shape_predictor(
            FACIAL_LANDMARK_MODEL_PATH
        )

        # get landmark location as n x 2 array
        self.train_landmarks = self.predict_landmarks(self.train_image)

        # image dimensions
        (self.train_height, self.train_width) = self.train_image.shape[:2]

    @property
    def train_image_overlay(self):
        return overlay_landmarks(
            self.train_image, self.train_landmarks)

    def predict_landmarks(self, img: np.array):
        """
        :param img: input image
        :return: landmarks as n x 2 array
        """
        predicted_landmarks = self.facial_landmark_predictor(
            image=img,
            box=self.face_detector(img)[0]
        )
        predicted_landmarks_arr = np.array([
            [predicted_landmarks.part(n).x, predicted_landmarks.part(n).y]
            for n in range(predicted_landmarks.num_parts)
        ])

        return predicted_landmarks_arr[np.array(LANDMARKS_PICKED) - 1, :]

    def compute_fit_params(self, input_landmarks):
        """
        Computes the transform in image coordinates (i.e. 0, 0 is top-left)
        This implementation is different from the analytical solution in the accompanying .pdf because
        of this change in directionality (i.e. theta is actually clockwise instead of anti-clockwise in
        regular cartesian coordinates

        :param input_landmarks:  n x 2 array of landmarks to be transformed
        :return: (scale_factor, ccw_rotate_rads, x_translate, y_translate)
        """
        assert len(input_landmarks) == len(self.train_landmarks)

        N = len(input_landmarks)
        [x_input_sum, y_input_sum] = input_landmarks.sum(axis=0)
        [x_train_sum, y_train_sum] = self.train_landmarks.sum(axis=0)

        p = - x_train_sum * x_input_sum / N - y_train_sum * y_input_sum / N + (input_landmarks * self.train_landmarks).sum()
        q = - x_train_sum * y_input_sum / N + y_train_sum * x_input_sum / N + (input_landmarks[:, 1] * self.train_landmarks[:, 0]).sum() - (input_landmarks[:, 0] * self.train_landmarks[:, 1]).sum()
        cw_rotate_rads = np.arctan(-q / p)

        alpha = (input_landmarks * input_landmarks).sum() - (input_landmarks.sum(axis = 0) * input_landmarks).sum() / N
        beta = ((self.train_landmarks.sum(axis=0) / N - self.train_landmarks) * input_landmarks).sum()
        gamma = -((x_train_sum / N - self.train_landmarks[:, 0]) * input_landmarks[:, 1]).sum() + ((y_train_sum / N - self.train_landmarks[:, 1]) * input_landmarks[:, 0]).sum()
        scale_factor = -(beta * np.cos(cw_rotate_rads) + gamma * np.sin(cw_rotate_rads)) / alpha

        x_translate = -scale_factor * x_input_sum / N * np.cos(cw_rotate_rads) + scale_factor * y_input_sum / N * np.sin(cw_rotate_rads) + x_train_sum / N
        y_translate = -scale_factor * x_input_sum / N * np.sin(cw_rotate_rads) - scale_factor * y_input_sum / N * np.cos(cw_rotate_rads) + y_train_sum / N

        return scale_factor, -cw_rotate_rads, x_translate, y_translate

    def compute_canvas_scaling(
            self,
            canvas_corners: np.array
    ) -> float:
        """
        Let center be (0.5 * train_width, 0.5 * train_height)
        Given the polygon vertices of some canvas, compute how much it needs to be enlarged from the center
        so we can crop out a centered rectangle of (train_width, train_height)
        :param canvas_corners:
        :return:
        """
        canvas_box = Polygon(zip(canvas_corners[:, 0], -canvas_corners[:, 1]))

        train_corners = np.array([
            (0, 0), (self.train_width, 0), (self.train_width, self.train_height), (0, self.train_height)])
        diagonals_from_center = [
            LineString([
                (self.train_width / 2, -self.train_height / 2), (x, -y)
            ]) for (x, y) in train_corners
        ]

        min_diagonal_intersect_len = min([
            canvas_box.intersection(line).length
            for line in diagonals_from_center
        ])
        canvas_scale_factor = np.linalg.norm([self.train_width, self.train_height]) / min_diagonal_intersect_len / 2

        return canvas_scale_factor

    def fit(
            self,
            img: np.array,
            debug_mode: bool = False,
            crop_inside_input_canvas: bool = False
    ):
        """
        Fits landmarks of input image to the training image

        :param img: input image
        :param debug_mode: if true returns image with training landmarks in green and transformed landmarks in blue
        :crop_inside_input_canvas: if true, crops the image strictly within the input canvas
        :return: fitted image
        """
        input_landmarks = self.predict_landmarks(img)
        transform_matrix_0 = make_transform_matrix(*self.compute_fit_params(input_landmarks))

        # This section deals with the case when the ideal cropped image crops out of the the input canvas
        (height, width) = img.shape[:2]
        transformed_corners = scale_rotate_translate_coords(
            np.array([(0, 0), (width, 0), (width, height), (0, height)]),
            transform_matrix_0
        )
        canvas_scale_factor = self.compute_canvas_scaling(transformed_corners) if crop_inside_input_canvas else 1.0
        transform_matrix_1 = make_transform_matrix(
            canvas_scale_factor,
            0,
            (1 - canvas_scale_factor) * self.train_width / 2,
            (1 - canvas_scale_factor) * self.train_height / 2
        )

        transform_matrix_combined = np.dot(transform_matrix_1, transform_matrix_0)
        transformed_image = scale_rotate_translate_img(
            img=img, m=transform_matrix_combined, crop=(self.train_width, self.train_height)
        )

        if canvas_scale_factor > CANVAS_SCALING_WARN_THRESHOLD:
            warnings.warn("Your input image could be too closely cropped, try taking a photo from further away")

        if not debug_mode:
            return transformed_image

        else:
            overlaid1 = overlay_landmarks(
                transformed_image, self.train_landmarks
            )
            overlaid2 = overlay_landmarks(
                overlaid1,
                scale_rotate_translate_coords(
                    input_landmarks, transform_matrix_combined
                ),
                color=OVERLAY_COLOR_2
            )

        return overlaid2

