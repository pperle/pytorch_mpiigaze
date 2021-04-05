from typing import List

import dlib
import numpy as np
import yacs.config
import mediapipe as mp

from gaze_estimation.gaze_estimator.common import Face


class LandmarkEstimator:
    def __init__(self, config: yacs.config.CfgNode):
        self.mode = config.face_detector.mode
        if self.mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(
                config.face_detector.dlib.model)
        elif self.mode == 'MediaPipe':
            self.detect_face_landmarks_mediapipe = mp.solutions.face_mesh.FaceMesh()
        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'dlib':
            return self._detect_faces_dlib(image)
        elif self.mode == 'MediaPipe':
            return self._detect_faces_mediapipe(image)
        else:
            raise ValueError

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        width, height = image.shape[:2][::-1]

        results = self.detect_face_landmarks_mediapipe.process(image[:, :, ::-1])
        detected = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([(pt.x * width, pt.y * height) for pt in face_landmarks.landmark], dtype=np.float)
                bbox = np.array([[landmarks[:, 0].min(), landmarks[:, 1].min()], [landmarks[:, 0].max(), landmarks[:, 1].max()]], dtype=np.float)
                detected.append(Face(bbox, landmarks))
                return detected

        return []