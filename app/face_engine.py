from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


class FaceValidationError(ValueError):
    pass


@dataclass
class ExtractedFace:
    embedding: np.ndarray
    quality: float


class FaceEngine:
    def __init__(self, model_dir: str):
        directory = Path(model_dir)
        detector_path = directory / "face_detection_yunet_2023mar.onnx"
        recognizer_path = directory / "face_recognition_sface_2021dec.onnx"
        if not detector_path.exists() or not recognizer_path.exists():
            raise RuntimeError("Face models are missing. Run scripts/download-models.ps1 before starting the service.")
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.72, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def extract(self, raw: bytes) -> ExtractedFace:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FaceValidationError("The captured image could not be decoded.")
        height, width = image.shape[:2]
        if min(height, width) < 240:
            raise FaceValidationError("Camera resolution is too low. Use at least 240px on the shorter side.")
        max_side = max(width, height)
        scale = 640.0 / max_side if max_side > 640 else 1.0
        detect_img = cv2.resize(image, (int(round(width * scale)), int(round(height * scale)))) if scale < 1.0 else image
        det_h, det_w = detect_img.shape[:2]
        self.detector.setInputSize((det_w, det_h))
        _, faces = self.detector.detect(detect_img)
        if faces is None or len(faces) == 0:
            raise FaceValidationError("No face found. Centre your face in good light and retry.")
        if len(faces) != 1:
            raise FaceValidationError("Exactly one face must be visible.")
        face = faces[0].copy()
        if scale < 1.0:
            face[:14] /= scale
        x, y, fw, fh = face[:4]
        confidence = float(face[-1])
        if min(fw, fh) < 80:
            raise FaceValidationError("Move closer to the camera.")
        crop = image[max(0, int(y)):min(height, int(y + fh)), max(0, int(x)):min(width, int(x + fw))]
        if crop.size == 0:
            raise FaceValidationError("No face found. Centre your face in good light and retry.")
        gray = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 18:
            raise FaceValidationError("The image is blurry. Hold still and retry.")
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
        feature /= np.linalg.norm(feature)
        return ExtractedFace(feature, min(1.0, sharpness / 250.0) * confidence)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
