import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import csv

class YOLODetector:
    def __init__(self, path_of_model):
        self.model = YOLO(path_of_model)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def preprocess(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (640, 640))
        image = image / 255.0
        return image

    def predict_frame(self, frame):
        original_image = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed_image = self.preprocess(image)

        # Convert back to PIL Image for YOLO model compatibility
        preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
        image = Image.fromarray(preprocessed_image)

        # Predict and process results using the YOLO model
        results = self.model(image)
        if results is None or len(results[0].boxes) == 0:
            return original_image, []

        detections = results[0].boxes.xyxy
        class_ids = results[0].boxes.cls
        confidences = results[0].boxes.conf

        plate_numbers = []

        # Check for "WithoutHelmet" class before processing license plates
        without_helmet_detected = any(cls == 2 for cls in class_ids)  # Assuming 2 is the class ID for "WithoutHelmet"

        for idx, (box, class_id, confidence) in enumerate(zip(detections, class_ids, confidences)):
            plate_number = None  # Initialize plate_number for each iteration
            if class_id == 0 and without_helmet_detected:  # Class ID for license plate and "WithoutHelmet" condition
                # Convert coordinates
                scale_x = original_image.shape[1] / 640
                scale_y = original_image.shape[0] / 640
                x1, y1, x2, y2 = (box.cpu().numpy() * [scale_x, scale_y, scale_x, scale_y]).round().astype(int)

                # Ensure bounding box is within image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(original_image.shape[1], x2), min(original_image.shape[0], y2)

                # Crop the plate region
                plate_region = original_image[y1:y2, x1:x2]
                if plate_region is None or plate_region.size == 0:
                    continue

                # Preprocess and apply OCR using PaddleOCR
                try:
                    ocr_results = self.ocr.ocr(plate_region, cls=True)
                    text = ''.join([line[1][0] for line in ocr_results[0]])
                    plate_number = text.strip()
                    if plate_number:
                        plate_numbers.append(plate_number)
                        print(f"PaddleOCR result: {plate_number}")
                except Exception as e:
                    print(f"Error during OCR processing: {e}")

                # Draw the bounding box and label
                cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_image, plate_number if plate_number else "", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return original_image, plate_numbers

    def process_video(self, video_path, output_csv):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")

        all_plate_numbers = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, plate_numbers = self.predict_frame(frame)
            if plate_numbers:
                all_plate_numbers.update(plate_numbers)

            # Optionally display the frame with annotations
            cv2.imshow('Frame', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Write plate numbers to CSV
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Plate Number'])
            for plate in all_plate_numbers:
                writer.writerow([plate])

        print(f"Saved plate numbers to {output_csv}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and paths
path_of_model = 'yolo-weights/best.pt'
path_of_video = "videos/22.mp4"
PlateNumber = "output/plate_numbers.txt"
model = YOLODetector(path_of_model)

# Process video and save results
model.process_video(path_of_video, PlateNumber)
