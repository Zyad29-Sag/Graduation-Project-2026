import cv2
import mediapipe as mp
import os
import numpy as np

# Mediapipe setup
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = {}  # name -> feature (histogram)

    def load_known_faces(self, folder):
        print(f"📂 Loading known faces from: {folder}")

        try:
            files = os.listdir(folder)
            print("📑 Files found:", files)
        except FileNotFoundError:
            print("❌ Folder not found.")
            return

        for file in files:
            print(f"\n🔍 Checking file: {file}")
            path = os.path.join(folder, file)

            # Check valid extension
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                print("⚠ Skipped (not an image)")
                continue

            # Load image
            img = cv2.imread(path)
            if img is None:
                print(f"❌ Could not read {file}")
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)

            if results.detections:
                for i, det in enumerate(results.detections):
                    box = det.location_data.relative_bounding_box
                    h, w, _ = img.shape
                    x1, y1 = int(box.xmin * w), int(box.ymin * h)
                    x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)

                    face_crop = img[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        print(f"⚠ Face crop failed for {file}")
                        continue

                    # Convert face crop to grayscale histogram (simple feature)
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()

                    name = os.path.splitext(file)[0]
                    if len(results.detections) > 1:
                        name = f"{name}_{i}"  # group photo multiple faces

                    self.known_faces[name] = hist
                    print(f"✅ Loaded {name} from {file}")

                    # Show detected face preview
                    preview = img.copy()
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Preview - press any key", preview)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print(f"⚠ No face found in {file}")

        print(f"\n✅ Total known faces loaded: {len(self.known_faces)}")

    def recognize_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        recognized = []
        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(box.xmin * w), int(box.ymin * h)
                x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                name = "Unknown"
                best_score = 0.5  # threshold
                for known_name, known_hist in self.known_faces.items():
                    score = cv2.compareHist(known_hist.astype("float32"), hist.astype("float32"), cv2.HISTCMP_CORREL)
                    if score > best_score:
                        best_score = score
                        name = known_name

                # Draw box + name
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                recognized.append(name)
        return recognized

def main():
    system = FaceRecognitionSystem()
    system.load_known_faces(r"C:\Users\Zyad\Desktop\Graduation\images_folder")

    cap = cv2.VideoCapture(0)
    print("🎥 Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        recognized = system.recognize_face(frame)
        cv2.imshow("Face Recognition (Mediapipe)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
