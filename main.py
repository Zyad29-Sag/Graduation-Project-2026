import cv2
import face_recognition
import os
import numpy as np


class FaceRecognitionSystem:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []

    def load_known_faces(self, folder):
        print(f"📂 Loading known faces from: {folder}")

        try:
            files = os.listdir(folder)
            print("📑 Files found:", files)
        except FileNotFoundError:
            print("❌ Folder not found.")
            return

        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(folder, file)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)

            if encodings:
                encoding = encodings[0]
                name = os.path.splitext(file)[0]
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠ No face found in {file}")

        print(f"\n✅ Total known faces loaded: {len(self.known_names)}")

    def recognize_face(self, frame):
        # ✅ Resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings on small frame
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        recognized = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_names[first_match_index]

            # ✅ Scale back face locations since we reduced frame size
            top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]

            # Draw box + name on original frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            recognized.append(name)

        return recognized


def main():
    system = FaceRecognitionSystem()
    system.load_known_faces(r"H:\Graduation-Project-2026\images_folder")

    cap = cv2.VideoCapture(0)
    print("🎥 Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        system.recognize_face(frame)  # runs on resized frame internally
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
