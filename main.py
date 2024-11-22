import cv2
import time
from face_rec import load_known_faces, recognize_face
from face_detect import detect_faces, draw_bounding_box
from SilentFaceAntiSpoofing.test import test_from_image

# Load known faces
known_faces, known_names = load_known_faces()

def recognize_faces():
    """Real-time face recognition with integrated SilentFace anti-spoofing."""
    video_capture = cv2.VideoCapture(0)
    last_recognition_time = 0  # Timestamp for recognition interval
    base_margin = 20  # Margin for bounding box adjustments
    recognized_faces = []  # Cache recognized faces
    model_dir = "SilentFaceAntiSpoofing/resources/anti_spoof_models"  # Path to model directory

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces continuously
        faces = detect_faces(frame)

        # Perform recognition every 2 seconds
        current_time = time.time()
        if current_time - last_recognition_time >= 2:
            last_recognition_time = current_time
            recognized_faces.clear()  # Clear previous recognition results

            for (x, y, width, height) in faces:
                face_region = frame[y:y + height, x:x + width]

                # Perform face recognition first
                name, _ = recognize_face(face_region, known_faces, known_names)

                if name != "Unknown":
                    # If recognized, check for spoofing
                    try:
                        spoof_result = test_from_image(face_region, model_dir)
                        if spoof_result == "Fake":
                            name = "Spoof Detected"
                    except Exception as e:
                        print(f"Error during anti-spoofing detection: {e}")
                        name = "Error in Spoof Check"

                recognized_faces.append((name, x, y, width, height))

        # Draw bounding boxes and names
        for (x, y, width, height) in faces:
            name = "Unknown"
            for stored_name, stored_x, stored_y, stored_width, stored_height in recognized_faces:
                if abs(x - stored_x) < base_margin and abs(y - stored_y) < base_margin:
                    name = stored_name
                    break
            draw_bounding_box(frame, x, y, width, height, name)

        # Display the video feed with bounding boxes
        cv2.imshow('Face Recognition with Anti-Spoofing', cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()



























