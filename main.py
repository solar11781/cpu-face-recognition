import cv2
import time
from face_rec import load_known_faces, recognize_face
from face_detect import detect_faces, draw_bounding_box
from anti_spoof import detect_liveness

# Load known faces
known_faces, known_names = load_known_faces()

def recognize_faces():
    """Real-time face recognition with anti-spoofing."""
    video_capture = cv2.VideoCapture(0)
    last_recognition_time = 0  # Timestamp for recognition interval
    base_margin = 20  # Base margin for bounding box adjustments
    recognized_faces = []  # List to store recognized faces

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces continuously for real-time bounding box tracking
        faces = detect_faces(frame)

        # Perform recognition only every 2 seconds
        current_time = time.time()
        if current_time - last_recognition_time >= 2:
            last_recognition_time = current_time
            recognized_faces.clear()  # Clear previous recognition results

            # Recognize each detected face
            for (x, y, width, height) in faces:
                # Process face regions for spoof and recognition separately
                spoof_face_region = frame[y:y+height, x:x+width]
                recog_face_region = frame[max(0, y-20):y+height+20, max(0, x-20):x+width+20]

                is_spoof = not detect_liveness(spoof_face_region)

                if is_spoof:
                    name = "Spoof Detected"
                else:
                    # If all checks pass, perform recognition
                    name, _ = recognize_face(recog_face_region, known_faces, known_names)

                # Add recognized face to the list
                recognized_faces.append((name, x, y, width, height))

        # Update bounding boxes for real-time tracking even without recognition
        for (x, y, width, height) in faces:
            name = "Unknown"
            for stored_name, stored_x, stored_y, stored_width, stored_height in recognized_faces:
                if abs(x - stored_x) < base_margin and abs(y - stored_y) < base_margin:
                    name = stored_name
                    break
            draw_bounding_box(frame, x, y, width, height, name)

        # Show the frame
        cv2.imshow('Face Recognition with Anti-Spoofing', cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()

























