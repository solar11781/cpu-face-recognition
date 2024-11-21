import cv2
import time
from face_rec import load_known_faces, recognize_face
from face_detect import detect_faces, draw_bounding_box
from anti_spoof import detect_liveness, detect_blink, check_background_consistency
from utils import init_background_subtractor

# Load known faces
known_faces, known_names = load_known_faces()
background_subtractor = init_background_subtractor()

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
                # Adjust bounding box for recognition with dynamic margin
                margin = int(min(width, height) * 0.1) + base_margin
                recog_x = max(0, x - margin)
                recog_y = max(0, y - margin)
                recog_width = min(width + 2 * margin, frame.shape[1] - recog_x)
                recog_height = min(height + 2 * margin, frame.shape[0] - recog_y)

                # Extract the face region for recognition
                recog_face_region = frame[recog_y:recog_y+recog_height, recog_x:recog_x+recog_width]

                # Perform spoof detection with different cropping logic
                spoof_margin = int(min(width, height) * 0.2)  # Larger margin for spoof detection
                spoof_x = max(0, x - spoof_margin)
                spoof_y = max(0, y - spoof_margin)
                spoof_width = min(width + 2 * spoof_margin, frame.shape[1] - spoof_x)
                spoof_height = min(height + 2 * spoof_margin, frame.shape[0] - spoof_y)
                spoof_face_region = frame[spoof_y:spoof_y+spoof_height, spoof_x:spoof_x+spoof_width]

                # Check for liveness, blinking, and background consistency
                is_spoof = not detect_liveness(spoof_face_region)
                is_blinking = detect_blink(frame, (x, y, width, height))
                is_background_consistent = check_background_consistency(frame, background_subtractor)

                # Decide the result based on anti-spoofing checks
                if is_spoof:
                    name = "Spoof Detected"
                elif not is_blinking:
                    name = "No Blink Detected"
                elif not is_background_consistent:
                    name = "Static Background"
                else:
                    # If all checks pass, perform recognition
                    name, _ = recognize_face(recog_face_region, known_faces, known_names)

                # Add recognized face to the list
                recognized_faces.append((name, x, y, width, height))

        # Update bounding boxes for real-time tracking even without recognition
        tracked_faces = []
        for (x, y, width, height) in faces:
            # Adjust bounding box with dynamic margin for drawing
            margin = int(min(width, height) * 0.1) + base_margin
            draw_x = max(0, x - margin)
            draw_y = max(0, y - margin)
            draw_width = min(width + 2 * margin, frame.shape[1] - draw_x)
            draw_height = min(height + 2 * margin, frame.shape[0] - draw_y)

            # Find the closest match in recognized_faces
            name = "Unknown"
            for stored_name, stored_x, stored_y, stored_width, stored_height in recognized_faces:
                # Check proximity and size similarity
                distance = abs(x - stored_x) + abs(y - stored_y)
                size_diff = abs(width - stored_width) + abs(height - stored_height)
                if distance < margin and size_diff < margin:
                    name = stored_name
                    break

            tracked_faces.append((name, draw_x, draw_y, draw_width, draw_height))

        # Draw bounding boxes for all tracked faces
        for (name, x, y, width, height) in tracked_faces:
            draw_bounding_box(frame, x, y, width, height, name)

        # Show the frame
        cv2.imshow('Face Recognition with Anti-Spoofing', cv2.resize(frame, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()























