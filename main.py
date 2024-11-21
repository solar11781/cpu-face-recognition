# import cv2
# import time
# from face_rec import load_known_faces, recognize_face
# from face_detect import detect_faces, draw_bounding_box

# # Load known faces
# known_faces, known_names = load_known_faces()

# def recognize_faces():
#     video_capture = cv2.VideoCapture(0)
#     last_recognition_time = 0
#     base_margin = 20  # Base margin to increase bounding box size for matching
#     recognized_faces = []  # List to store recognized faces with names and bounding boxes

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         # Detect faces continuously for real-time bounding box tracking
#         faces = detect_faces(frame)

#         # Perform recognition only every 2 seconds
#         current_time = time.time()
#         if current_time - last_recognition_time >= 2:
#             last_recognition_time = current_time
#             recognized_faces.clear()  # Clear previous recognition results

#             # Recognize each detected face
#             for (x, y, width, height) in faces:
#                 # Adjust bounding box with base margin
#                 margin = int(min(width, height) * 0.1) + base_margin  # Dynamic margin based on face size
#                 x = max(0, x - margin)
#                 y = max(0, y - margin)
#                 width = min(width + 2 * margin, frame.shape[1] - x)
#                 height = min(height + 2 * margin, frame.shape[0] - y)

#                 # Extract the face region for recognition
#                 face_region = frame[y:y+height, x:x+width]
#                 name, _ = recognize_face(face_region, known_faces, known_names)

#                 # Add the recognized face to the list with its bounding box
#                 recognized_faces.append((name, x, y, width, height))

#         # Update bounding boxes for continuous tracking even without new recognition
#         tracked_faces = []
#         for (x, y, width, height) in faces:
#             # Adjust bounding box with dynamic margin
#             margin = int(min(width, height) * 0.1) + base_margin
#             x = max(0, x - margin)
#             y = max(0, y - margin)
#             width = min(width + 2 * margin, frame.shape[1] - x)
#             height = min(height + 2 * margin, frame.shape[0] - y)

#             # Find the closest match in recognized_faces by proximity and size
#             name = None
#             best_match = None
#             min_distance = float('inf')
#             for stored_name, stored_x, stored_y, stored_width, stored_height in recognized_faces:
#                 # Calculate distance based on position and size similarity
#                 distance = abs(x - stored_x) + abs(y - stored_y)
#                 size_difference = abs(width - stored_width) + abs(height - stored_height)
                
#                 # Update best match if this face is closer and has similar size
#                 if distance < min_distance and size_difference < margin:
#                     min_distance = distance
#                     best_match = (stored_name, stored_x, stored_y, stored_width, stored_height)

#             # Use the best match's name if found
#             if best_match:
#                 name = best_match[0]

#             tracked_faces.append((name, x, y, width, height))

#         # Draw bounding boxes and names for each face in tracked_faces
#         for (name, x, y, width, height) in tracked_faces:
#             draw_bounding_box(frame, x, y, width, height, name)

#         # Show the frame
#         cv2.imshow('Face Recognition with OpenCV DNN', cv2.resize(frame, (640, 480)))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     recognize_faces()

###################################################################################

import cv2
from face_detect import detect_faces, draw_bounding_box
from face_rec import load_known_faces, recognize_face
from anti_spoof import detect_liveness, detect_blink, track_head_movement, check_background_consistency
from utils import crop_face, init_background_subtractor

# Load known faces
known_faces, known_names = load_known_faces()
background_subtractor = init_background_subtractor()

def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    prev_boxes = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        faces = detect_faces(frame)
        bounding_boxes = []
        for (x, y, width, height) in faces:
            face_region = crop_face(frame, x, y, width, height)
            bounding_boxes.append((x, y, width, height))

            # Anti-Spoofing
            if not detect_liveness(face_region):
                print("Spoof detected!")
                continue

            # Head Movement
            if prev_boxes and not track_head_movement(bounding_boxes, prev_boxes):
                print("No head movement detected! Potential spoof.")
                continue

            # Background Consistency
            if not check_background_consistency(frame, background_subtractor):
                print("Static background detected! Potential spoof.")
                continue

            # Blink Detection
            if not detect_blink(frame, (x, y, width, height)):
                print("No blinking detected! Potential spoof.")
                continue

            # Recognition
            name, _ = recognize_face(face_region, known_faces, known_names)
            draw_bounding_box(frame, x, y, width, height, name)

        prev_boxes = bounding_boxes
        cv2.imshow('Real-Time Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()




###################################################################################
#save image that is used to put into vgg face 
# def recognize_faces():
#     video_capture = cv2.VideoCapture(0)
#     last_recognition_time = 0
#     base_margin = 20  # Base margin to increase bounding box size for matching
#     recognized_faces = []  # List to store recognized faces with names and bounding boxes
#     save_dir = os.path.dirname(__file__)  # Directory to save the images

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         # Detect faces continuously for real-time bounding box tracking
#         faces = detect_faces(frame)

#         # Perform recognition only every 2 seconds
#         current_time = time.time()
#         if current_time - last_recognition_time >= 2:
#             last_recognition_time = current_time
#             recognized_faces.clear()  # Clear previous recognition results

#             # Recognize each detected face
#             for i, (x, y, width, height) in enumerate(faces):
#                 # Adjust bounding box with base margin
#                 margin = int(min(width, height) * 0.1) + base_margin
#                 x = max(0, x - margin)
#                 y = max(0, y - margin)
#                 width = min(width + 2 * margin, frame.shape[1] - x)
#                 height = min(height + 2 * margin, frame.shape[0] - y)

#                 # Extract the face region for recognition
#                 face_region = frame[y:y+height, x:x+width]
                
#                 # Save the face image to the directory
#                 face_image_path = os.path.join(save_dir, f"face_input_{i}_{int(current_time)}.jpg")
#                 cv2.imwrite(face_image_path, face_region)

#                 # Perform recognition
#                 name, _ = recognize_face(face_region, known_faces, known_names)

#                 # Add the recognized face to the list with its bounding box
#                 recognized_faces.append((name, x, y, width, height))

#         # Update bounding boxes for continuous tracking even without new recognition
#         tracked_faces = []
#         for (x, y, width, height) in faces:
#             # Adjust bounding box with dynamic margin
#             margin = int(min(width, height) * 0.1) + base_margin
#             x = max(0, x - margin)
#             y = max(0, y - margin)
#             width = min(width + 2 * margin, frame.shape[1] - x)
#             height = min(height + 2 * margin, frame.shape[0] - y)

#             # Find the closest match in recognized_faces by proximity and size
#             name = None
#             best_match = None
#             min_distance = float('inf')
#             for stored_name, stored_x, stored_y, stored_width, stored_height in recognized_faces:
#                 # Calculate distance based on position and size similarity
#                 distance = abs(x - stored_x) + abs(y - stored_y)
#                 size_difference = abs(width - stored_width) + abs(height - stored_height)
                
#                 # Update best match if this face is closer and has similar size
#                 if distance < min_distance and size_difference < margin:
#                     min_distance = distance
#                     best_match = (stored_name, stored_x, stored_y, stored_width, stored_height)

#             # Use the best match's name if found
#             if best_match:
#                 name = best_match[0]

#             tracked_faces.append((name, x, y, width, height))

#         # Draw bounding boxes and names for each face in tracked_faces
#         for (name, x, y, width, height) in tracked_faces:
#             draw_bounding_box(frame, x, y, width, height, name)

#         # Show the frame
#         cv2.imshow('Face Recognition with OpenCV DNN', cv2.resize(frame, (640, 480)))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     recognize_faces()

















