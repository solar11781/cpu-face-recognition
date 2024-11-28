# # BASE VERSION WITH NO ANTI SPOOFING

# import os
# from deepface import DeepFace

# KNOWN_FACES_DIR = 'D:/testing ground/Face recognition asm/Mini-Face-recognition-using-Deepface-and-GAN/Face testing'
# RECOGNITION_THRESHOLD = 0.5  # Adjust as needed (lower is stricter)

# def load_known_faces():
#     known_faces = []
#     known_names = []
#     for person_name in os.listdir(KNOWN_FACES_DIR):
#         person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
#         if not os.path.isdir(person_folder):
#             continue
#         for image_name in os.listdir(person_folder):
#             image_path = os.path.join(person_folder, image_name)
#             try:
#                 # Use MTCNN as the face detection backend
#                 face = DeepFace.represent(img_path=image_path, model_name="VGG-Face", detector_backend="mtcnn")
#                 known_faces.append(face)
#                 known_names.append(person_name)
#             except Exception as e:
#                 print(f"Error processing {image_path}: {e}")
#     return known_faces, known_names

# def recognize_face(frame, known_faces, known_names):
#     try:
#         # Use MTCNN as the face detection backend for real-time recognition
#         result = DeepFace.find(frame, db_path=KNOWN_FACES_DIR, model_name="VGG-Face", enforce_detection=False, detector_backend="mtcnn")
#         if result:
#             for df in result:
#                 if not df.empty:
#                     # Get the closest match
#                     best_match = df.loc[df['distance'].idxmin()]
#                     if best_match['distance'] <= RECOGNITION_THRESHOLD:
#                         recognized_name = os.path.basename(os.path.dirname(best_match['identity']))
#                         return recognized_name, best_match
#                     else:
#                         return "Unknown", None
#     except Exception as e:
#         print(f"Error during recognition: {e}")
#     return "Unknown", None

###########################################################################################################################################

import os
from deepface import DeepFace

KNOWN_FACES_DIR = 'Face testing'
RECOGNITION_THRESHOLD = 0.5

def load_known_faces():
    known_faces = []
    known_names = []
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            try:
                face = DeepFace.represent(img_path=image_path, model_name="VGG-Face", detector_backend="mtcnn")
                known_faces.append(face)
                known_names.append(person_name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return known_faces, known_names

def recognize_face(frame, known_faces, known_names):
    try:
        result = DeepFace.find(frame, db_path=KNOWN_FACES_DIR, model_name="VGG-Face", enforce_detection=False, detector_backend="mtcnn")
        if result:
            for df in result:
                if not df.empty:
                    best_match = df.loc[df['distance'].idxmin()]
                    if best_match['distance'] <= RECOGNITION_THRESHOLD:
                        recognized_name = os.path.basename(os.path.dirname(best_match['identity']))
                        return recognized_name, best_match
                    else:
                        return "Unknown", None
    except Exception as e:
        print(f"Error during recognition: {e}")
    return "Unknown", None



