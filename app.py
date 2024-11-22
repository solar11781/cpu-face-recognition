from flask import Flask, request, render_template, Response, stream_with_context
import os
import cv2
from deepface import DeepFace
from main import recognize_faces
from face_rec import load_known_faces

app = Flask(__name__)

# Directories
FACE_TESTING_DIR = "./Face testing"
BUFFER_DIR = "./buffer"
if not os.path.exists(FACE_TESTING_DIR):
    os.makedirs(FACE_TESTING_DIR)
if not os.path.exists(BUFFER_DIR):
    os.makedirs(BUFFER_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    @stream_with_context
    def generate():
        # Step 1: Get the name of the person from the form
        name = request.form.get('person_name')
        if not name:
            print("Error: Please enter your name.")
            yield "Error: Please enter your name.\n"
            return

        # Step 2: Create the user's directory inside FACE_TESTING_DIR
        user_dir = os.path.join(FACE_TESTING_DIR, name)
        os.makedirs(user_dir, exist_ok=True)

        # Step 3: Initialize the camera feed
        cap = cv2.VideoCapture(0)  # Change 0 to another index for an external camera
        if not cap.isOpened():
            print("Error: Unable to access the camera.")
            yield "Error: Unable to access the camera.\n"
            return

        yield "Press SPACE to capture an image, ESC to exit.\n"
        print("Press SPACE to capture an image, ESC to exit.")

        # Step 4: Define the array of face positions
        face_positions = [
            "Up", "Down", "Left", "Right", "Front",
            "Face tilted down to the left", "Face tilted down to the right"
        ]
        captured_images = 0  # Count of successfully captured images

        while captured_images < 7:  # Maximum 7 images
            ret, frame = cap.read()  # Capture a frame from the camera
            if not ret:
                print("Error: Unable to read from the camera.")
                yield "Error: Unable to read from the camera.\n"
                break

            # Step 5: Show the webcam feed to the user
            cv2.imshow("Camera - Press SPACE to Capture, ESC to Exit", frame)

            # Step 6: Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            elif key == 32:  # SPACE key to capture an image
                # Save the captured image to the buffer folder
                buffer_path = os.path.join(BUFFER_DIR, f"{name}_buffer.jpg")
                cv2.imwrite(buffer_path, frame)
                print(f"Captured image saved in buffer: {buffer_path}")

                # Check for face orientation
                detected_position = process_buffer_image(buffer_path)

                # Step 8: Validate the returned position
                if detected_position in face_positions:
                    # If the position is valid, move the image to the embeddings folder
                    final_path = os.path.join(user_dir, f"{name}_{captured_images + 1}.jpg")
                    os.rename(buffer_path, final_path)
                    print(f"Image moved to: {final_path}")
                    captured_images += 1
                    print(f"Captured {captured_images} valid images so far.")
                    
                    # Remove the detected position from the array
                    face_positions.remove(detected_position)
                    print(f"Position '{detected_position}' removed from the list. Remaining positions: {face_positions}")

                    yield f"Captured {captured_images} valid images so far.\n"
                else:
                    # If the position is invalid, delete the image from the buffer
                    os.remove(buffer_path)
                    print(f"Invalid position detected. Image deleted from buffer.")
                    yield f"Invalid position detected. Remaining positions: {face_positions}\n"

        # Step 9: Release the camera and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()

        yield f"Registration completed for {name} with {captured_images} images."
        yield f"REDIRECT:/success/{name}/{captured_images}"

        print(f"Registration completed for {name} with {captured_images} images.")
    return Response(generate(), content_type="text/html")

@app.route('/success/<name>/<int:captured_images>')
def success(name, captured_images):
    return render_template(
        'success.html',
        message=f"Registration completed for {name} with {captured_images} images."
    )

# Dummy function to return a face position
def process_buffer_image(buffer_path):
    if not hasattr(process_buffer_image, "index"):
        process_buffer_image.index = 0
    
    face_positions = [
        "Up", "Down", "Left", "Right", "Front",
        "Face tilted down to the left", "Face tilted down to the right"
    ]
    
    # Get the current face position and update the index
    position = face_positions[process_buffer_image.index]
    process_buffer_image.index = (process_buffer_image.index + 1) % len(face_positions)
    
    return position

# Load known faces
known_faces, known_names = load_known_faces()

@app.route('/recognize', methods=['POST'])
def recognize():
    message = recognize_faces(known_faces, known_names)
    return render_template('recognition_completed.html', message=message)
    
if __name__ == '__main__':
    app.run(debug=True)