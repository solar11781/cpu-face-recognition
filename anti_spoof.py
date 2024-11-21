import cv2
import torch
from torchvision import transforms
from SilentFaceAntiSpoofing.resources.anti_spoof_models.MiniFASNet import MiniFASNetV2

def load_anti_spoofing_model(model_path="minifasnet_model.pth"):
    # Update the model parameters to match the pre-trained model
    model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5), drop_p=0.2, num_classes=3, img_channel=3)
    
    # Load the state_dict and adjust keys if needed
    state_dict = torch.load("minifasnet_model.pth", map_location=torch.device("cpu"))
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.eval()
    return model




anti_spoofing_model = load_anti_spoofing_model("minifasnet_model.pth")

# Load eye detector
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def detect_liveness(face_region):
    """Run MiniFASNet anti-spoofing model on the cropped face."""
    input_tensor = transform(face_region).unsqueeze(0)
    with torch.no_grad():
        output = anti_spoofing_model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
    return probabilities[0, 1].item() > 0.5  # Live face class corresponds to 1.

def detect_blink(frame, face_bbox):
    """Detect blinks using OpenCV's Haar cascades."""
    x, y, w, h = face_bbox
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    return len(eyes) > 0  # Blink detected if at least one eye is detected.

def track_head_movement(bounding_boxes, prev_boxes):
    """Detect head movements based on bounding box deltas."""
    movements = []
    for (x, y, w, h), (px, py, pw, ph) in zip(bounding_boxes, prev_boxes):
        delta = abs(x - px) + abs(y - py)
        movements.append(delta)
    return any(delta > 10 for delta in movements)

def check_background_consistency(frame, background_subtractor):
    """Check if the background is consistent (e.g., not static)."""
    fg_mask = background_subtractor.apply(frame)
    motion_percentage = (cv2.countNonZero(fg_mask) / float(frame.size)) * 100
    return motion_percentage > 5.0  # Threshold for significant motion

