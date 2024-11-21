import cv2
import torch
from torchvision import transforms
from SilentFaceAntiSpoofing.resources.anti_spoof_models.MiniFASNet import MiniFASNetV2


def load_anti_spoofing_model(model_path="adjusted_minifasnet_model.pth"):
    """Load MiniFASNet anti-spoofing model."""
    model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5), drop_p=0.2, num_classes=3, img_channel=3)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


anti_spoofing_model = load_anti_spoofing_model()

# Transform pipeline for input preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def detect_liveness(face_region):
    """Run MiniFASNet anti-spoofing model on the cropped face."""
    try:
        input_tensor = transform(face_region).unsqueeze(0)  # Add batch dimension
        print("Input tensor shape:", input_tensor.shape)
        with torch.no_grad():
            output = anti_spoofing_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
        print("Liveness probabilities:", probabilities.tolist())
        liveness_score = probabilities[0, 1].item()
        print(f"Liveness Score: {liveness_score}")
        return liveness_score > 0.5  # Lowered threshold
    except Exception as e:
        print(f"Error in detect_liveness: {e}")
        return False


def detect_blink(frame, face_bbox):
    """Detect if the person is blinking."""
    x, y, w, h = face_bbox
    face_roi = frame[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's pre-trained eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return True if at least one eye is detected (blinking can be further enhanced with temporal checks)
    print(f"Blink detection: {'Blink detected' if len(eyes) > 0 else 'No blink detected'}")
    return len(eyes) > 0


def check_background_consistency(frame, background_subtractor):
    """Check if the background is static."""
    fg_mask = background_subtractor.apply(frame)
    motion_detected = cv2.countNonZero(fg_mask) > (0.01 * frame.size)
    print(f"Background check: {'Motion detected' if motion_detected else 'Static background detected'}")
    return motion_detected




