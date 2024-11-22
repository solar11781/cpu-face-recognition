import cv2

def crop_face(frame, x, y, width, height, margin=40):
    """Crop face region with added margin."""
    h, w = frame.shape[:2]
    x = max(0, x - margin)
    y = max(0, y - margin)
    width = min(width + 2 * margin, w - x)
    height = min(height + 2 * margin, h - y)
    cropped_face = frame[y:y+height, x:x+width]
    print("Cropped face shape:", cropped_face.shape)  # Debugging
    return cropped_face


def init_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

