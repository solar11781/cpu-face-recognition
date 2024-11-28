import cv2
import torch
from gfpgan import GFPGANer

def load_gfpgan_model(model_path='GFPGAN\\experiments\\pretrained_models\\GFPGANv1.3.pth'):
    """
    Load the GFPGAN model for face restoration.
    :param model_path: Path to the GFPGAN pre-trained model.
    :param device: Device to run the model on ('cuda' or 'cpu').
    :return: GFPGANer model object.
    """
    gfpgan_model = GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return gfpgan_model

def enhance_frame_with_gfpgan(frame, gfpgan_model):
    """
    Enhance a frame using GFPGAN.
    :param frame: Input image frame in BGR format.
    :param restorer: GFPGANer model object.
    :return: Enhanced image frame in BGR format.
    """
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Restore faces
        cropped_faces, restored_faces, restored_img = gfpgan_model.enhance(
            frame_rgb, has_aligned=False, only_center_face=False, paste_back=True
        )
        if restored_img is not None:
            # Convert RGB back to BGR
            restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            print("Frame enhanced successfully")
            return restored_img_bgr
        else:
            print("No faces were restored.")
            return frame
    except Exception as e:
        print(f"Error enhancing frame: {e}")
        return frame