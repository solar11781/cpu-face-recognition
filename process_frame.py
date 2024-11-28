import cv2
from realesrgan import RealESRGANer


def load_realesrgan_model(model_path='Real-ESRGAN\experiments\pretrained_models\RealESRGAN_x4plus.pth'):
    """
    Load the Real-ESRGAN model for image enhancement.
    :param model_path: Path to the Real-ESRGAN pre-trained model.
    :return: RealESRGAN model object.
    """
    model = RealESRGANer(scale=4, model_path=model_path, half=True)
    return model

def enhance_frame_with_realesrgan(frame, model):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enhanced_frame, _ = model.enhance(frame_rgb, outscale=4)
        print("Frame enhanced successfully")
        return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error enhancing frame: {e}")
        return frame