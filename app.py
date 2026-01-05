import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import os

# Import model architecture
from models.Posmer_7cls import pyramid_mamba_expr2


CHECKPOINT_PATH = "checkpoint/posmer_acc_092047.pth"
D_STATE = 32  # Must match HPO settings (8, 16, or 32)
GPU_ID = "0"


if GPU_ID:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
device = torch.device("cuda" if torch.cuda.is_available() and GPU_ID else "cpu")

CLASS_NAMES = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']


def load_model():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    try:
        # Initialize Architecture
        model = pyramid_mamba_expr2(img_size=224, num_classes=7, d_state=D_STATE)

        # Load Weights
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Clean 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model.\n{e}")
        return None


# Load model ONCE when script starts
model = load_model()


def predict_emotion(image):
    """
    Args:
        image: A PIL Image provided by Gradio
    Returns:
        confidences: Dictionary {'Happy': 0.9, 'Sad': 0.1}
    """
    if model is None:
        return {"Error: Model not loaded": 0.0}

    # 1. Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img_tensor = transform(image).unsqueeze(0).to(device)

        # 2. Inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]  # Get first batch item

        # 3. Format for Gradio
        confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}

        return confidences

    except Exception as e:
        return {f"Error: {str(e)}": 0.0}


# Launch Gradio Interface
if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_emotion,
        inputs=gr.Image(type="pil", label="Upload Face"),
        outputs=gr.Label(num_top_classes=7, label="Predictions"),
        title="POSMER Macro Emotion Recognition",
        description=f"Running on **{device}**. Upload an image to detect emotions using the Mamba-based architecture.",
        flagging_mode="never"
    )

    demo.launch()
    # share=True creates a public link
