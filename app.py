import streamlit as st
import torch
import tempfile
import cv2
from models.deepfake_model import DeepfakeDetector
from preprocessing.face_detector import mtcnn
from torchvision import transforms

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎥",
    layout="centered"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load("training/deepfake_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- HEADER --------------------
st.markdown(
    "<h1 style='text-align: center;'>🎥 AI Deepfake Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload a video and let AI detect if it is <b>Real</b> or <b>Fake</b>.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    if st.button("🔍 Analyze Video"):
        progress = st.progress(0)

        with st.spinner("Analyzing frames..."):

            cap = cv2.VideoCapture(video_path)
            frames = []
            count = 0

            while len(frames) < 10:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = mtcnn(frame_rgb)

                if face is not None:
                    face = face.permute(1, 2, 0).cpu().numpy()
                    face = transform(face)
                    frames.append(face)

                count += 1
                progress.progress(min(count * 10, 100))

            cap.release()

            progress.empty()

            if len(frames) == 0:
                st.error("❌ No face detected in the video.")
            else:
                frames = torch.stack(frames).to(device)

                with torch.no_grad():
                    outputs = model(frames)
                    prediction = outputs.mean().item()

                st.markdown("---")
                st.subheader("🔍 Result")

                if prediction > 0.007:
                    confidence = prediction
                    st.success("✅ REAL VIDEO")
                else:
                    confidence = 1 - prediction
                    st.error("🚨 FAKE VIDEO DETECTED")

                # -------------------- CONFIDENCE BAR --------------------
                st.write(f"**Confidence:** {confidence:.2f}")
                st.progress(confidence)

                with st.expander("📊 View Technical Details"):
                    st.write("Frame Predictions:", outputs.cpu().numpy())
                    st.write("Fake Probability Score:", prediction)

# -------------------- FOOTER --------------------
st.markdown("---")
# st.caption("Built using CNN + PyTorch | Deepfake Detection Project")

# if prediction > 0.007:
#     st.success(f"✅ REAL VIDEO")
#     st.write(f"Confidence: {1 - prediction:.2f}")
# else:
#     st.error(f"🚨 FAKE VIDEO")
#     st.write(f"Confidence: {prediction:.2f}")