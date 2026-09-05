import streamlit as st
from PIL import Image
import numpy as np
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Banana Ripeness Classification",
    page_icon="🍌",
    layout="centered"
)

st.title("🍌 Banana Ripeness Classification")
st.write(
    "Upload a fruit image or use real-time camera detection. "
    "The system applies the background removal preprocessing pipeline before inference."
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def clear_uploader():
    """Callback function to reset the file uploader"""
    st.session_state.uploader_key += 1

@st.cache_resource
def load_model(choice):
    if choice == "Baseline (Best QWK)":
        return YOLO("yolov8_baseline.pt") 
    else:
        return YOLO("yolov8_tuned.pt")

def preprocess_frame(pil_image, target_size=(416, 416)):
    # Convert PIL (RGB) to OpenCV format (BGR)
    img_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Resize & Median Blur (Noise Suppression)
    resized = cv2.resize(image_bgr, target_size)
    median = cv2.medianBlur(resized, 5)

    enhanced = median

    # HSV Conversion & Otsu's Thresholding on Saturation Channel
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1] # Extract the S channel
    
    # Automatically calculate threshold using Otsu's
    _, banana_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleaning (Remove residual noise and fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    banana_mask = cv2.morphologyEx(banana_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component (the main fruit)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(banana_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        banana_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Apply mask to isolate the banana on a pure black background
    banana_only = cv2.bitwise_and(enhanced, enhanced, mask=banana_mask)

    # Convert BGR back to RGB for Streamlit display and YOLO input
    processed_rgb = cv2.cvtColor(banana_only, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb)

def run_inference(pil_image):
    processed_image = preprocess_frame(pil_image)
    results = model(processed_image, verbose=False)
    probs = results[0].probs
    top1_idx = probs.top1
    top1_class = results[0].names[top1_idx].upper()
    top1_conf = float(probs.top1conf.item()) * 100
    return processed_image, top1_class, top1_conf, results

def display_results(raw_image, processed_image, top1_class, top1_conf, results):
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Raw Input", use_container_width=True)
    with col2:
        st.image(processed_image, caption="Preprocessed Frame", use_container_width=True)

    st.markdown("---")
    st.subheader("Grading Result")
    st.success(f"**Predicted Stage:** {top1_class}")
    st.info(f"**Confidence Score:** {top1_conf:.2f}%")

    probs = results[0].probs
    with st.expander("View Full Class Probabilities"):
        for class_idx, class_name in results[0].names.items():
            conf_val = float(probs.data[class_idx].item()) * 100
            st.write(f"- **{class_name.capitalize()}**: {conf_val:.2f}%")
            st.progress(conf_val / 100.0)

class BananaProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.infer_every_n = 5 
        self.last_label = "Waiting for detection..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % self.infer_every_n == 0:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            try:
                _, top1_class, top1_conf, _ = run_inference(pil_img)
                self.last_label = f"{top1_class} ({top1_conf:.1f}%)"
            except Exception:
                self.last_label = "Inference error"

        cv2.putText(
            img, self.last_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("---")
st.subheader("⚙️ Model Configuration")
model_choice = st.radio(
    "Select YOLOv8 Version:",
    ["Baseline (Best QWK)", "Tuned (Highest Accuracy)"],
    horizontal=True,
)
model = load_model(model_choice)

input_mode = st.radio(
    "Select Input Method:",
    ["Upload Image(s)", "Real-Time Webcam"],
    horizontal=True
)

if input_mode == "Upload Image(s)":
    uploaded_files = st.file_uploader(
        "Upload banana image(s) (.jpg, .jpeg, .png)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    ) 

    if uploaded_files:
        st.button("🗑️ Clear Images", on_click=clear_uploader)

        if len(uploaded_files) == 1:
            # Single Image Processing
            raw_image = Image.open(uploaded_files[0]).convert("RGB")
            with st.spinner("Processing image..."):
                processed_image, top1_class, top1_conf, results = run_inference(raw_image)
            display_results(raw_image, processed_image, top1_class, top1_conf, results)
            
        else:
            # Batch Image Processing
            st.markdown("---")
            st.subheader(f"Batch Processing Results ({len(uploaded_files)} images)")
            
            progress_bar = st.progress(0)
            summary_data = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                raw_image = Image.open(uploaded_file).convert("RGB")
                processed_image, top1_class, top1_conf, results = run_inference(raw_image)
                
                summary_data.append({
                    "File": uploaded_file.name,
                    "Raw": raw_image,
                    "Processed": processed_image,
                    "Prediction": top1_class,
                    "Confidence": top1_conf,
                    "Results": results
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            progress_bar.empty()
            
            # Show Summary Metrics dynamically based on classes detected
            class_counts = {}
            for item in summary_data:
                pred = item["Prediction"]
                class_counts[pred] = class_counts.get(pred, 0) + 1
                
            cols = st.columns(len(class_counts))
            for idx, (cls_name, count) in enumerate(class_counts.items()):
                cols[idx].metric(label=f"{cls_name.capitalize()} Bananas", value=count)
                
            st.markdown("---")
            
            # Show Individual Results in Expanders
            for item in summary_data:
                with st.expander(f"🍌 {item['File']} — {item['Prediction']} ({item['Confidence']:.2f}%)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(item['Raw'], caption="Raw Input", use_container_width=True)
                    with c2:
                        st.image(item['Processed'], caption="Preprocessed Frame", use_container_width=True)
                    
                    st.success(f"**Predicted Stage:** {item['Prediction']} | **Top Confidence:** {item['Confidence']:.2f}%")
                    
                    # Detailed Percentage Breakdown for Batch Items
                    res_probs = item['Results'][0].probs
                    st.markdown("**Full Class Probabilities:**")
                    for class_idx, class_name in item['Results'][0].names.items():
                        conf_val = float(res_probs.data[class_idx].item()) * 100
                        st.write(f"- **{class_name.capitalize()}**: {conf_val:.2f}%")
                        st.progress(conf_val / 100.0)
                        
else:  
    st.write("Live detection — point the camera at a banana. Label updates every few frames.")
    webrtc_streamer(
        key="banana-realtime",
        video_processor_factory=BananaProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )