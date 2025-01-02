import streamlit as st
import cv2
import tempfile
from PIL import Image
import numpy as np

# Function to extract the middle frame from the video
def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return frame
    return None

# Function to apply tilt-shift effect
def apply_tilt_shift(frame, focus_position, focus_width, blur_intensity, saturation_level):
    height, width, _ = frame.shape

    # Reverse the direction of focus position (100 = top, 0 = bottom)
    focus_position = 100 - focus_position

    # Create a gradient mask for the blur
    mask = np.zeros((height, width), dtype=np.float32)
    focus_start = int((focus_position - focus_width / 2) / 100 * height)
    focus_end = int((focus_position + focus_width / 2) / 100 * height)
    gradient_height = int(focus_width / 2 / 100 * height)

    for y in range(height):
        if y < focus_start - gradient_height:
            mask[y, :] = 0.0
        elif focus_start - gradient_height <= y < focus_start:
            mask[y, :] = (y - (focus_start - gradient_height)) / gradient_height
        elif focus_start <= y <= focus_end:
            mask[y, :] = 1.0
        elif focus_end < y <= focus_end + gradient_height:
            mask[y, :] = 1.0 - (y - focus_end) / gradient_height
        else:
            mask[y, :] = 0.0

    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)
    frame = np.uint8(frame * mask[:, :, None] + blurred_frame * (1 - mask[:, :, None]))

    # Apply saturation adjustment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_level / 100)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame

# Function to process the entire video
def process_tilt_shift_video(input_video_path, output_video_path, focus_position, focus_width, blur_intensity, saturation_level, speed):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec
    fps = int(cap.get(cv2.CAP_PROP_FPS) * (speed / 100))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply tilt-shift effect to each frame
        processed_frame = apply_tilt_shift(frame, focus_position, focus_width, blur_intensity, saturation_level)
        out.write(processed_frame)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()

# Streamlit UI
st.title("Create a miniature video using tilt shift")
st.markdown("Upload a video or picture taken from a top viewpoint, adjust the tilt-shift effect parameters, and download the processed result.")

# File uploader for video or picture
uploaded_file = st.file_uploader("Upload a video or picture", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file:
    is_picture = uploaded_file.type.startswith("image")

    if is_picture:
        input_image = Image.open(uploaded_file)
        input_image_array = np.array(input_image)
    else:
        temp_input_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input_video.write(uploaded_file.read())
        temp_input_video.close()
        middle_frame = extract_middle_frame(temp_input_video.name)

    # Adjustable parameters
    focus_position = st.slider("Line of focus position (%)", min_value=0, max_value=100, step=1, value=50)
    focus_width = st.slider("Line of focus width (%)", min_value=5, max_value=50, step=1, value=20)
    blur_intensity = st.slider("Blur Intensity", min_value=3, max_value=51, step=2, value=15)
    saturation_level = st.slider("Saturation Level (%)", min_value=50, max_value=200, step=10, value=100)

    if not is_picture:
        speed = st.slider("Speed (%)", min_value=50, max_value=200, step=10, value=100)

    # Display the live preview
    st.markdown("### Live Preview")
    if is_picture:
        processed_image = apply_tilt_shift(input_image_array, focus_position, focus_width, blur_intensity, saturation_level)
        st.image(Image.fromarray(processed_image), caption="Processed Picture", use_container_width=True)

        # Add a download button for the processed picture
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image_file:
            processed_image_pil = Image.fromarray(processed_image)
            processed_image_pil.save(temp_image_file.name)
            st.download_button(
                label="Download Processed Picture",
                data=open(temp_image_file.name, "rb").read(),
                file_name="processed_picture.jpg",
                mime="image/jpeg"
            )
    else:
        if middle_frame is not None:
            processed_frame = apply_tilt_shift(middle_frame, focus_position, focus_width, blur_intensity, saturation_level)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Frame (Live Preview)", use_container_width=True)

        # Process video only when download is requested
        if st.button("Download Processed Video"):
            temp_output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            process_tilt_shift_video(
                temp_input_video.name,
                temp_output_video.name,
                focus_position,
                focus_width,
                blur_intensity,
                saturation_level,
                speed
            )
            with open(temp_output_video.name, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="tilt_shift_video.mp4",
                    mime="video/mp4"
                )
