import streamlit as st
import cv2
import tempfile
from moviepy.editor import VideoFileClip, vfx
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
    
    # Create a gradient mask for the blur
    mask = np.zeros((height, width), dtype=np.float32)
    focus_start = int((focus_position - focus_width / 2) / 100 * height)
    focus_end = int((focus_position + focus_width / 2) / 100 * height)
    mask[focus_start:focus_end, :] = 1.0
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    # Apply the blur effect
    blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)
    frame = np.uint8(frame * mask[:, :, None] + blurred_frame * (1 - mask[:, :, None]))
    
    # Apply saturation adjustment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation_level / 100)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return frame

# Function to process the entire video with tilt-shift effect
def process_tilt_shift_video(input_video_path, output_video_path, focus_position, focus_width, blur_intensity, saturation_level, speed):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec
    fps = int(cap.get(cv2.CAP_PROP_FPS) * (speed / 100))  # Adjust FPS for speed
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = 0
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply tilt-shift effect to each frame
        processed_frame = apply_tilt_shift(frame, focus_position, focus_width, blur_intensity, saturation_level)
        out.write(processed_frame)

        # Update progress bar
        progress += 1
        progress_bar.progress(progress / total_frames)

    cap.release()
    out.release()

# Streamlit UI
st.title("Create a miniature video using tilt shift")
st.markdown("Upload a video that is taken from a top viewpoint, adjust the tilt-shift effect parameters, and download the processed video.")

# Video uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save the uploaded video to a temporary file
    temp_input_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input_video.write(uploaded_video.read())
    temp_input_video.close()

    # Extract the middle frame from the video
    middle_frame = extract_middle_frame(temp_input_video.name)

    if middle_frame is not None:
        # Convert the frame from BGR to RGB for displaying
        middle_frame_rgb = cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB)

        # Sliders for tilt-shift parameters
        focus_position = st.slider("Line of focus position (%)", min_value=0, max_value=100, step=1, value=50)
        focus_width = st.slider("Line of focus width (%)", min_value=5, max_value=50, step=1, value=20)
        blur_intensity = st.slider("Blur Intensity", min_value=3, max_value=51, step=2, value=15)
        saturation_level = st.slider("Saturation Level (%)", min_value=50, max_value=200, step=10, value=100)
        speed = st.slider("Speed (%)", min_value=100, max_value=200, step=10, value=100)

        # Ensure blur intensity is odd (required for GaussianBlur)
        if blur_intensity % 2 == 0:
            blur_intensity += 1

        # Apply tilt-shift effect to the middle frame
        processed_frame = apply_tilt_shift(middle_frame, focus_position, focus_width, blur_intensity, saturation_level)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Display the processed middle frame
        st.markdown("### Preview")
        st.image(Image.fromarray(processed_frame_rgb), caption="Preview", use_container_width=True)

        # Button to process the entire video
        if st.button("Process and Download Full Video"):
            # Output file path
            temp_output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_output_video.close()

            # Process the full video
            with st.spinner("Processing the full video..."):
                process_tilt_shift_video(
                    temp_input_video.name,
                    temp_output_video.name,
                    focus_position,
                    focus_width,
                    blur_intensity,
                    saturation_level,
                    speed
                )

            # Display success message and provide download link
            st.success("Full video processed successfully!")
            with open(temp_output_video.name, "rb") as file:
                st.download_button(
                    label="Download Tilt-Shift Video",
                    data=file,
                    file_name="tilt_shift_video.mp4",
                    mime="video/mp4"
                )
    else:
        st.error("Could not extract a frame from the uploaded video.")
