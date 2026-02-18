import streamlit as st
import json
import cv2
from ultralytics import YOLOv10
import numpy as np
import math
import re
import os
import sqlite3
from datetime import datetime
from paddleocr import PaddleOCR
import format_mapper
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create necessary directories if they don't exist
os.makedirs("temp", exist_ok=True)
os.makedirs("json", exist_ok=True)

# Initialize database if needed
conn = sqlite3.connect('licensePlatesDatabase.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS LicensePlates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        license_plate TEXT,
        valid_or_invalid TEXT
    )
''')
conn.commit()
conn.close()

# Streamlit UI
st.title("License Plate Detection and Recognition")

# Sidebar for configuration
with st.sidebar:
    st.header("Processing Settings")
    # Add a slider for frame processing rate
    frame_skip = st.slider("Frame Skip Rate (higher = faster)", 0, 10, 2, 
                         help="Skip frames to speed up processing. Higher values mean faster processing but might miss some plates.")
    
    # Add a slider for resizing factor
    resize_factor = st.slider("Resize Factor (lower = faster)", 0.3, 1.0, 0.5, 0.1,
                            help="Resize input frames for faster processing. Lower values mean faster processing but might reduce accuracy.")
    
    # Add confidence threshold slider
    conf_threshold = st.slider("Detection Confidence", 0.3, 0.9, 0.45, 0.05,
                             help="Minimum confidence score for license plate detection.")
    
    # Add processing mode selection
    processing_mode = st.radio("Processing Mode", 
                              ["Balanced", "Speed Priority", "Accuracy Priority"],
                              help="Select processing mode based on your requirements")

uploaded_file = st.file_uploader("Upload a Video or Image", type=["mp4", "jpg", "png"])

# Add a checkbox to enable/disable real-time display
show_realtime = st.checkbox("Show real-time detection", value=True)

# Apply processing mode settings
if processing_mode == "Speed Priority":
    resize_factor = min(resize_factor, 0.4)
    frame_skip = max(frame_skip, 3)
    conf_threshold = 0.4
elif processing_mode == "Accuracy Priority":
    resize_factor = max(resize_factor, 0.7)
    frame_skip = min(frame_skip, 1)
    conf_threshold = 0.35

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("File Uploaded Successfully!")
    
    cap = cv2.VideoCapture(file_path) if uploaded_file.type == "video/mp4" else None
    
    # Initialize the YOLOv10 Model with device selection
    model = YOLOv10("weights/best.pt")

    # Class Names
    className = ["License"]
    
    # Initialize the Paddle OCR with lower precision for speed
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang="en")
    
    def paddle_ocr(frame, x1, y1, x2, y2):
        # Extract license plate region with padding
        plate_img = frame[max(0, y1-5):min(frame.shape[0], y2+5), 
                         max(0, x1-5):min(frame.shape[1], x2+5)]
        
        if plate_img.size == 0:
            return ""
            
        # Try enhancing the image for better OCR
        plate_img = cv2.resize(plate_img, (0, 0), fx=1.5, fy=1.5)
        plate_img = cv2.convertScaleAbs(plate_img, alpha=1.2, beta=10)  # Increase contrast
        
        result = ocr.ocr(plate_img, det=False, rec=True, cls=False)
        text = ""
        for r in result:
            scores = r[0][1]
            scores = 0 if np.isnan(scores) else int(scores * 100)
            if scores > 60:
                text = r[0][0]
        
        # Clean up the text
        text = re.sub(r'\W', '', text).replace("O", "0").replace(" ", "").replace("\n", "")
        return str(text)
    
    def save_json(license_plates, startTime, endTime):
        result = format_mapper.checker(license_plates)
        license_plate = result[0]
        if not license_plate:
            return
        interval_data = {
            "Start Time": startTime.isoformat(),
            "End Time": endTime.isoformat(),
            "License Plate": license_plate,
            "Valid or Invalid": result[1]
        }
        cummulative_file_path = "json/LicensePlateData.json"
        existing_data = []
        if os.path.exists(cummulative_file_path):
            with open(cummulative_file_path, 'r') as f:
                existing_data = json.load(f)
        existing_data.append(interval_data)
        with open(cummulative_file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        save_to_database(license_plate, startTime, endTime, result[1])
    
    def save_to_database(license_plate, start_time, end_time, valid_or_invalid):
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO LicensePlates(start_time, end_time, license_plate, valid_or_invalid)
            VALUES (?, ?, ?, ?)
        ''', (start_time.isoformat(), end_time.isoformat(), license_plate, valid_or_invalid))
        conn.commit()
        conn.close()
    
    # Create a placeholder for the real-time video display
    if show_realtime and cap:
        video_placeholder = st.empty()
        status_text = st.empty()
        current_plates_text = st.empty()
        performance_metrics = st.empty()
    elif not cap and uploaded_file.type in ["image/jpeg", "image/png"]:
        image_placeholder = st.empty()
    
    startTime = datetime.now()
    license_plates = set()
    count = 0
    frame_count = 0
    processing_times = []
    
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        # Handle image
        frame = cv2.imread(file_path)
        
        # Resize for faster processing
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))
        
        start_process = time.time()
        results = model.predict(frame_resized, conf=conf_threshold)
        end_process = time.time()
        
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Scale coordinates back to original size
                x1, y1, x2, y2 = map(int, box.xyxy[0] / resize_factor)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Draw rectangle
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Process OCR
                label = paddle_ocr(frame, x1, y1, x2, y2)
                
                # Add text label
                if label:
                    license_plates.add(label)
                    cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image with detections
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        st.image(vis_frame_rgb, caption="License Plate Detection", use_container_width=True)
        
        st.text(f"Processing time: {end_process - start_process:.3f} seconds")
        
        if license_plates:
            st.subheader("Detected License Plates:")
            for plate in license_plates:
                st.write(plate)
            
            # Save the detected plates
            endTime = datetime.now()
            save_json(license_plates, startTime, endTime)
            
    elif cap and cap.isOpened():
        current_license_plates = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames to speed up processing
            if frame_count % (frame_skip + 1) != 0 and frame_count > 1:
                continue
                
            currentTime = datetime.now()
            count += 1
            
            # Resize frame for faster processing
            h, w = frame.shape[:2]
            frame_resized = cv2.resize(frame, (int(w * resize_factor), int(h * resize_factor)))
            
            # Start timing the processing
            start_process = time.time()
            
            # Run model on resized frame
            results = model.predict(frame_resized, conf=conf_threshold)
            
            # Create a copy for visualization
            vis_frame = frame.copy()
            
            detected_in_frame = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Scale coordinates back to original size
                    x1, y1, x2, y2 = map(int, box.xyxy[0] / resize_factor)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Draw rectangle
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Process OCR only if box is large enough (filter out tiny detections)
                    if (x2-x1)*(y2-y1) > 1000:
                        label = paddle_ocr(frame, x1, y1, x2, y2)
                        if label and len(label) >= 4:  # Only consider reasonable length license plates
                            license_plates.add(label)
                            detected_in_frame.append(label)
                            
                            # Add text label on the visualization
                            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # End timing and calculate processing time
            end_process = time.time()
            process_time = end_process - start_process
            processing_times.append(process_time)
            
            # Update progress bar
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            # Display the frame if real-time viewing is enabled
            if show_realtime:
                vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(vis_frame_rgb, caption="Real-time License Plate Detection", use_container_width=True)
                
                # Show currently visible plates
                if detected_in_frame:
                    current_plates_text.markdown(f"**Currently visible plates:** {', '.join(detected_in_frame)}")
                
                # Show performance metrics
                avg_time = sum(processing_times[-20:]) / max(len(processing_times[-20:]), 1)
                effective_fps = 1 / max(avg_time, 0.001)
                time_elapsed = (currentTime - startTime).total_seconds()
                
                metrics_text = (
                    f"**Performance Metrics:**\n"
                    f"- Processing time: {process_time:.3f} sec/frame\n"
                    f"- Effective FPS: {effective_fps:.1f}\n"
                    f"- Elapsed time: {time_elapsed:.1f} seconds\n"
                    f"- Frames processed: {count}/{frame_count}\n"
                    f"- Total detections: {len(license_plates)}"
                )
                performance_metrics.markdown(metrics_text)
            
            if (currentTime - startTime).seconds >= 10:
                endTime = currentTime
                save_json(license_plates, startTime, endTime)
                startTime = currentTime
                license_plates.clear()
            
            if cv2.waitKey(1) & 0xFF == ord('1'):
                break
        
        cap.release()
    
    cv2.destroyAllWindows()
    
    # Display JSON Data as Table
    st.subheader("Extracted License Plate Data")
    if os.path.exists("json/LicensePlateData.json"):
        with open("json/LicensePlateData.json", "r") as f:
            json_data = json.load(f)
            st.dataframe(json_data)