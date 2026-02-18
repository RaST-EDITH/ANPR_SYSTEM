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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit UI
st.title("License Plate Detection and Recognition")
uploaded_file = st.file_uploader("Upload a Video or Image", type=["mp4", "jpg", "png"])

if uploaded_file:
    file_path = f"temp/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("File Uploaded Successfully!")
    
    cap = cv2.VideoCapture(file_path) if uploaded_file.type == "video/mp4" else None
    # if cap :
    #     cap.set(cv2.CAP_PROP_FPS) # Set fps rate if needed
    
    # Initialize the YOLOv10 Model
    model = YOLOv10("weights/best.pt")

    # Class Names
    className = ["License"]
    
    # Initialize the Paddle OCR
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False)
    
    def paddle_ocr(frame, x1, y1, x2, y2):
        frame = frame[y1:y2, x1:x2]
        result = ocr.ocr(frame, det=False, rec=True, cls=False)
        text = ""
        for r in result:
            scores = r[0][1]
            scores = 0 if np.isnan(scores) else int(scores * 100)
            if scores > 60:
                text = r[0][0]
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
    
    startTime = datetime.now()
    license_plates = set()
    count = 0
    
    while cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            currentTime = datetime.now()
            count += 1
            results = model.predict(frame, conf=0.45)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
            
            if (currentTime - startTime).seconds >= 10:
                endTime = currentTime
                save_json(license_plates, startTime, endTime)
                startTime = currentTime
                license_plates.clear()
            
            if cv2.waitKey(1) & 0xFF == ord('1'):
                break
        else:
            break
    
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    # Display JSON Data as Table
    st.subheader("Extracted License Plate Data")
    if os.path.exists("json/LicensePlateData.json"):
        with open("json/LicensePlateData.json", "r") as f:
            json_data = json.load(f)
            st.table(json_data)
