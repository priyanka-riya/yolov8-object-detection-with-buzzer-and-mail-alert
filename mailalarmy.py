import cv2
import torch
import time
import os
import threading
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
from gtts import gTTS
import pygame
from datetime import datetime

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)

# Initialize pygame mixer for sound
pygame.mixer.init()
alert_file = "alert.mp3"

# Function to generate alert sound
def generate_alert_sound():
    alert_text = "Person detected! Alert!"
    alert_tts = gTTS(text=alert_text, lang='en')
    alert_tts.save(alert_file)

# Function to play alert sound
def play_alert():
    pygame.mixer.music.load(alert_file)
    pygame.mixer.music.play()

# Function to send email alert
def send_email_alert():
    sender_email = " "  # Replace with your email
    receiver_email = " "  # Replace with the recipient's email
    password = " "

     # Use an App Password for security

    subject = "Person Detected Alert"
    body = f"Person detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    msg = MIMEText("hello person detected")
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Generate alert sound
generate_alert_sound()

# Timestamp tracking
person_detected_time = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference
    results = model(frame)
    annotated_frame = results[0].plot()

    # Check if a person is detected
    person_detected = any(int(box.cls[0].item()) == 0 for box in results[0].boxes)
    

    if person_detected:
        if person_detected_time is None:  
            person_detected_time = time.time()  # Start 5-second countdown
        elif time.time() - person_detected_time >= 5:
            print(f"Person detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Playing alert and sending email...")
            threading.Thread(target=play_alert).start()
            threading.Thread(target=send_email_alert).start()
            person_detected_time = None  # Reset timer after alert

    else:
        person_detected_time = None  # Reset if no person is detected

    # Display annotated frame
    cv2.imshow("YOLOv8 Person Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Webcam closed successfully.")
