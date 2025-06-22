import cv2
import dlib
import os
import csv
import numpy as np
import face_recognition
from datetime import datetime
import pandas as pd
import getpass
from email.mime.text import MIMEText
import smtplib

# ========== CONFIG ==========
PREDICTOR_PATH = "shape_predictor.dat"
IMAGES_PATH = "images"
ATTENDANCE_FILE = "attendance.csv"
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2
HEAD_TURN_THRESHOLD = 15
# ============================

# Initialize
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
(lStart, lEnd) = (42, 48)  # Right eye
(rStart, rEnd) = (36, 42)  # Left eye

blink_counter = 0
frame_counter = 0
head_turn_detected = False
blink_detected = False
initial_nose_dx = None

# Distance function
def euclidean(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# Load known faces
def load_known_faces():
    known_encodings = []
    known_names = []
    for person in os.listdir(IMAGES_PATH):
        person_dir = os.path.join(IMAGES_PATH, person)
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)
    return known_encodings, known_names

# Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Head turn detection
def get_nose_dx(shape):
    nose = shape.part(30)
    chin = shape.part(8)
    return abs(nose.x - chin.x)

# Mark attendance
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['roll_number', 'Date', 'Time'])

    with open(ATTENDANCE_FILE, 'r') as f:
        lines = f.readlines()
        if any(name in line and date in line for line in lines):
            return

    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])
        print(f"✅ Attendance marked for {name}")

# ====== Main Face Recognition + Liveness =======
known_encodings, known_names = load_known_faces()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)

        leftEye = [shape.part(i) for i in range(lStart, lEnd)]
        rightEye = [shape.part(i) for i in range(rStart, rEnd)]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                blink_counter += 1
                blink_detected = True
            frame_counter = 0

        dx = get_nose_dx(shape)
        if initial_nose_dx is None:
            initial_nose_dx = dx
        elif abs(dx - initial_nose_dx) > HEAD_TURN_THRESHOLD:
            head_turn_detected = True

        if blink_detected and head_turn_detected:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for encoding, location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    match_idx = matches.index(True)
                    name = known_names[match_idx]
                    mark_attendance(name)

                top, right, bottom, left = location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        status = f"Blink: {'✔' if blink_detected else '✖'} | HeadTurn: {'✔' if head_turn_detected else '✖'}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Attendance", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# ====== Post Attendance Processing ======
def process_csv_files():
    try:
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
        new_df = pd.read_csv('new.csv')

        name_column = 'roll_number'
        if name_column not in new_df.columns or name_column not in attendance_df.columns:
            print(f"Error: '{name_column}' column not found.")
            return

        present_df = new_df[new_df[name_column].isin(attendance_df[name_column])]
        absent_df = new_df[~new_df[name_column].isin(attendance_df[name_column])]

        today_date = datetime.today().date()
        os.makedirs('datewise', exist_ok=True)
        os.makedirs('datewise_absent', exist_ok=True)

        present_df.to_csv(f'datewise/{today_date}.csv', index=False)
        absent_df.to_csv(f'datewise_absent/{today_date}.csv', index=False)

        absent_rolls = absent_df[name_column].tolist()
        with open('rolldb_sem.csv', 'a+', newline='') as file:
            csv.writer(file).writerow(absent_rolls)

        print("📁 CSV Processing Done.")
    except Exception as e:
        print(f"❌ Error processing CSVs: {e}")

def send_email(absent_list):
    if 'email' not in absent_list.columns:
        print("Missing 'email' column.")
        return

    HOST = "smtp-mail.outlook.com"
    PORT = 587
    FROM_EMAIL = "sih2024ponder@outlook.com"
    PASSWORD = getpass.getpass("Enter email password: ")

    subject = 'Attendance Alert - SAM Portal'

    try:
        with smtplib.SMTP(HOST, PORT) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(FROM_EMAIL, PASSWORD)

            for email in absent_list['email']:
                body = f"You were marked absent today.\nLecture: Mentor1\n\n- SAM Team"
                msg = MIMEText(body)
                msg['Subject'] = subject
                msg['From'] = FROM_EMAIL
                msg['To'] = email

                smtp.sendmail(FROM_EMAIL, email, msg.as_string())
                print(f"📧 Email sent to {email}")
    except Exception as e:
        print(f"❌ Email error: {e}")

def read_absent_and_send_emails():
    try:
        today_date = datetime.today().date()
        absent_data = pd.read_csv(f'datewise_absent/{today_date}.csv')
        send_email(absent_data)
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

# Call post-processing
process_csv_files()
read_absent_and_send_emails()
