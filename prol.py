# ============================
# AI DRIVER SYSTEM (FINAL FIX)
# ============================

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3
import requests
import os
import winsound
from scipy.spatial import distance

# ================= CONFIG =================
EAR_THRESHOLD = 0.32
MAR_THRESHOLD = 0.30
ALERT_GAP = 4
EMERGENCY_RISK = 90
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ================= VOICE =================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

ai_speaking = False
alert_message = ""
alert_time = 0

def speak(text):
    global ai_speaking, alert_message, alert_time
    ai_speaking = True
    alert_message = text.lower()
    alert_time = time.time()
    engine.say(text)
    engine.runAndWait()
    ai_speaking = False

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

def beep():
    winsound.Beep(1000, 400)

# ================= AI =================
def ask_ai(prompt, state):
    if not OPENAI_KEY:
        return "api key missing"
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "you are a driving safety assistant. be brief."},
                    {"role": "user", "content": f"driver condition: {state}\nquestion: {prompt}"}
                ],
                "max_tokens": 300
            },
            timeout=10
        )
        return r.json()["choices"][0]["message"]["content"].lower()
    except:
        return "ai service unavailable"

# ================= TEXT WRAP =================
def wrap_text(text, width):
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line + " " + w) <= width:
            line += (" " if line else "") + w
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines

# ================= FACE =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 311, 78, 308]

def EAR(eye):
    return (distance.euclidean(eye[1], eye[5]) +
            distance.euclidean(eye[2], eye[4])) / (2 * distance.euclidean(eye[0], eye[3]))

def MAR(mouth):
    return (distance.euclidean(mouth[1], mouth[5]) +
            distance.euclidean(mouth[2], mouth[4])) / (2 * distance.euclidean(mouth[0], mouth[3]))

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cv2.namedWindow("AI DRIVER SYSTEM", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI DRIVER SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

user_text = ""
ai_lines = []

# ================= METRICS =================
start_time = time.time()
eye_c = yawn_c = tilt_c = down_c = 0
last_eye = last_yawn = last_tilt = last_down = 0
emergency_start = None

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    cam_w = ui_w = w // 2

    cam = frame[:, :cam_w]
    ui = np.zeros((h, ui_w, 3), dtype=np.uint8)

    now = time.time()
    state = "normal"

    results = face_mesh.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        le = np.array([(int(lm[i].x * cam_w), int(lm[i].y * h)) for i in LEFT_EYE])
        re = np.array([(int(lm[i].x * cam_w), int(lm[i].y * h)) for i in RIGHT_EYE])
        mo = np.array([(int(lm[i].x * cam_w), int(lm[i].y * h)) for i in MOUTH])

        ear = (EAR(le) + EAR(re)) / 2
        mar = MAR(mo)

        # -------- EYES CLOSED --------
        if ear < EAR_THRESHOLD and now - last_eye > ALERT_GAP:
            eye_c += 1
            state = "eyes closed"
            beep()
            speak_async("warning eyes closed")
            last_eye = now

        # -------- YAWNING --------
        if mar > MAR_THRESHOLD and now - last_yawn > ALERT_GAP:
            yawn_c += 1
            state = "yawning"
            beep()
            speak_async("yawning detected take a break")
            last_yawn = now

        # -------- HEAD TILT --------
        left_eye_y = lm[33].y
        right_eye_y = lm[263].y

        if abs(left_eye_y - right_eye_y) > 0.03 and now - last_tilt > ALERT_GAP:
            tilt_c += 1
            state = "head tilt"
            beep()
            speak_async("warning head tilt detected")
            last_tilt = now

        # -------- HEAD DOWN --------
        eye_center_y = (lm[159].y + lm[386].y) / 2
        nose_y = lm[1].y

        if nose_y - eye_center_y > 0.10 and now - last_down > ALERT_GAP:
            down_c += 1
            state = "head down"
            beep()
            speak_async("warning head down detected")
            last_down = now

    # ================= RISK SCORE =================
    minutes = (now - start_time) / 60
    risk = min(100, int(
        eye_c * 10 +
        yawn_c * 8 +
        tilt_c * 6 +
        down_c * 6 +
        minutes * 2
    ))

    # ================= EMERGENCY MODE =================
    if risk > EMERGENCY_RISK:
        if emergency_start is None:
            emergency_start = now
        elif now - emergency_start > 30:
            beep()
            speak_async("emergency mode activated stop immediately")
            cam[:] = (0, 0, 255)
    else:
        emergency_start = None

    # ================= ALERT BOX =================
    if alert_message and time.time() - alert_time < 3:
        cv2.rectangle(cam, (10, 10), (cam_w - 10, 80), (0, 0, 255), -1)
        cv2.putText(cam, alert_message, (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ================= UI PANEL =================
    cv2.putText(ui, "ai assistant", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(ui, f"risk score: {risk}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if risk > 70 else (0, 255, 0), 2)

    y = 120
    for l in ai_lines[-20:]:
        for line in wrap_text(l, int(ui_w / 11)):
            cv2.putText(ui, line, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

    # ================= INPUT =================
    cv2.rectangle(cam, (0, h - 60), (cam_w, h), (0, 0, 0), -1)
    cv2.putText(cam, "type & press enter:", (20, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(cam, user_text[-60:], (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("AI DRIVER SYSTEM", np.hstack((cam, ui)))

    key = cv2.waitKey(1) & 0xFF

    if key == 13 and user_text.strip():
        reply = ask_ai(user_text, state)
        ai_lines.append(reply)
        speak_async(reply)
        user_text = ""

    elif key == 8:
        user_text = user_text[:-1]

    elif 32 <= key <= 126:
        user_text += chr(key)

    elif key == 27:
        break

# ================= REPORT =================
with open("driver_report.txt", "w") as f:
    f.write(f"drive time: {int(minutes)} minutes\n")
    f.write(f"eye alerts: {eye_c}\n")
    f.write(f"yawns: {yawn_c}\n")
    f.write(f"head tilt: {tilt_c}\n")
    f.write(f"head down: {down_c}\n")
    f.write(f"final risk score: {risk}%\n")

cap.release()
cv2.destroyAllWindows()
