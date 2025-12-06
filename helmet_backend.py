from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import time
import threading

PERSON_MODEL_PATH = "yolov8n.pt" 
HELMET_MODEL_PATH = "Helmet.v9-helmet9.yolov11/best.pt" 

CAMERA_INDEX = 0
CONF_THRESHOLD = 0.5
HELMET_CLASS_NAMES = ["helmet", "no_helmet"] 


print("[INFO] Loading person detection model...")
person_model = YOLO(PERSON_MODEL_PATH)
print("[INFO] Loading helmet detection model...")
helmet_model = YOLO(HELMET_MODEL_PATH)
print("[INFO] Models loaded successfully!")


app = FastAPI()
frame_lock = threading.Lock()
output_frame = None
latest_alerts = []  
alert_history = []  


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def box_iou(box1, box2):
    """Calculate Intersection over Union of two boxes.
    Boxes are [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def camera_loop():
    global output_frame, latest_alerts, alert_history

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

      
        person_results = person_model(frame, verbose=False)
        person_dets = person_results[0].boxes

       
        helmet_results = helmet_model(frame, verbose=False)
        helmet_dets = helmet_results[0].boxes

        persons = []
        helmets = []

       
        for box in person_dets:
            cls_id = int(box.cls)
            label = person_model.names[cls_id]
            conf = float(box.conf)
            if label == "person" and conf > CONF_THRESHOLD:
                persons.append({
                    "box": box.xyxy[0].cpu().numpy().tolist(),  # [x1,y1,x2,y2]
                    "conf": conf
                })

    
        for box in helmet_dets:
            cls_id = int(box.cls)
            label = helmet_model.names[cls_id]
            conf = float(box.conf)
            if label in HELMET_CLASS_NAMES and conf > CONF_THRESHOLD:
                helmets.append({
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "label": label,
                    "conf": conf
                })

     
        alerts = []
        all_wearing_helmet = True
        for person in persons:
            px1, py1, px2, py2 = person["box"]
            person_box = [px1, py1, px2, py2]
            helmet_found = False

            for h in helmets:
                hx1, hy1, hx2, hy2 = h["box"]
                helmet_box = [hx1, hy1, hx2, hy2]
                iou = box_iou(person_box, helmet_box)
                if iou > 0.3 and h["label"] == "helmet":  
                    helmet_found = True
                    break
              
                if iou > 0.3 and h["label"] == "no_helmet":
                    helmet_found = False
                    break

            if not helmet_found:
                all_wearing_helmet = False

            alerts.append({
                "person_box": [int(px1), int(py1), int(px2), int(py2)],
                "helmet_detected": helmet_found,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            
            color = (0, 255, 0) if helmet_found else (0, 0, 255)
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), color, 2)
            label_text = "Helmet" if helmet_found else "No Helmet"
            cv2.putText(frame, label_text, (int(px1), int(py1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

     
        for h in helmets:
            hx1, hy1, hx2, hy2 = map(int, h["box"])
            label = h["label"]
            color = (0, 255, 255) if label == "helmet" else (0, 0, 255)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 1)
            cv2.putText(frame, f"{label} {h['conf']:.2f}", (hx1, hy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        helmet_detected = all_wearing_helmet if len(persons) > 0 else None

        with frame_lock:
            output_frame = frame.copy()
            latest_alerts = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "helmet_detected": helmet_detected,
                "alerts": alerts
            }
            alert_history.append(latest_alerts)
            if len(alert_history) > 20:  # keep last 20
                alert_history.pop(0)

        time.sleep(0.03)  # ~30 FPS


threading.Thread(target=camera_loop, daemon=True).start()

# -----------------------
# API ROUTES
# -----------------------

@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if output_frame is None:
                    continue
                ret, buffer = cv2.imencode(".jpg", output_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/alerts")
def get_alerts():
    with frame_lock:
        return JSONResponse(content=latest_alerts)

@app.get("/alerts/history")
def get_alert_history():
    with frame_lock:
        return JSONResponse(content=alert_history)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
