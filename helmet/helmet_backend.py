
# YOLOv3-tiny backend using OpenCV DNN
import time
import threading
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import uvicorn


# ---------------------------
# Config
# ---------------------------
MODEL_CFG = "./yolov3-tiny.cfg"
MODEL_WEIGHTS = "./yolov3-tiny_final.weights"
NAMES_PATH = "./hardhat.names"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4


with open(NAMES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Load YOLOv3-tiny model
net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

cap = cv2.VideoCapture(CAMERA_INDEX)
frame_lock = threading.Lock()
latest_frame = None
latest_detections = []

print(f"[INFO] Model loaded: {MODEL_WEIGHTS}")
print(f"[INFO] Classes: {CLASS_NAMES}")

# ---------------------------
# Detection thread
# ---------------------------
def detection_loop():
    global latest_frame, latest_detections
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONF_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


        detections = []
        # Ensure indices is always defined and handle all possible return types
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if indices is not None and len(indices) > 0:
            # indices can be a tuple, list, or np.ndarray
            if isinstance(indices, tuple):
                indices = list(indices)
            for i in indices:
                idx = int(i)
                x, y, w, h = boxes[idx]
                label = CLASS_NAMES[class_ids[idx]]
                conf = confidences[idx]
                
                if label == "white" and conf < 0.99:
                    label = "none"
                
                if label in ["blue", "yellow", "white", "red"]:
                    display_label = f"{label} helmet"
                else:
                    display_label = label
                det = {
                    "box": (x, y, x + w, y + h),
                    "label": display_label,
                    "conf": round(conf, 2),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
              
                if label in ["blue", "yellow", "white", "red"]:
                    det["helmet_detected"] = True
                    color = (0, 255, 0)
                elif label == "none":
                    det["helmet_detected"] = False
                    color = (0, 0, 255)
                else:
                    det["helmet_detected"] = None
                    color = (255, 255, 0)
                detections.append(det)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{display_label} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        with frame_lock:
            latest_frame = frame.copy()
            latest_detections = detections

# Start background detection thread
threading.Thread(target=detection_loop, daemon=True).start()

# ---------------------------
# API routes
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/video")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                _, jpeg = cv2.imencode(".jpg", latest_frame)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpeg.tobytes() + b"\r\n")
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detections")
def get_detections():
    with frame_lock:
        return JSONResponse(content=latest_detections)

# ---------------------------
# Run directly with python
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
