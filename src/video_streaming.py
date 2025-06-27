import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import numpy as np
import uuid
import time
from starlette.websockets import WebSocketState
import threading
# from backend.yolo_processor import YOLOProcessor
from backend.pca_yunet_processor import YuNetProcessor
from backend.attendance_tracker import AttendanceTracker
import os
from contextlib import asynccontextmanager

import pickle
from backend.pca_yunet_processor import YuNetProcessor, PCA

# === Absolute Base Directory ===
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Mount static files
app = FastAPI(lifespan=lambda app: lifespan(app))
app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

# Global flag
global_processor_running = True
public_url = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_processor_running, public_url
    global_processor_running = True
    task = asyncio.create_task(frame_processor_task())
    yield
    global_processor_running = False
    task.cancel()

# Server Config
host = "127.0.0.1"  # Listen on all interfaces for ngrok
port = 8080
max_clients = 4
batch_size = 10
max_queue_size_input = 2
max_queue_size_output = 2

# Paths
# model_path = os.path.join(MODELS_DIR, 'yolov8n.pt')
log_path = os.path.join(LOGS_DIR, 'attendance_log.csv')

# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"YOLO model file not found at {model_path}")

# Init processing classes
client_queues = {}
client_queues_lock = asyncio.Lock()

yunet_processor = YuNetProcessor(model_path=os.path.join(MODELS_DIR, 'face_detection_yunet_2023mar.onnx'))
attendance_tracker = AttendanceTracker(csv_path=log_path)

with open(os.path.join(MODELS_DIR, 'pca_model.pkl'), 'rb') as f:
    pca_model = pickle.load(f)
pca_model.image_size = (92, 112)
pca_model.N = 92 * 112

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ========== ROUTES ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "websocket_clients": len(client_queues),
        "processor_running": global_processor_running
    }

@app.get("/api/attendance", response_class=JSONResponse)
async def get_attendance():
    try:
        attendance_log = attendance_tracker.get_attendance_log()
        return JSONResponse(content=attendance_log)
    except Exception as e:
        print(f"Error retrieving attendance log: {e}")
        return JSONResponse(content=[], status_code=500)

@app.get("/api/config")
async def get_config():
    global public_url
    local_url = f"http://127.0.0.1:{port}"
    return {
        "backendUrl": public_url if public_url else local_url,
        "wsUrl": public_url.replace("http", "ws") + "/ws/webcam_stream" if public_url else f"ws://127.0.0.1:{port}/ws/webcam_stream"
    }

@app.websocket("/ws/webcam_stream")
async def websocket_webcam_stream(websocket: WebSocket):
    session_id = None
    receive_task = None
    send_task = None
    try:
        print(f"[WebSocket] Received connection attempt from {websocket.client.host}")
        async with client_queues_lock:
            if len(client_queues) >= max_clients:
                await websocket.close(code=1013, reason=f"Too many clients: {max_clients}")
                return
        await websocket.accept()
        session_id = str(uuid.uuid4())
        print(f"[WebSocket] Client connected: /ws/webcam_stream, session_id: {session_id}, IP: {websocket.client.host}")

        async with client_queues_lock:
            client_queues[session_id] = {
                "input": asyncio.Queue(maxsize=max_queue_size_input),
                "output": asyncio.Queue(maxsize=max_queue_size_output)
            }

        receive_task = asyncio.create_task(receive_frames(websocket, session_id))
        send_task = asyncio.create_task(send_processed_frames(websocket, session_id))
        await asyncio.gather(receive_task, send_task)

    except WebSocketDisconnect as e:
        print(f"[WebSocket] Client {session_id} disconnected: {e.code} - {e.reason}")
    except Exception as e:
        print(f"[WebSocket] Unexpected error for session {session_id}: {e}")
    finally:
        async with client_queues_lock:
            if session_id in client_queues:
                del client_queues[session_id]
                print(f"[WebSocket] Cleaned up queues for session {session_id}")
        if receive_task:
            receive_task.cancel()
        if send_task:
            send_task.cancel()
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000, reason="Normal closure")
            except Exception as e:
                print(f"[WebSocket] Error closing WebSocket for session {session_id}: {e}")



async def frame_processor_task():
    print("[Processor] Thread started, waiting for frames from clients.")
    process_frame_count = 0
    process_start_time = time.time()

    while global_processor_running:
        frames_to_process = []
        client_session_ids = []

        async with client_queues_lock:
            for session_id in list(client_queues.keys()):
                input_queue = client_queues[session_id]["input"]
                if not input_queue.empty():
                    while input_queue.qsize() > 1:
                        await input_queue.get()

                    try:
                        frame_data_bytes = await asyncio.wait_for(input_queue.get(), timeout=0.01)
                        np_arr = np.frombuffer(frame_data_bytes, np.uint8)
                        frame_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame_decoded is not None:
                            frames_to_process.append(frame_decoded)
                            client_session_ids.append(session_id)
                        else:
                            print(f"[Processor] Could not decode frame for session {session_id}. Skipping.")
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"[Processor] Error getting frame for session {session_id}: {e}")
                        import traceback
                        traceback.print_exc()

        if frames_to_process:
            try:
                processed_frames_batch, detections_batch, face_crops_batch = yunet_processor.process_frames(frames_to_process)

                for idx, (processed_frame, detections, face_crops) in enumerate(zip(processed_frames_batch, detections_batch, face_crops_batch)):
                    session_id = client_session_ids[idx]

                    for det, face_crop in zip(detections, face_crops):
                        try:
                            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                                continue

                            # ✅ Resize face crop về (92, 112)
                            target_width, target_height = pca_model.image_size[1], pca_model.image_size[0]
                            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                            resized_face = cv2.resize(gray_crop, (target_width, target_height)).astype(np.float32)

                            # ✅ Flatten
                            face_vector = resized_face.flatten().reshape(-1, 1)

                            if face_vector.shape != (pca_model.N, 1):
                                print(f"[Warning] Skipped face with invalid shape: {face_vector.shape}, expected: {(pca_model.N, 1)}")
                                continue

                            # ✅ Nhận diện PCA
                            label = pca_model.recognize_face(face_vector, threshold=4000)
                            det["label"] = label  # Gắn label vào detection

                            # ✅ Vẽ bbox và label từ PCA
                            x1, y1, x2, y2 = det["box"]
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(processed_frame, str(label), (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as face_e:
                            print(f"[Processor] Error recognizing face: {face_e}")

                    # ✅ Ghi nhận người dựa vào label PCA
                    attendance_tracker.track_objects(detections, session_id)

                    if session_id in client_queues:
                        output_queue = client_queues[session_id]["output"]

                        if not isinstance(processed_frame, np.ndarray) or processed_frame.size == 0 or len(processed_frame.shape) != 3:
                            print(f"[Processor] Invalid processed frame for session {session_id}, replacing with default.")
                            processed_frame = np.zeros((480, 640, 3), dtype=np.uint8)

                        processed_frame = cv2.UMat(processed_frame)
                        ret_proc, buffer_proc = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                        if ret_proc:
                            try:
                                if output_queue.qsize() >= max_queue_size_output:
                                    output_queue.get_nowait()
                                output_queue.put_nowait(buffer_proc.tobytes())
                            except asyncio.QueueFull:
                                print(f"[Processor] Output queue full for session {session_id}, dropping frame")
                            except Exception as e:
                                print(f"[Processor] Error putting frame to output queue for session {session_id}: {e}")
                        else:
                            print(f"[Processor] Could not encode processed frame for session {session_id}.")

                process_frame_count += len(frames_to_process)
                if time.time() - process_start_time >= 1.0:
                    print(f"[Processor] Processed {process_frame_count} frames in 1s. FPS: {process_frame_count / (time.time() - process_start_time):.2f}")
                    process_frame_count = 0
                    process_start_time = time.time()
            except Exception as e:
                print(f"[Processor] Error processing batch: {e}")
                import traceback
                traceback.print_exc()

        await asyncio.sleep(0.001)




async def receive_frames(websocket: WebSocket, session_id: str):
    while websocket.client_state == WebSocketState.CONNECTED:
        try:
            frame_bytes_from_client = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.05)
            async with client_queues_lock:
                if session_id in client_queues:
                    input_queue = client_queues[session_id]["input"]
                    try:
                        if input_queue.qsize() >= max_queue_size_input:
                            await input_queue.get()
                        input_queue.put_nowait(frame_bytes_from_client)

                    except asyncio.QueueFull:
                        print(f"[Receive Task] Input queue full for session {session_id}, dropping frame")
                    except Exception as e:
                        print(f"[Receive Task] Error putting frame to input queue for session {session_id}: {e}")
                        break
        except asyncio.TimeoutError:
            continue
        except (WebSocketDisconnect, RuntimeError) as e:
            print(f"[Receive Task] Client {session_id} disconnected or runtime error: {e}")
            break
        except Exception as e:
            print(f"[Receive Task] Unexpected error receiving from client {session_id}: {e}")
            import traceback
            traceback.print_exc()
            break

async def send_processed_frames(websocket: WebSocket, session_id: str):
    while websocket.client_state == WebSocketState.CONNECTED:
        try:
            async with client_queues_lock:
                if session_id in client_queues:
                    output_queue = client_queues[session_id]["output"]
                    processed_bytes_to_client = await asyncio.wait_for(output_queue.get(), timeout=0.005)
                    await websocket.send_bytes(processed_bytes_to_client)
                    print(f"[Send Task] Sent processed frame to session {session_id}, queue size: {output_queue.qsize()}")
        except asyncio.TimeoutError:
            continue
        except (WebSocketDisconnect, RuntimeError) as e:
            print(f"[Send Task] Client {session_id} disconnected or runtime error: {e}")
            break
        except Exception as e:
            print(f"[Send Task] Unexpected error sending to client {session_id}: {e}")
            import traceback
            traceback.print_exc()
            break

def run():
    global public_url
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    local_url = f"http://127.0.0.1:{port}"

    try:
        from pyngrok import ngrok
        # public_url = ngrok.connect(port, bind_tls=True).public_url
        # print(f"FastAPI Public URL (ngrok): {public_url}")
    except ImportError:
        print("Ngrok not installed, running FastAPI locally at:", local_url)
    except Exception as e:
        print(f"Ngrok error: {e}, falling back to local URL: {local_url}")

    uvicorn_thread = threading.Thread(target=server.run)
    uvicorn_thread.daemon = True
    uvicorn_thread.start()
    time.sleep(1)
    return public_url, local_url

if __name__ == "__main__":
    print("Starting FastAPI server...")
    public_url, local_url = run()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down FastAPI server...")