import cv2
import os
import urllib.request
from tkinter import Tk, filedialog
import zipfile

# ========== 1. CHỌN FILE VIDEO TỪ DIALOG ==========
Tk().withdraw() 
video_path = filedialog.askopenfilename(
    title="Chọn file video khuôn mặt",
    filetypes=[("Video Files", "*.mp4 *.webm *.avi *.mov")]
)
if not video_path:
    raise ValueError("Bạn chưa chọn file video.")


# ========== 2. CẤU HÌNH ==========
model_path = "../models/face_detection_yunet_2023mar.onnx"
name = input("Enter your name: ")
zip_path = f"{name}.zip"
output_dir = f"./datasets/{name}"
os.makedirs(output_dir, exist_ok=True)


# ========== 4. KHỞI TẠO YUNET ==========
yunet = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000
)

# ========== 5. ĐỌC VIDEO ==========
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Không thể mở video: {video_path}")

frame_idx = 0
saved_faces = 0

# ========== 6. XỬ LÝ FRAME ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    yunet.setInputSize((w, h))

    _, faces = yunet.detect(frame)

    if faces is not None:
        for i, face in enumerate(faces):
            x, y, w_box, h_box = face[:4].astype(int)
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w_box, frame.shape[1]), min(y + h_box, frame.shape[0])

            face_crop = frame[y1:y2, x1:x2]
            face_crop_resized = cv2.resize(face_crop, (92, 112))

            out_path = os.path.join(output_dir, f"frame{frame_idx:04d}_face{i}.jpg")
            cv2.imwrite(out_path, face_crop_resized)
            saved_faces += 1

    frame_idx += 1

cap.release()

print(f"\n✅ Đã xử lý {frame_idx} frames và cắt được {saved_faces} khuôn mặt.")

# ========== 7. ĐÓNG GÓI KẾT QUẢ THÀNH ZIP ==========
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, output_dir)
            zipf.write(file_path, arcname)

print(f"📦 Đã nén ảnh khuôn mặt thành: {zip_path}")
print("🟢 Bạn có thể mở thư mục để xem ảnh hoặc gửi file zip.")
