import cv2
import numpy as np
from typing import Union, List, Tuple
import time

# # Yunet model processor
# class YuNetProcessor:
#     """
#     A class to encapsulate the YuNet model loading, configuration,
#     and frame processing logic for real-time face detection.

#     This class applies Object-Oriented Programming (OOP) principles like:
#     - Encapsulation: Bundling data (model, thresholds, input size) and methods
#       that operate on that data within a single unit.
#     - Abstraction: Hiding the complex internal details of model loading and
#       frame processing, exposing only necessary methods for interaction.
#     """

#     def __init__(self, model_path: str = None, initial_confidence_threshold: float = 0.9, 
#                  input_size: Tuple[int, int] = (640, 640)):
#         """
#         Constructor for the YuNetProcessor class.
#         Initializes the YuNet model, sets up the device (CPU by default), and defines initial parameters.

#         Args:
#             model_path (str): Path to the YuNet model weights (e.g., 'face_detection_yunet_2023mar.onnx').
#                             If None, uses the default model from OpenCV.
#             initial_confidence_threshold (float): The default confidence threshold for detections.
#                                                 Detections with confidence below this will be filtered out.
#             input_size (Tuple[int, int]): The target image size (width, height) for model inference.
#                                         YuNet typically uses square or rectangular inputs (e.g., 320x320 or 640x480).
#                                         Frames will be resized to this dimension before being fed to the model.
#         """
#         # Use default model if no path is provided
#         if model_path is None:
#             model_path = "face_detection_yunet_2023mar.onnx"

#         print(f"Loading YuNet model from {model_path}...")
#         # Load the YuNet model using OpenCV's FaceDetectorYN
#         self.model = cv2.FaceDetectorYN.create(
#             model=model_path,
#             config="",
#             input_size=input_size,
#             score_threshold=initial_confidence_threshold,
#             nms_threshold=0.3,
#             top_k=5000,
#             backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
#             target_id=cv2.dnn.DNN_TARGET_CPU
#         )
        
#         # Store the confidence threshold and image size as private attributes
#         self._confidence_threshold = initial_confidence_threshold
#         self._input_size = input_size
        
#         print("YuNet model loaded and initialized.")

#     def _pad_to_divisible(self, image: np.ndarray, stride: int = 32) -> Tuple[np.ndarray, int, int]:
#         """
#         A private helper method to pad an image.
#         Ensures that the image dimensions are divisible by the given stride,
#         which is often a requirement for deep learning models.

#         Args:
#             image (np.ndarray): The input image as a NumPy array (H, W, C).
#             stride (int): The stride value (e.g., 32 for YuNet).

#         Returns:
#             Tuple[np.ndarray, int, int]: A tuple containing:
#                 - padded_image (np.ndarray): The image with added padding.
#                 - pad_h (int): The amount of padding added to the height.
#                 - pad_w (int): The amount of padding added to the width.
#         """
#         h, w = image.shape[:2]
#         new_h = ((h + stride - 1) // stride) * stride
#         new_w = ((w + stride - 1) // stride) * stride
        
#         pad_h = new_h - h
#         pad_w = new_w - w

#         padded_image = cv2.copyMakeBorder(
#             image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
#         )
#         return padded_image, pad_h, pad_w

#     def set_confidence_threshold(self, threshold: float):
#         """
#         Public method to set (update) the confidence threshold for face detection.
#         This allows external components to dynamically change the model's behavior.

#         Args:
#             threshold (float): The new confidence threshold (must be between 0.0 and 1.0).
        
#         Raises:
#             ValueError: If the provided threshold is outside the valid range.
#         """
#         if not (0.0 <= threshold <= 1.0):
#             raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
#         self._confidence_threshold = threshold
#         self.model.setScoreThreshold(threshold)
#         print(f"YuNet confidence threshold updated to: {self._confidence_threshold:.2f}")

#     def get_confidence_threshold(self) -> float:
#         """
#         Public method to get the current confidence threshold.
#         """
#         return self._confidence_threshold

#     def set_input_size(self, input_size: Tuple[int, int]):
#         """
#         Public method to set (update) the target input size for model inference.
#         This can be useful if you want to change the input resolution of the model.

#         Args:
#             input_size (Tuple[int, int]): New target input size (width, height).
        
#         Raises:
#             ValueError: If the input_size is not a valid tuple of positive integers.
#         """
#         if not isinstance(input_size, tuple) or len(input_size) != 2 or not all(isinstance(x, int) and x > 0 for x in input_size):
#             raise ValueError("input_size must be a tuple of two positive integers (width, height).")
#         self._input_size = input_size
#         self.model.setInputSize(input_size)
#         print(f"YuNet inference input size updated to: {self._input_size}")

#     def get_input_size(self) -> Tuple[int, int]:
#         """
#         Public method to get the current inference input size.
#         """
#         return self._input_size

#     def process_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> Union[Tuple[np.ndarray, List[dict]], Tuple[List[np.ndarray], List[List[dict]]]]:
#         """
#         Main public method to process a single frame or a list of frames using the YuNet model.
#         It orchestrates the preprocessing, actual model inference, and post-processing.

#         Args:
#             frames (Union[np.ndarray, List[np.ndarray]]): A single image or a list of images (H, W, C).

#         Returns:
#             Union[Tuple[np.ndarray, List[dict]], Tuple[List[np.ndarray], List[List[dict]]]]: Processed frame(s) and detection(s).
#         """
#         try:
#             total_start_time = time.time()
#             pre_start = time.time()

#             is_batch_input = isinstance(frames, list)
#             frames_list = frames if is_batch_input else [frames]

#             processed_frames = []
#             detections_list = []

#             for frame in frames_list:
#                 if not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) != 3:
#                     print(f"Invalid frame detected: {frame}")
#                     processed_frames.append(np.zeros_like(frame) if isinstance(frame, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8))
#                     detections_list.append([])
#                     continue

#                 original_frame_sizes = (frame.shape[1], frame.shape[0])
#                 if self._input_size and (frame.shape[1] != self._input_size[0] or frame.shape[0] != self._input_size[1]):
#                     frame_resized = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_AREA)
#                 else:
#                     frame_resized = frame

#                 frame_padded, _, _ = self._pad_to_divisible(frame_resized, stride=32)
#                 preprocess_time = time.time() - pre_start

#                 # Inference directly with detect
#                 infer_start = time.time()
#                 faces = self.model.detect(frame_padded)  # Returns (faces, scores) or (None, None) if no detection
#                 inference_time = time.time() - infer_start

#                 post_start = time.time()
#                 output_frame = frame.copy()
#                 detections = []

#                 if faces[1] is not None:  # faces[1] contains the detection results
#                     for face in faces[1]:
#                         x1, y1, w_box, h_box, conf = map(float, face[:5])
#                         x2, y2 = x1 + w_box, y1 + h_box

#                         original_w, original_h = original_frame_sizes
#                         scale_x = original_w / self._input_size[0]
#                         scale_y = original_h / self._input_size[1]
#                         x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

#                         x1, x2 = np.clip([x1, x2], 0, original_w)
#                         y1, y2 = np.clip([y1, y2], 0, original_h)

#                         detections.append({
#                             "class_name": "Face",
#                             "confidence": conf,
#                             "box": [x1, y1, x2, y2]
#                         })

#                         # Crop the face region
#                         face_crop = frame[y1:y2, x1:x2]

#                         # Draw the bounding box and label
#                         cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         label = f"Face {conf:.2f}"
#                         text_y_pos = y1 - 10 if y1 - 10 > 0 else y1 + 20
#                         cv2.putText(output_frame, label, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#                 processed_frames.append(output_frame)
#                 detections_list.append(detections)

#                 postprocess_time = time.time() - post_start
#                 total_time = time.time() - total_start_time
#                 # print(f"[TIME] Pre: {preprocess_time:.3f}s | Infer: {inference_time:.3f}s | Post: {postprocess_time:.3f}s | Total: {total_time:.3f}s")

#             return (processed_frames[0], detections_list[0], face_crops[0]) if not is_batch_input else (processed_frames, detections_list, face_crops)


#         except Exception as e:
#             print(f"Error in YuNetProcessor.process_frames: {e}")
#             import traceback
#             traceback.print_exc()
#             if is_batch_input:
#                 return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frames_list], [[] for _ in frames_list]
#             else:
#                 return np.zeros((480, 640, 3), dtype=np.uint8), []
            

# # PCA model processor
# class PCA:
#     def __init__(self, training_set, labels, num_components, image_size=(92, 112)):

#         if num_components >= len(training_set):
#             raise ValueError("Number of components must be less than number of samples!")

#         # Validate input dimensions
#         expected_size = image_size[0] * image_size[1]

#         self.training_set = training_set
#         self.labels = labels
#         self.num_components = num_components
#         self.image_size = image_size
#         self.N = expected_size  # N^2 = 92 * 112 = 10304

#         # Step 3: Compute mean face (Psi)
#         self.mean_matrix = self._get_mean(training_set)

#         # Steps 4-7: Compute eigenfaces and eigenvalues
#         self.eigenfaces, self.eigenvalues = self._get_eigenfaces(training_set, num_components)

#         # Train KNN classifier
#         self.knn = self._train_knn()

#     def _get_mean(self, input_data):
#         """Step 3: Compute mean face vector (Psi).""" # Tính ảnh trung bình
#         mean = np.mean(input_data, axis=1).reshape(-1,1)
#         return mean

#     def _get_eigenfaces(self, input_data, K):
#         """Steps 4-7: Compute eigenfaces (u_i) and eigenvalues."""
#         M = len(input_data)

#         # Step 4: Compute difference vectors (Phi_i = Gamma_i - Psi)
#         Phi = input_data - self.mean_matrix  # Shape: (N^2, M)

#         # Step 5: Compute covariance matrix using A^T * A
#         A = Phi  # Shape: (N^2, M)
#         ATA = A.T @ A  # Shape: (M, M)

#         # Step 6: Compute eigenvectors and eigenvalues of A^T * A
#         eigenvalues, eigenvectors = np.linalg.eigh(ATA)

#         # Step 6.2: Sort eigenvalues in descending order
#         idx = np.argsort(eigenvalues)[::-1]
#         eigenvalues = eigenvalues[idx]
#         eigenvectors = eigenvectors[:, idx]

#         # Select non-zero eigenvalues
#         non_zero_idx = eigenvalues > 1e-10
#         eigenvalues = eigenvalues[non_zero_idx]
#         eigenvectors = eigenvectors[:, non_zero_idx]

#         # Vector riêng ứng với trị riêng ≈ 0 không mang thông tin (nằm ngoài không gian biến thiên). Nếu không loại bỏ, có thể gây:
#         #Sai khi chiếu dữ liệu.Làm tăng chiều không gian không cần thiết. Gây lỗi hoặc nhiễu trong phân loại.

#         # Kiểm tra xem: bạn có đủ K trị riêng khác 0 để chọn không
#         if len(eigenvalues) < K:
#             raise ValueError(f"Only {len(eigenvalues)} non-zero eigenvalues available, but K={K} requested!")

#         # Compute eigenvectors of C: u_i = A * v_i
#         u = A @ eigenvectors[:, :K]  # Shape: (N^2, K)

#         # Normalize u_i
#         u /= np.linalg.norm(u, axis=0)

#         return u, eigenvalues[:K]

#     def get_eigenfaces(self): # Lấy ra các vecto riêng của không gian khuôn mặt
#         """Return eigenfaces matrix (W)."""
#         return self.eigenfaces

#     def get_eigenvalues(self): # Lấy ra các trị riêng
#         """Return eigenvalues."""
#         return self.eigenvalues

#     def get_mean_matrix(self): # ma trận trung bình ảnh
#         """Return mean face vector."""
#         return self.mean_matrix

#     def get_projected_data(self): # chiếu dữ liệu ảnh gốc lên không gian đặc trưng
#         """Return projected data (w_i = eigenfaces^T * Phi_i)."""

#         Phi = self.training_set - self.mean_matrix
#         return self.eigenfaces.T @ Phi  # Output: (K, M)


#     def _train_knn(self, k=3):
#       """Train KNN classifier on projected data."""
#       projected_data = self.get_projected_data()  # Shape: (K, M)
#       X = projected_data.T  # Chuyển về (M, K)
#       y = np.array(self.labels)
#       knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#       knn.fit(X, y)
#       return knn


#     def recognize_face(self, new_image_vector, k=3, threshold=None):

#         if new_image_vector.shape != (self.N, 1):
#             raise ValueError(f"New image has incorrect shape: {new_image_vector.shape}, expected ({self.N}, 1)")

#         # Step 1: Compute difference vector (Phi_new)
#         phi_new = new_image_vector - self.mean_matrix

#         # Step 2: Project onto eigenfaces (w_new)
#         w_new = self.eigenfaces.T @ phi_new

#         # Step 3: Predict label using KNN
#         X_new = w_new.flatten().reshape(1, -1)
#         predicted_label = self.knn.predict(X_new)[0]

#         # Step 4: Check distance to closest neighbor (Lưu khoảng cách gần nhất khi nhận diện)
#         distances, indices = self.knn.kneighbors(X_new, n_neighbors=1)
#         closest_distance = distances[0][0]

#         #distances: ma trận khoảng cách Euclidean từ X_new đến các láng giềng gần nhất.
#         #indices: chỉ số (index) của các điểm gần nhất trong tập huấn luyện.
#         if threshold is not None and closest_distance > threshold:
#             return "unknown"  # hoặc return None
#         else:
#             return predicted_label
        

import cv2
import numpy as np
import time
from typing import Union, List, Tuple
from sklearn.neighbors import KNeighborsClassifier

# ========================= YuNetProcessor ============================
class YuNetProcessor:
    """
    Class này dùng để load mô hình YuNet, xử lý frame: resize, padding, phát hiện khuôn mặt,
    vẽ bounding box và cắt khuôn mặt ra từ frame.
    """
    def __init__(self, model_path: str = None, initial_confidence_threshold: float = 0.9, 
                 input_size: Tuple[int, int] = (640, 640)):
        if model_path is None:
            model_path = "face_detection_yunet_2023mar.onnx"
        print(f"Loading YuNet model from {model_path}...")
        self.model = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=input_size,
            score_threshold=initial_confidence_threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        self._confidence_threshold = initial_confidence_threshold
        self._input_size = input_size
        print("YuNet model loaded and initialized.")

    def _pad_to_divisible(self, image: np.ndarray, stride: int = 32) -> Tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        new_h = ((h + stride - 1) // stride) * stride
        new_w = ((w + stride - 1) // stride) * stride
        pad_h = new_h - h
        pad_w = new_w - w
        padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_image, pad_h, pad_w

    def set_confidence_threshold(self, threshold: float):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._confidence_threshold = threshold
        self.model.setScoreThreshold(threshold)
        print(f"YuNet confidence threshold updated to: {self._confidence_threshold:.2f}")

    def get_confidence_threshold(self) -> float:
        return self._confidence_threshold

    def set_input_size(self, input_size: Tuple[int, int]):
        if not isinstance(input_size, tuple) or len(input_size) != 2 or not all(isinstance(x, int) and x > 0 for x in input_size):
            raise ValueError("input_size must be a tuple of two positive integers (width, height).")
        self._input_size = input_size
        self.model.setInputSize(input_size)
        print(f"YuNet inference input size updated to: {self._input_size}")

    def get_input_size(self) -> Tuple[int, int]:
        return self._input_size

    def process_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> Union[
            Tuple[np.ndarray, List[dict], List[np.ndarray]], Tuple[List[np.ndarray], List[List[dict]], List[List[np.ndarray]]]]:
        """
        Xử lý khung hình (frame) đơn hoặc danh sách frame:
         - Resize, padding, inference với mô hình YuNet.
         - Vẽ bounding box và trích xuất khuôn mặt (crop).
         - Trả về: frame đã xử lý, danh sách detections và danh sách face crops.
        """
        try:
            total_start_time = time.time()
            pre_start = time.time()

            is_batch_input = isinstance(frames, list)
            frames_list = frames if is_batch_input else [frames]

            processed_frames = []
            detections_list = []
            face_crops_list = []

            for frame in frames_list:
                if not isinstance(frame, np.ndarray) or frame.size == 0 or len(frame.shape) != 3:
                    print(f"Invalid frame detected: {frame}")
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    processed_frames.append(dummy)
                    detections_list.append([])
                    face_crops_list.append([])
                    continue

                original_frame_sizes = (frame.shape[1], frame.shape[0])
                if self._input_size and (frame.shape[1] != self._input_size[0] or frame.shape[0] != self._input_size[1]):
                    frame_resized = cv2.resize(frame, self._input_size, interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame.copy()

                frame_padded, _, _ = self._pad_to_divisible(frame_resized, stride=32)
                preprocess_time = time.time() - pre_start

                infer_start = time.time()
                faces = self.model.detect(frame_padded)  # faces[1] chứa thông tin nhận dạng
                inference_time = time.time() - infer_start

                post_start = time.time()
                output_frame = frame.copy()
                detections = []
                face_crops = []

                if faces[1] is not None:
                    for face in faces[1]:
                        x1, y1, w_box, h_box, conf = map(float, face[:5])
                        x2, y2 = x1 + w_box, y1 + h_box

                        # Scale box về kích thước của frame gốc
                        original_w, original_h = original_frame_sizes
                        scale_x = original_w / self._input_size[0]
                        scale_y = original_h / self._input_size[1]
                        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                        x1, x2 = np.clip([x1, x2], 0, original_w)
                        y1, y2 = np.clip([y1, y2], 0, original_h)

                        detections.append({
                            "class_name": "Face",
                            "confidence": conf,
                            "box": [x1, y1, x2, y2]
                        })

                        # Vẽ bounding box và label xác định mức độ trust
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Face {conf:.2f}"
                        text_y_pos = y1 - 10 if y1 - 10 > 0 else y1 + 20
                        #cv2.putText(output_frame, label, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Cắt khuôn mặt ra từ frame gốc
                        face_crop = frame[y1:y2, x1:x2]
                        face_crops.append(face_crop)

                processed_frames.append(output_frame)
                detections_list.append(detections)
                face_crops_list.append(face_crops)
                postprocess_time = time.time() - post_start
                total_time = time.time() - total_start_time
                # Uncomment dòng dưới để in ra các thông số thời gian xử lý:
                # print(f"[TIME] Pre: {preprocess_time:.3f}s | Infer: {inference_time:.3f}s | Post: {postprocess_time:.3f}s | Total: {total_time:.3f}s")

            if not is_batch_input:
                return processed_frames[0], detections_list[0], face_crops_list[0]
            else:
                return processed_frames, detections_list, face_crops_list

        except Exception as e:
            print(f"Error in YuNetProcessor.process_frames: {e}")
            import traceback
            traceback.print_exc()
            if is_batch_input:
                dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frames_list]
                return dummy_frames, [[] for _ in frames_list], [[] for _ in frames_list]
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8), [], []


# ============================== PCA ================================
class PCA:
    """
    Class PCA để tính toán mô hình eigenfaces cho nhận diện khuôn mặt.
    Sau đó dùng KNN để phân loại dựa trên các thành phần chiếu (projected data).
    """
    def __init__(self, training_set: np.ndarray, labels: List, num_components: int, image_size=(92, 112)):
        if num_components >= training_set.shape[1]:
            raise ValueError("Number of components must be less than number of samples!")
        expected_size = image_size[0] * image_size[1]
        self.training_set = training_set  # Dữ liệu huấn luyện có dạng vector cột: (N^2, M)
        self.labels = labels
        self.num_components = num_components
        self.image_size = image_size
        self.N = expected_size  # Ví dụ: 92*112 = 10304

        # Tính ảnh trung bình (Psi)
        self.mean_matrix = self._get_mean(training_set)
        # Tính eigenfaces và eigenvalues
        self.eigenfaces, self.eigenvalues = self._get_eigenfaces(training_set, num_components)
        # Huấn luyện KNN classifier
        self.knn = self._train_knn()

    def _get_mean(self, input_data: np.ndarray) -> np.ndarray:
        mean = np.mean(input_data, axis=1).reshape(-1,1)
        return mean

    def _get_eigenfaces(self, input_data: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        M = input_data.shape[1]  # Số mẫu
        # Tính các vector hiệu: Phi_i = Gamma_i - Psi
        Phi = input_data - self.mean_matrix  # (N^2, M)
        A = Phi  # (N^2, M)
        ATA = A.T @ A  # (M, M)

        eigenvalues, eigenvectors = np.linalg.eigh(ATA)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Lọc các giá trị riêng khác 0
        non_zero_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[non_zero_idx]
        eigenvectors = eigenvectors[:, non_zero_idx]

        if len(eigenvalues) < K:
            raise ValueError(f"Only {len(eigenvalues)} non-zero eigenvalues available, but K={K} requested!")

        # Tính eigenfaces: u_i = A * v_i
        u = A @ eigenvectors[:, :K]  # (N^2, K)
        # Chuẩn hóa từng vector
        u /= np.linalg.norm(u, axis=0)
        return u, eigenvalues[:K]

    def get_eigenfaces(self) -> np.ndarray:
        return self.eigenfaces

    def get_eigenvalues(self) -> np.ndarray:
        return self.eigenvalues

    def get_mean_matrix(self) -> np.ndarray:
        return self.mean_matrix

    def get_projected_data(self) -> np.ndarray:
        Phi = self.training_set - self.mean_matrix
        return self.eigenfaces.T @ Phi  # (K, M)

    def _train_knn(self, k=3) -> KNeighborsClassifier:
        projected_data = self.get_projected_data()  # (K, M)
        X = projected_data.T  # (M, K)
        y = np.array(self.labels)
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(X, y)
        return knn

    def recognize_face(self, new_image_vector: np.ndarray, threshold: float = None) -> str:
        if new_image_vector.shape != (self.N, 1):
            raise ValueError(f"New image has incorrect shape: {new_image_vector.shape}, expected ({self.N}, 1)")
        phi_new = new_image_vector - self.mean_matrix
        w_new = self.eigenfaces.T @ phi_new
        X_new = w_new.flatten().reshape(1, -1)
        predicted_label = self.knn.predict(X_new)[0]
        distances, _ = self.knn.kneighbors(X_new, n_neighbors=1)
        closest_distance = distances[0][0]
        if threshold is not None and closest_distance > threshold:
            return "unknown"
        else:
            return predicted_label




