import numpy as np
import cv2


class pca_class:
    def __init__(self, images, y, target_names, no_of_elements, num_components):
        """
        images: Mảng 2D (N^2 x M), mỗi cột là một ảnh đã flatten.
        y: Danh sách nhãn ứng với từng ảnh.
        target_names: Tên lớp (ví dụ: tên người).
        no_of_elements: Danh sách số ảnh trong mỗi lớp.
        num_components: Số lượng thành phần chính giữ lại.
        """
        self.no_of_elements = no_of_elements
        self.images = np.asarray(images)
        self.y = y
        self.target_names = target_names


        # 1. Tính ảnh trung bình
        mean = np.mean(self.images, axis=1)  # trung bình theo chiều ngang (M x 1)
        self.mean_face = mean.reshape(-1, 1)

        # 2. Chuẩn hóa dữ liệu bằng cách trừ ảnh trung bình
        self.training_set = self.images - self.mean_face

        # 3. Tính eigenfaces và trị riêng
        self.eigenfaces, self.eigenvalues = self._get_eigenfaces(self.training_set, num_components)

        # 4. Chiếu toàn bộ tập huấn luyện lên không gian đặc trưng
        self.new_coordinates = self.get_projected_data()

    def _get_eigenfaces(self, input_data, K):
        """
        Tính các eigenfaces từ tập ảnh đã chuẩn hóa.
        """
        A = input_data  # A = Phi
        ATA = A.T @ A  # (M x M)

        # Trị riêng và vector riêng
        eigenvalues, eigenvectors = np.linalg.eigh(ATA)

        # Sắp xếp giảm dần
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Lọc trị riêng khác 0
        non_zero_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[non_zero_idx]
        eigenvectors = eigenvectors[:, non_zero_idx]

        if len(eigenvalues) < K:
            raise ValueError(f"Chỉ có {len(eigenvalues)} trị riêng khác 0, nhưng yêu cầu {K} thành phần chính!")
        # u_i = A * v_i
        eigenfaces = A @ eigenvectors[:, :K]
        eigenfaces /= np.linalg.norm(eigenfaces, axis=0)  # chuẩn hóa

        return eigenfaces, eigenvalues[:K]

    def get_eigenfaces(self):
        return self.eigenfaces

    def get_eigenvalues(self):
        return self.eigenvalues

    def get_projected_data(self):
        """
        Chiếu dữ liệu huấn luyện lên không gian đặc trưng (w_i = U^T * (x_i - mean)).
        """
        Phi = self.training_set  # đã trừ mean trước đó
        return self.eigenfaces.T @ Phi  # (K x M)

    def recognize_face_knn(self, face_vector, k=3, threshold=None):
        """
        Nhận diện khuôn mặt bằng thuật toán KNN.

        Parameters:
            face_vector: vector ảnh đầu vào (N x 1), đã flatten.
            k: số lượng lân cận gần nhất.
            threshold: ngưỡng khoảng cách tùy chọn (nếu muốn loại bỏ ảnh quá khác biệt).

        Returns:
            Tên người được nhận diện hoặc 'Unknown'

        Ta sẽ:
        Chiếu ảnh mới vào không gian đặc trưng PCA như trước.
        Tính khoảng cách Euclidean giữa vector mới với tất cả vector huấn luyện (không chỉ trung bình lớp).
        Lấy k điểm gần nhất.
        Bỏ phiếu (majority voting) theo nhãn để xác định lớp.
        """
        if face_vector.shape != self.mean_face.shape:
            raise ValueError(f"Kích thước ảnh không khớp: {face_vector.shape} != {self.mean_face.shape}")

        # Chuẩn hóa và chiếu vào không gian đặc trưng
        phi = face_vector - self.mean_face
        projected_vector = self.eigenfaces.T @ phi  # (K x 1)

        # Tính khoảng cách đến toàn bộ tập huấn luyện
        projected_vector = projected_vector.flatten()  # (K, )
        all_vectors = self.new_coordinates.T  # (M x K)
        distances = np.linalg.norm(all_vectors - projected_vector, axis=1)  # (M, )

        # Lấy k vector gần nhất
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [self.y[i] for i in nearest_indices]

        # Bỏ phiếu theo nhãn
        votes = {}
        for label in nearest_labels:
            votes[label] = votes.get(label, 0) + 1

        # Tìm nhãn được bình chọn nhiều nhất
        predicted_label = max(votes, key=votes.get)

        # Nếu dùng threshold để kiểm tra khoảng cách gần nhất
        if threshold is not None:
            min_distance = distances[nearest_indices[0]]
            if min_distance > threshold:
                return "Unknown"

        return self.target_names[predicted_label]
