import numpy as np

class PCA:
    def find_k(eigenvalues, quality_percent):
        """
        Tìm số lượng thành phần chính (K) sao cho giữ lại ít nhất quality_percent % tổng năng lượng.

        Args:
            eigenvalues (np.ndarray): Mảng trị riêng đã sắp xếp giảm dần.
            quality_percent (float): Tỷ lệ phần trăm chất lượng cần giữ lại (0-100).

        Returns:
            int: Số lượng thành phần chính cần thiết.
        """
        total_energy = np.sum(eigenvalues)
        threshold = quality_percent / 100 * total_energy
        temp = 0
        K = 0
        while temp < threshold:
            temp += eigenvalues[K]
            K += 1
        return K

    def compute_auto_threshold(self, method='percentile', value=95):
        dists = []
        for label in np.unique(self.y):
            indices = self.label_indices[label]
            mean_vec = self.class_means[label]
            for idx in indices:
                sample = self.new_coordinates[:, idx]
                dist = np.linalg.norm(sample - mean_vec)
                dists.append(dist)
        dists = np.array(dists)
        return np.percentile(dists, value)

    def __init__(self, images, y, target_names, no_of_elements, quality_percent=100):
        """
        images: Mảng 2D (N^2 x M), mỗi cột là một ảnh đã flatten.
        y: Danh sách nhãn ứng với từng ảnh.
        target_names: Tên lớp (ví dụ: tên người).
        no_of_elements: Danh sách số ảnh trong mỗi lớp.
        quality_percent: Tỷ lệ phần trăm chất lượng cần giữ lại (0-100).
        """
        self.no_of_elements = no_of_elements
        self.images = np.asarray(images)
        self.y = y
        self.target_names = target_names
        self.quality_percent = quality_percent

        # 1. Tính ảnh trung bình
        mean = np.mean(self.images, axis=1)  # trung bình theo chiều ngang (M x 1)
        self.mean_face = mean.reshape(-1, 1)

        # 2. Chuẩn hóa dữ liệu bằng cách trừ ảnh trung bình
        self.training_set = self.images - self.mean_face

        # 3. Tính eigenfaces và trị riêng
        self.eigenfaces, self.eigenvalues = self._get_eigenfaces(self.training_set)

        # 4. Chiếu toàn bộ tập huấn luyện lên không gian đặc trưng
        self.new_coordinates = self.get_projected_data()

        # 5. Sau khi có new_coordinates, tính vector trung bình từng lớp
        self._compute_class_means()

        # 6. Tính threshold tự động từ dữ liệu training (nếu muốn)
        self.threshold = self.compute_auto_threshold()

    def _get_eigenfaces(self, input_data):
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

        # Tính số lượng thành phần chính cần thiết
        K = PCA.find_k(eigenvalues, self.quality_percent)

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

    def _compute_class_means(self):
        """
        Tính vector trung bình trong không gian đặc trưng cho mỗi lớp (Ω_k).
        """
        self.class_means = {}  # {label: mean_vector}
        #dictionary rỗng để lưu vector trung bình PCA-space cho từng lớp.
        # Key: label (ví dụ: 0, 1, 2 là mã số các người khác nhau)
        # Value: mean_vector là vector trung bình của các ảnh thuộc lớp đó trong không gian PCA.

        self.label_indices = {}

        # Duyệt qua tất cả các nhãn khác nhau (label) trong tập huấn luyện
        for label in np.unique(self.y):

            # Trả về indices là danh sách vị trí ảnh có nhãn đó
            indices = np.where(np.array(self.y) == label)[0]

            # Lưu danh sách chỉ số ảnh của lớp label vào dictionary label_indices
            self.label_indices[label] = indices

            #Lấy các vector đặc trưng PCA tương ứng với các ảnh thuộc lớp label
            class_vectors = self.new_coordinates[:, indices]

            # Tính vector trung bình trong PCA-space của lớp label
            self.class_means[label] = np.mean(class_vectors, axis=1)
    
    def recognize_face_knn(self, face_vector, k=3, threshold=None):
        """
        Nhận diện khuôn mặt sử dụng ngưỡng + KNN trong không gian PCA.
        """
        if face_vector.shape != self.mean_face.shape:
            raise ValueError("Ảnh không cùng kích thước với ảnh huấn luyện.")
        if threshold is None:
            threshold = self.threshold
            
        phi = face_vector - self.mean_face
        projected = self.eigenfaces.T @ phi
        projected = projected.flatten()

        # Tính khoảng cách đến trung bình từng lớp
        distances_to_class = {
            label: np.linalg.norm(projected - mean_vec) for label, mean_vec in self.class_means.items()
        }

        # Lọc các lớp có khoảng cách < threshold
        valid_labels = [label for label, dist in distances_to_class.items() if dist < threshold]

        if not valid_labels:
            return "Unknown"

        # Lấy tất cả ảnh trong các lớp valid
        candidate_indices = []
        for label in valid_labels:
            candidate_indices.extend(self.label_indices[label])
            # self.label_indices[label] là danh sách các chỉ số ảnh trong lớp label.
            # extend(...) để thêm tất cả các chỉ số ảnh của lớp đó vào danh sách candidate_indices
        
        # Tính khoảng cách đến từng ảnh ứng viên
        candidates = self.new_coordinates[:, candidate_indices].T  
        dists = np.linalg.norm(candidates - projected, axis=1)

        # Lấy k láng giềng gần nhất
        if len(dists) < k:
            # Kiểm tra nếu số lượng ảnh hợp lệ < k, thì giảm k xuống bằng số ảnh hiện có
            k = len(dists)

        #  trả về thứ tự chỉ số sắp xếp của các ảnh theo khoảng cách tăng dần.
        knn_indices = np.argsort(dists)[:k]

        # Lấy nhãn label tương ứng với k ảnh gần nhất.
        knn_labels = [self.y[candidate_indices[i]] for i in knn_indices]

        # Bỏ phiếu
        vote_count = {}
        for label in knn_labels:
            vote_count[label] = vote_count.get(label, 0) + 1

        predicted_label = max(vote_count, key=vote_count.get)
        return self.target_names[predicted_label]