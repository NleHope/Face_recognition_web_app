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

    def recognize_face(self, new_cord_pca, k=0):
        """
        Dự đoán người dựa trên vector PCA đầu vào bằng khoảng cách Euclidean với trung bình lớp.
        """
        classes = len(self.no_of_elements)
        start = 0
        distances = []

        for i in range(classes):
            class_vectors = self.new_coordinates[:, start:start + self.no_of_elements[i]]
            class_mean = np.mean(class_vectors, axis=1)
            dist = np.linalg.norm(new_cord_pca - class_mean)
            distances.append(dist)
            start += self.no_of_elements[i]

        min_index = np.argmin(distances)
        threshold = 100000

        if distances[min_index] < threshold:
            print(f"Person {k} : {min_index} - {self.target_names[min_index]}")
            return self.target_names[min_index]
        else:
            print(f"Person {k} : {min_index} - Unknown")
            return "Unknown"
