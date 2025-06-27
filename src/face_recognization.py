import cv2
import time
import numpy as np

# importing algorithms
from algo_pca import PCA

# importing feature extraction classes
from images_to_matrix import ImageMatrixConverter 
from dataset import DatasetClass


# Số ảnh dùng để train cho mỗi người (còn lại sẽ dùng để test)
no_of_images_of_one_person = 20
dataset_obj = DatasetClass(no_of_images_of_one_person)

# Dữ liệu huấn luyện
images_names = dataset_obj.images_name_for_train
y = dataset_obj.y_for_train
no_of_elements = dataset_obj.no_of_elements_for_train
target_names = dataset_obj.target_name_as_array

# Dữ liệu kiểm tra
images_names_for_test = dataset_obj.images_name_for_test
y_for_test = dataset_obj.y_for_test

# Bắt đầu tính thời gian train
training_start_time = time.process_time()

# Kích thước ảnh
img_width, img_height = 92, 112

# Chuyển ảnh sang ma trận
i_t_m_c = ImageMatrixConverter(images_names, img_width, img_height)
scaled_face = i_t_m_c.get_matrix()  # Dạng: (N^2, M)

# Hiển thị ảnh gốc đầu tiên
cv2.imshow("Original Image", cv2.resize(np.reshape(scaled_face[:, 1], (img_height, img_width)).astype(np.uint8), (200, 200)))
cv2.waitKey(0)

# Khởi tạo PCA và huấn luyện
my_algo = PCA(scaled_face, y, target_names, no_of_elements, quality_percent=95)

# Hiển thị một ảnh đã được chiếu lên không gian đặc trưng PCA
# Lấy dữ liệu đã chiếu lên PCA space
projected_data = my_algo.get_projected_data()

# Chiếu ảnh thứ 1 lên không gian đặc trưng rồi tái tạo lại
coord = projected_data[:, 1].reshape(-1, 1)  # Đảm bảo là cột vector (K, 1)
image_reconstructed = my_algo.get_eigenfaces() @ coord + my_algo.mean_face  # (N^2, 1)
image_reconstructed = image_reconstructed.reshape(img_height, img_width).astype(np.uint8)
cv2.imshow("After PCA Image", cv2.resize(image_reconstructed, (200, 200)))
cv2.waitKey(0)


training_time = time.process_time() - training_start_time

# Bắt đầu nhận diện
correct = 0
wrong = 0
net_time_of_reco = 0

for i, img_path in enumerate(images_names_for_test):
    time_start = time.process_time()

    # Đọc và xử lý ảnh test
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img_vector = img.flatten().reshape(-1, 1)
    
    # Nhận diện
    predicted_name = my_algo.recognize_face_knn(img_vector, k=5, threshold=4000)

    time_elapsed = time.process_time() - time_start
    net_time_of_reco += time_elapsed

    real_y = y_for_test[i]
    real_name = target_names[real_y]

    if predicted_name == real_name: 
        correct += 1
        print("Correct - Name:", predicted_name)
    else:
        wrong += 1
        print("Wrong - Real Name:", real_name, "Predicted:", predicted_name)

# In kết quả thống kê
print("\n===== Evaluation Report =====")
print("Correct:", correct)
print("Wrong:", wrong)
print("Total Test Images:", i + 1)
print("Accuracy (%):", round(correct / (i + 1) * 100, 2))
print("Total Persons:", len(target_names))
print("Total Train Images:", no_of_images_of_one_person * len(target_names))
print("Total Time for Recognition:", round(net_time_of_reco, 4), "seconds")
print("Average Time per Recognition:", round(net_time_of_reco / (i + 1), 4), "seconds")
print("Training Time:", round(training_time, 4), "seconds")


# # Xuất model ra file .pkl
import pickle
with open("../models/pca_model.pkl", "wb") as f:
    pickle.dump(my_algo, f)