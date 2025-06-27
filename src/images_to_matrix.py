import cv2
import numpy as np

# Dùng để chuyển danh sách các ảnh xám (grayscale) thành ma trận đặc trưng 2D 
# mỗi cột là một ảnh được flatten (ravel) thành vector
class ImageMatrixConverter:

    def __init__(self, images_name, img_width, img_height):

        self.images_name = images_name # Danh sách đường dẫn ảnh
        self.img_width = img_width # Chiều rộng ảnh resize về
        self.img_height = img_height# Chiều cao ảnh resize về
        self.img_size = (img_width * img_height) # Số pixel ảnh sau khi resize



    def get_matrix(self):

        col = len(self.images_name)  # Số ảnh → số cột của ma trận
        img_mat = np.zeros((self.img_size, col))  # Ma trận đặc trưng 2D, mỗi cột là một ảnh

        i = 0
        for name in self.images_name: # Duyệt qua từng ảnh
            gray = cv2.imread(name, 0)   # Đọc ảnh ở chế độ grayscale
            gray = cv2.imread(name, 0)
            if gray is None:
                raise FileNotFoundError(f"Không đọc được ảnh: {name}")

            gray = cv2.resize(gray, (self.img_width, self.img_height))  # Resize ảnh
            img_mat[:, i] = gray.flatten()  # Chuyển ảnh thành vector và lưu vào cột thứ i của ma trận
            i += 1
        return img_mat

