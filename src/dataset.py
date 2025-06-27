
import os

# Khai báo một class có nhiệm vụ quản lý dữ liệu ảnh 

class DatasetClass:

    def __init__(self, required_no): 
    # required_no: số ảnh cần lấy cho tập huấn luyện mỗi người (mỗi thư mục con tương ứng 1 người)

        self.dir = ("./datasets") # Đường dẫn đến thư mục chứa ảnh
        
        # Khởi tạo các biến để lưu trữ thông tin về ảnh và nhãn
        self.images_name_for_train = [] # Danh sách đường dẫn ảnh train
        self.y_for_train = []  # Danh sách nhãn tương ứng ảnh train
        self.no_of_elements_for_train = [] # là một danh sách, mỗi phần tử là số lượng ảnh train của người tương ứng.

        self.target_name_as_array= [] # Danh sách tên người tương ứng với nhãn
        self.label_to_name_dict = {} # Từ điển ánh xạ từ nhãn → tên người

        self.images_name_for_test = [] # Danh sách đường dẫn ảnh test
        self.y_for_test = [] # Danh sách nhãn tương ứng ảnh test
        self.no_of_elements_for_test = [] # Số ảnh test mỗi người


        person_id = 0 # Biến đếm số người (thư mục con) đã xử lý
        for name in os.listdir(self.dir):
            dir_path = os.path.join(self.dir, name)
            if os.path.isdir(dir_path):
                if len(os.listdir(dir_path)) >= required_no: 
                #Nếu thư mục này chứa ít nhất required_no ảnh, thì tiếp tục xử lý.
                   
                    i = 0 # Biến đếm số ảnh đã xử lý trong thư mục hiện tại
                   
                    for img_name in os.listdir(dir_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):

                        # Duyệt qua từng ảnh trong thư mục đó:
                            img_path = os.path.join(dir_path, img_name)


                            if i < required_no:
                                # Ảnh cho tập train
                                self.images_name_for_train += [img_path]
                                self.y_for_train += [person_id]

                                # Theo dõi số lượng ảnh huấn luyện của từng người
                                if len(self.no_of_elements_for_train) > person_id:
                                    # Nếu danh sách no_of_elements_for_train đã có phần tử cho người person_id
                                    # thì tăng đếm lên +1.

                                    self.no_of_elements_for_train[person_id] += 1 
                                
                                else:
                                    self.no_of_elements_for_train += [1]
                                    # Nếu chưa có → thêm phần tử 1 vào (ảnh đầu tiên của người này).
                                
                                # Lưu tên người tương ứng với nhãn số
                                if i == 0:
                                    self.target_name_as_array += [name]
                                    self.label_to_name_dict[person_id] = name

                            else:
                                # Ảnh cho tập test
                                self.images_name_for_test += [img_path]
                                self.y_for_test += [person_id]

                                # Theo dõi số lượng ảnh test của từng người
                                if len(self.no_of_elements_for_test) > person_id:
                                    self.no_of_elements_for_test[person_id] += 1
                                else:
                                    self.no_of_elements_for_test += [1]

                            i += 1
                        
                    person_id += 1 # Cập nhật chỉ số người


