from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# tải dataset 
dataset_slug = "lvtkhoa/face-detection-dataset"
api.dataset_download_files(dataset_slug, path="datasets/", unzip=True)
