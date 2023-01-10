        from google.colab import drive
        drive.mount('/content/drive/')

        rootDir = '/content/drive/MyDrive/deepLearningProject-main/*'

        !cp -r $rootDir .

        !pip install --upgrade --no-cache-dir gdown 
        
Structure of Project is as follow 
```
deepLearningProject
├── Datasets
|   ├── english_test.txt
|   ├── english_train.txt
|   ├── english_validation.txt
|   ├── image_index_test.txt
|   ├── image_index_train.txt
|   ├── image_index_validation.txt
|   ├── sentiment_test.txt
|   ├── sentiment_train.txt
|   ├── sentiment_validation.txt
|   ├── test                    ----> test image directory
|   ├── train                   ----> train image directory                                             
|   └── validation              ----> validation image directory
|
|
├── libs
│   ├── config.json             ----> saved some constant information for Downloading
│   ├── datasetDownloader.py    ----> datasetDownloader class handles downloading and extraction of zip files
│   ├── MSCTDdataset.py         ----> Dataset class for loading  and preparing data 
│   └── transforms.py           ----> transforms include some functions and compose class for adding augmentations and other changes in data
|
|
├── phase0.ipynb                           ----> main code for phase 0 of project is present here
├── README.md
├── test_face_count_dict_mtcnn.pkl
├── train_face_count_dict_mtcnn.pkl
└── validation_face_count_dict_mtcnn.pkl
```       


Structure of Dataset is as follow
```
Datasets
├── english_test.txt
├── english_train.txt
├── english_validation.txt
├── image_index_test.txt
├── image_index_train.txt
├── image_index_validation.txt
├── sentiment_test.txt
├── sentiment_train.txt
├── sentiment_validation.txt
├── test
├── train
└── validation

3 directories, 9 files
```
          
Note
Our Notebook needs All of the files to be in one directory and the path of Datasets should be set in root_dir as input of the Dataset class.
if a full Dataset exists please insert the path of Datasets as explained beforehand otherwise you can download it easily by our code as you make Dataset instance.
