<h1 align="center">Deep Learning 2022 - Project </h1>
<h3 align="center">student contributors : Abolfazl Yarian - Mehran Morabbi Pazoki - Farhmand Alizadeh </h3>

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
|       ├── *.jpg
|   ├── train                   ----> train image directory  
|       ├── *.jpg
|   └── validation              ----> validation image directory
|       ├── *.jpg
|
|
├── libs
│   ├── config.json             ----> saved some constant information for Downloading
│   ├── datasetDownloader.py    ----> datasetDownloader class handles downloading and extraction of zip files
│   ├── MSCTDdataset.py         ----> Dataset class for loading  and preparing data 
│   └── transforms.py           ----> transforms include some functions and compose class for adding augmentations and other changes in data
|
|
├── phase0.ipynb                                ----> main code for phase 0 of project is present here
├── README.md
├── test_face_count_dict_mtcnn.pkl              -----> face count dictionary for test images 
├── train_face_count_dict_mtcnn.pkl             -----> face count dictionary for train images 
└── validation_face_count_dict_mtcnn.pkl        -----> face count dictionary for validation images 
```       
Our Notebook needs All of the files to be in one directory and the path of Datasets should be set in `root_dir` as input of the Dataset class.
if a full Dataset exists please insert the path of Datasets as explained beforehand otherwise you can download it easily by our code as you make Dataset instance and set `download=true` in MSTCD class argument.
for example if you want to load training dataset manually, you set `root_dir="trainrootpath"` and `download=False` in loading dataset with MSTCD class
```
trainrootpath
├── Datasets
        ├── english_train.txt
        ├── image_index_train.txt
        ├── sentiment_train.txt
        └── train
                ├── *.jpg
```
--------------------
### Run on google colab
If you want to use google colab to test our codes, first you upload our project's code in your drive as `deepLearningProject` then open phase0.ipynb and add the below commands to the top of it. they copy requirement files(libs files and .pkl data) to "/content" directory in colab. you notice that you have to download Datasets, so you set `root_dir="/content"` and `download=True` in loading dataset with MSTCD class
```
from google.colab import drive
drive.mount('/content/drive/')

rootDir = '/content/drive/MyDrive/deepLearningProject/*'
!cp -r $rootDir .

!pip install --upgrade --no-cache-dir gdown 
```       
### How Run Phase3_1_2
in this section we use Google colab to train transfomer network and we prepare a notebook for all part of in phase 3-1-2
to run it correctly first copy Transformer file in your googel drive then open phase3-1-2.ipynb in colab second move our dataset to your google drive (dont put it in any floder)  these step is enough for training.but for testing model one step is remain move out pretrained model in transformer/snap in your google drive.

```
Dataset = https://drive.google.com/file/d/1qGVdcPBOznaprIpS_SUHI0SEQTEWLFyf/view?usp=sharing
bestModel = https://drive.google.com/file/d/1-8kwsyct7z2UX0pszBJbzaGM_NH-OrKn/view?usp=sharing
```
