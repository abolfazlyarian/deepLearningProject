<h1 align="center">Deep Learning 2022 - Project </h1>
<h3 align="center"> Student contributors: Abolfazl Yarian - Mehran Morabbi Pazoki - Farahmand Alizadeh </h3>

The structure of the Project is as follows 
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
├── phase0.ipynb                                ----> main code for phase 0 of the project is present here
├── README.md
├── test_face_count_dict_mtcnn.pkl              -----> face count dictionary for test images 
├── train_face_count_dict_mtcnn.pkl             -----> face count dictionary for train images 
└── validation_face_count_dict_mtcnn.pkl        -----> face count dictionary for validation images 
```       
Our Notebook needs All of the files to be in one directory and the path of Datasets should be set in `root_dir` as input of the Dataset class.
if a full Dataset exists please insert the path of Datasets as explained beforehand otherwise you can download it easily by our code as you make a Dataset instance and set `download=true` in the MSTCD class argument.
for example, if you want to load the training dataset manually, you set `root_dir="trainrootpath"` and `download=False` in loading the dataset with MSTCD class
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
### How to Run Phase3_1_2
in this section, we use Google Colab to train the transformer network and we prepare a notebook for all parts of phase 3-1-2 to run it correctly first, copy the Transformer file in your google drive then open phase3-1-2.ipynb in Colab second move our dataset to your google drive (don't put it in any folder)  this step is enough for training. but for testing the model one step remains to move out the pre-trained model in the transformer/snap in your google drive.

```
Dataset = https://drive.google.com/file/d/1qGVdcPBOznaprIpS_SUHI0SEQTEWLFyf/view?usp=sharing
bestModel = https://drive.google.com/file/d/1-8kwsyct7z2UX0pszBJbzaGM_NH-OrKn/view?usp=sharing
```
