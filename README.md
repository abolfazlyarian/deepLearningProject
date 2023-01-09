        from google.colab import drive
        drive.mount('/content/drive/')

        rootDir = '/content/drive/MyDrive/deepLearningProject-main/*'

        !cp -r $rootDir .

        !pip install --upgrade --no-cache-dir gdown 

        Structure of Code is as follow 

libs    --->

        DatasetMSCTD.py       ----> Dataset class for loading  and preparing data 
    
        config.json           ----> saved some constant information for Downloading
    
        datasetDownloader.py  ----> datasetDownloader class handles downloading and extraction of zip files
    
        transforms.py         ----> transforms include some functions and compose class for adding augmentations and other changes in data
    
phase0.ipynb  ----> 

         main code for phase 0 of project is present here



Structure of Dataset is as follow

Datasets-->

          ----->train
          
          ----->test
          
          ----->validation
          
          ----->english_train.txt
          
          ----->english_test.txt
          
          ----->english_validation.txt
          
          ----->image_index_train.txt
          
          ----->image_index_test.txt
          
          ----->image-index-validation.txt
          
          ----->sentiment_train.txt
          
          ----->sentiment_test.txt
          
          ----->sentiment_validation.txt
          
Note
Our Notebook needs All of the files to be in one directory and the path of Datasets should be set in root_dir as input of the Dataset class.
if a full Dataset exists please insert the path of Datasets as explained beforehand otherwise you can download it easily by our code as you make Dataset instance.
