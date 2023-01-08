from google.colab import drive
drive.mount('/content/drive/')

rootDir = '/content/drive/MyDrive/deepLearningProject-main/*'

!cp -r $rootDir .

!pip install --upgrade --no-cache-dir gdown 
