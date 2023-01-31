import numpy as np
import torch.utils.data as data
import cv2
import libs.datasetDownloader as downloader
from torchvision.transforms import Compose
from libs.utils import read_data_file, read_index_file, read_sentiment_text, read_text_file
    
class MSCTD(data.Dataset):
    def __init__(self,mode, download=True, root_dir=".",transformer=Compose([]), read_mode='scene'):
        """
            Parameters:
            ------------------------
            `mode` : specifies `train` , `validation` or `test` dataset
            `root_dir` : is the path where the `train/validation/test` data is stored. they should be in ./Datasets/ directory. e.g. for train dataset, you should place train.zip in root_dir/Datasets/
            `download` : downloads the data from the (google drive) if it's (.zip file) not available at `root_dir/Datasets/`.The name of our datasets differs slightly from the original datasets.
            `transformer` : dataset transformation\\
            `read_mode` : 
                `scene`  : maintaining time-series \\ 
                `signle` : wihtout maintaining time-series
        """
        self.transformer = transformer
        self.read_mode = read_mode
        self.root_dir = root_dir
        datasetobj = downloader.driveDownloader(mode=mode,rootDir=root_dir)
        if download:
            if not datasetobj.cacheIsExist:
                datasetobj.fileDownloader()
                datasetobj.unzipFile()
            else:
                datasetobj.unzipFile()
        else:
            if datasetobj.cacheIsExist:
                datasetobj.unzipFile()
            elif datasetobj.fileIsExist:
                pass
            else:
                raise Exception(f'{root_dir}/Datasets/ and datasets in this directory are not existed or their names are wrong! ')
            
        self.img_list=read_data_file(root_dir + "/Datasets/" + mode)
        self.image_index=read_index_file(root_dir+"/Datasets/"+"image_index_"+mode+".txt")
        self.english_text=read_text_file(root_dir+"/Datasets/"+"english_"+mode+".txt")
        self.sentiment=read_sentiment_text(root_dir+"/Datasets/"+"sentiment_"+mode+".txt")
        
    def __getitem__(self, index):
        
        if self.read_mode == "scene":
            img = []
            text = []
            sentiment = []
            img_index = self.image_index[index]
            for i in img_index:
                img.append(self.transformer(np.array( cv2.cvtColor( cv2.imread(self.img_list[i]), cv2.COLOR_BGR2RGB), dtype=np.float32)))
                text.append(self.english_text[i])
                sentiment.append(self.sentiment[i])
            
            return img,text,sentiment,index

        elif self.read_mode == "single":

            img = []
            text = []
            sentiment = []
            
            img.append(self.transformer(np.array( cv2.cvtColor( cv2.imread(self.img_list[index]), cv2.COLOR_BGR2RGB), dtype=np.float32)))
            text.append(self.english_text[index])
            sentiment.append(self.sentiment[index])
            
            return img,text,sentiment,index

    def __len__(self):
        if self.read_mode == "scene":
            return len(self.image_index)
        elif self.read_mode == "single":
            return len(self.img_list)