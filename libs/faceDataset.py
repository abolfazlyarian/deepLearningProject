import os
import numpy as np
import glob
import torch.utils.data as data
import cv2
from libs.utils import read_sentiment_text
from torchvision import transforms


class faceDataset(data.Dataset):
    def __init__(self, mode, root_dir=".", transformer=transforms.Compose([]), augmentation=["diffeo","filt","color"]) -> None:
        super().__init__()
        self.transformer=transformer
        self.aug = augmentation
        
        if not os.path.exists(os.path.join(root_dir,"faceDataset","orginalFace",mode)):
            raise Exception("orginalFace Not found!")
        else:
            if len(augmentation) and mode=='train':
                if not os.path.exists(os.path.join(root_dir,"faceDataset","augmentationFace",mode)):
                    raise Exception("augmentationFace Not found!") 
                else:
        
                    self.img_list = []
                    self.img_list = sorted(np.array(glob.glob(os.path.join(root_dir,"faceDataset","orginalFace",mode, '*.jpg'))).tolist())
                    self.sentiment = read_sentiment_text(root_dir+"/Datasets/"+"sentiment_"+mode+".txt")

                    for i in augmentation:
                        if i == 'diffeo':
                            self.img_list.extend(sorted(np.array(glob.glob(os.path.join(root_dir,"faceDataset","augmentationFace",mode, '*_0.jpg'))).tolist()))
                        elif i == 'color':
                            self.img_list.extend(sorted(np.array(glob.glob(os.path.join(root_dir,"faceDataset","augmentationFace",mode, '*_1.jpg'))).tolist()))
                        elif i == 'filt':
                            self.img_list.extend(sorted(np.array(glob.glob(os.path.join(root_dir,"faceDataset","augmentationFace",mode, '*_2.jpg'))).tolist()))
                        else:
                            raise Exception("augmentation is wrong. augmentation = ['diffeo', 'color', 'filt']")
                    
            else:
                self.img_list = sorted(np.array(glob.glob(os.path.join(root_dir,"faceDataset","orginalFace",mode, '*.jpg'))).tolist())
                self.sentiment = read_sentiment_text(root_dir+"/Datasets/"+"sentiment_"+mode+".txt")

    def __getitem__(self, index):
        path=self.img_list[index]
        img=self.transformer(np.array(cv2.imread(path), dtype=np.float32))
        try:
            image_index = int(path[0:-4].rsplit('/')[-1].split("_")[0])
        except :
            image_index = int(path[0:-4].rsplit('\\')[-1].split("_")[0])

        sentiment=self.sentiment[image_index]
        
        return img, int(sentiment)
    
    def __len__(self):
        return len(self.img_list)

