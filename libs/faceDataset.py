import os
import numpy as np
import glob
import torch.utils.data as data
import cv2
from libs.utils import read_sentiment_text
from torchvision import transforms
import torch
from libs.MSCTDdataset import MSCTD


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

#TODO augmentation input ....
class faceNetwrokDataset(data.Dataset):
    def __init__(self,mode, root_dir='.', transformer=transforms.Compose([])) -> None:
        super(faceNetwrokDataset,self).__init__()
        
        self.transformer=transformer
        self.facePath=os.path.join(root_dir,'FaceDataset','faceImage',mode) 
        self.main_Dataset=MSCTD(mode=mode,download=False,root_dir=root_dir,transformer=transforms.Compose([]),read_mode='single')
    
    def __getitem__(self, index):
        face_paths=glob.glob(os.path.join(self.facePath,f"{index}_*.jpg"))
        _,_,sentiment,_=self.main_Dataset[index]
        x=[]
        if len(face_paths):
            for i in face_paths[0:6]:
                x.append(self.transformer(cv2.imread(i))[None,:])
            return torch.concat(x),torch.tensor(np.array(sentiment,dtype=int)),torch.tensor(np.array([index for i in range(len(face_paths[0:6]))],dtype=int))
        else : # np.array(sentiment,dtype=int)  np.array([index for i in range(len(face_paths))
            return torch.tensor([]),torch.tensor(np.array(sentiment,dtype=int)),torch.tensor(np.array([index for i in range(len(face_paths[0:6]))],dtype=int))
    
    def __len__(self):
        return self.main_Dataset.__len__()