import os
import scipy.io
import numpy as np
import glob
import torch.utils.data as data
from torch.utils.data import default_collate
import scipy.misc
from PIL import Image
import cv2
# import Mytransforms
from math import ceil
def read_data_file(root_dir):
    """get train or val images
        return: image list: train or val images list
    """
    image_arr = np.array(glob.glob(os.path.join(root_dir, '*.jpg')))
    image_nums_arr = np.array([int(s.rsplit('/')[-1][0:-4]) for s in image_arr])
    sorted_image_arr = image_arr[np.argsort(image_nums_arr)]
    return sorted_image_arr.tolist()
def read_index_file(root_dir):
    f = open(root_dir, "r")
    L=f.read().splitlines()
    f.close()
    return list(map(lambda x:list(map(int, x[1:-1].split(','))),L))
def read_text_file(root_dir):
    f = open(root_dir, "r")
    L= f.read().splitlines()
    f.close()
    return L
def read_sentiment_text(root_dir):
    f = open(root_dir, "r")
    L=f.read().splitlines()
    f.close()
    return L

class MSCTD(data.Dataset):
    def __init__(self,mode='train', root_dir="/content", transformer=None,read_mode='scene'):
        """
            mode : Enter type of dataset train , dev , test
            root_dir : Just Enter path of Dataset files 
            transformer : expand dataset.
            read_mode : scene or signle
        """
        self.transformer = transformer
        self.read_mode=read_mode
        self.root_dir = root_dir
        self.imgsize=(400,400)
        if mode=="train":
          self.img_list=read_data_file(root_dir+"//"+"train_ende")
        else :
          self.img_list=read_data_file(root_dir+"//"+mode)

        self.image_index=read_index_file(root_dir+"//"+"image_index_"+mode+".txt")
        self.english_text=read_text_file(root_dir+"//"+"english_"+mode+".txt")
        self.sentiment=read_sentiment_text(root_dir+"//"+"sentiment_"+mode+".txt")
        
    def __getitem__(self, index):
        
        if self.read_mode == "scene":
            img = []
            text = []
            sentiment = []
            img_index = self.image_index[index]
            for i in img_index:
                img.append(self.transformer(np.array(cv2.resize(cv2.imread(self.img_list[i]),dsize=self.imgsize), dtype=np.float32)))
                text.append(self.english_text[i])
                sentiment.append(self.sentiment[i])
            
            return img,text,sentiment,index
            # return tuple([img,torch.tensor(list(map(int,sentiment)))])
        elif self.read_mode == "single":

            img = []
            text = []
            sentiment = []
            
            img.append(self.transformer(np.array(cv2.resize(cv2.imread(self.img_list[index]),dsize=self.imgsize), dtype=np.float32)))
            text.append(self.english_text[index])
            sentiment.append(self.sentiment[index])
            
            return img,text,sentiment,index
            # return tuple([img,torch.tensor(list(map(int,sentiment)))])
    def __len__(self):
        if self.read_mode == "scene":
            return len(self.image_index)
        elif self.read_mode == "single":
            return len(self.img_list)
        