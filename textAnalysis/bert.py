import torch
import os
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel

import os
import numpy as np
import glob
import cv2
from libs.utils import read_sentiment_text,read_text_file
from torchvision import transforms
import torch
from libs.MSCTDdataset import MSCTD



class faceTextDataset(Dataset):
    def __init__(self, mode, root_dir=".", transformer=transforms.Compose([])) -> None:
        """
            face dataset with original face and their augmentation

            Parameters:
            ------------------------
            `mode` : specifies `train` , `validation` or `test` dataset
            `root_dir` : is the path where the `train/validation/test` data is stored. they should be in ./Datasets/ directory. e.g. for train dataset, you should place train.zip in root_dir/Datasets/
            `transformer` : dataset transformation\\
            `augmentation` : which data augmentation ['diffeo', 'color', 'filt']
            `just_aug` : using augmentation or not 
        """
        super().__init__()
        self.transformer=transformer
        
        if not os.path.exists(os.path.join(root_dir,"faceDataset","originalFace",mode)):
            raise Exception("originalFace Not found!")
        else:
                self.img_list = sorted(np.array(glob.glob(os.path.join(root_dir,"faceDataset","originalFace",mode, '*.jpg'))).tolist())
                self.sentiment = read_sentiment_text(root_dir+"/Datasets/"+"sentiment_"+mode+".txt")
                self.english_text=read_text_file(root_dir+"/Datasets/"+"english_"+mode+".txt")


    def __getitem__(self, index):
        path=self.img_list[index]
        img=self.transformer(np.array(cv2.imread(path), dtype=np.float32))
        try:
            image_index = int(path[0:-4].rsplit('/')[-1].split("_")[0])
        except :
            image_index = int(path[0:-4].rsplit('\\')[-1].split("_")[0])

        sentiment=self.sentiment[image_index]
        text = self.english_text[index]
        
        return img, int(sentiment), text
    
    def __len__(self):
        return len(self.img_list)



class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    


class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 3)
        # self.activation = nn.ReLU()
  
    def forward(self, input_ids, attention_mask):
        bert = self.bert(input_ids, attention_mask=attention_mask)[1]
        fc_1 = self.fc1(bert)
        # fc1 = self.activation(fc1)
        fc_2 = self.fc2(fc_1)
        # fc2 = self.activation(fc2)
        fc_3 = self.fc3(fc_2)
        return bert,fc_1,fc_2,fc_3