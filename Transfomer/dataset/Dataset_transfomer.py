import os
import numpy as np
import torch.utils.data as data
import json

class Transfomer_Dataset(data.Dataset):
    def __init__(self,root_dir='.',mode='train') -> None:
        super(Transfomer_Dataset,self).__init__()
        f = open(os.path.join(root_dir,mode+'_dataset.json'))
        self.Dataset = json.load(f)

    def __getitem__(self, index):

        data=self.Dataset[str(index)]
        
        # Normalize the boxes (to 0 ~ 1)
        img_h,img_w=data['size'][0:2]
        boxes = np.array(data['pos'])
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        
        feats = np.array(data['embedding']).copy()
        dialog = data['text']
        sentiment = data['sentiment_cls']

        return boxes,feats,dialog,sentiment
    def __len__(self):
        return len(self.Dataset)