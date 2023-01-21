import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os
from tqdm import tqdm
from PIL import Image

class faceExtractor():
    def __init__(self,rootDir,mode="train") -> None:
        self.root=rootDir


        if not os.path.exists(f"{self.root}/FaceDataset"):
          os.mkdir(f"{self.root}/FaceDataset")
        if not os.path.exists(f"{self.root}/FaceDataset/faceImage"):
          os.mkdir(f"{self.root}/FaceDataset/faceImage")
        if not os.path.exists(f"{self.root}/FaceDataset/faceImage/"+mode):
          os.mkdir(f"{self.root}/FaceDataset/faceImage"+mode)
        # if not os.path.exists(f"{self.root}/FaceDataset/labels"):
        #   os.mkdir(f"{self.root}/FaceDataset/labels")

        self.fault=0
        self.fault_list=[]
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
    def run(self,Dataset):
        self.app.prepare(ctx_id=0)
        for img,_,sentiment,idx in tqdm(Dataset):
            faces = self.app.get(img[0])
            img_PIL=Image.fromarray(img[0].astype(np.uint8))
            if len(faces)==0:
              self.fault+=1
              self.fault_list.append(idx)
            else :
              if len(faces)>2:
                a=2
              else:
                a=len(faces)

              for i,f in zip(range(a),faces):
                  face=img_PIL.crop(f['bbox'])
                  face.save(f"{self.root}/FaceDataset/faceImage/"+f'{idx}_{i}.jpg')
                

