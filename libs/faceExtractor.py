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
        self.mode=mode

        if not os.path.exists(f"{self.root}/FaceDataset"):
          os.mkdir(f"{self.root}/FaceDataset")
        if not os.path.exists(f"{self.root}/FaceDataset/faceImage"):
          os.mkdir(f"{self.root}/FaceDataset/faceImage")
        if not os.path.exists(f"{self.root}/FaceDataset/faceImage/"+mode):
          os.mkdir(f"{self.root}/FaceDataset/faceImage/"+mode)
        # if not os.path.exists(f"{self.root}/FaceDataset/labels"):
        #   os.mkdir(f"{self.root}/FaceDataset/labels")
        self.fault=0
        self.idx_fault=[]
        self.max_len=0
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
    def run(self,Dataset):
        self.app.prepare(ctx_id=0,det_size=(640,960))
        for img,_,sentiment,idx in tqdm(Dataset):
            faces = self.app.get(img[0])
            img_PIL=Image.fromarray(img[0].astype(np.uint8))
            face_count=len(faces)

            if face_count > self.max_len:
              self.max_len = face_count

            if face_count==0:
                self.fault+=1
                self.idx_fault.append(idx)

            else:

              if face_count>6:
                a = 6
              else :
                a = face_count

              for i,f in zip(range(a),faces):
                  face=img_PIL.crop(f['bbox'])
                  face.save(f"{self.root}/FaceDataset/faceImage/{self.mode}/"+f'{idx}_{i}.jpg')
                
