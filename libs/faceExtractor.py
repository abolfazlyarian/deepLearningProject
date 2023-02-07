import numpy as np
from insightface.app import FaceAnalysis
import os
import json
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from libs.PRIME.diffeomorphism import Diffeo
from libs.PRIME.rand_filter import RandomFilter
from libs.PRIME.color_jitter import RandomSmoothColor

class faceExtractor():
    def __init__(self,
                 mode,
                 rootDir=".",
                 max_num=1,
                 transform_input=transforms.Compose([transforms.ToPILImage()]),
                 transform_original=transforms.Compose([]),
                 transform_aug=transforms.Compose([transforms.PILToTensor(),transforms.Resize((224,224))]),
                 augmentation=False) -> None:
        
        """
          Parameters:
          ----------------------------
          `mode` : specifies `train` , `validation` or `test` dataset
          `root_dir` : face saving path
          `max_num` : find maximum number of face in each image
          `augmentation` : data augmentation ['diffeo', 'color', 'filt']
          `transform` : apply transforms
        """
        
        self.root=rootDir
        self.mode=mode
        
        if not os.path.exists(os.path.join(self.root, "faceDataset")):
          os.mkdir(os.path.join(self.root, "faceDataset"))
        if not os.path.exists(os.path.join(self.root, "faceDataset","originalFace")):
          os.mkdir(os.path.join(self.root, "faceDataset","originalFace"))
        if not os.path.exists(os.path.join(self.root, "faceDataset","originalFace",mode)):
          os.mkdir(os.path.join(self.root, "faceDataset","originalFace",mode))
        
        if augmentation and not os.path.exists(os.path.join(self.root, "faceDataset","augmentationFace")):
          os.mkdir(os.path.join(self.root, "faceDataset","augmentationFace"))
        if augmentation and not os.path.exists(os.path.join(self.root, "faceDataset","augmentationFace", mode)):
          os.mkdir(os.path.join(self.root, "faceDataset","augmentationFace", mode))

        
        self.fault=0
        self.idx_fault=[]
        self.max_len=0
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
        self.tf_input = transform_input
        self.tf_original = transform_original
        self.tf_aug =transform_aug
        self.aug=augmentation
        self.max_num = max_num
        
        if augmentation:
            with open('libs/PRIME/configPRIME.json') as f:
                self.configPRIME = json.load(f)

            par_diffeo = self.configPRIME["diffeo"]
            par_color = self.configPRIME["color_jit"]
            par_rand = self.configPRIME["rand_filter"]

            self.diffeo = Diffeo(sT=par_diffeo["sT"], rT=par_diffeo["rT"],
                            scut=par_diffeo["scut"], rcut=par_diffeo["rcut"],
                            cutmin=par_diffeo["cutmin"], cutmax=par_diffeo["cutmax"],
                            alpha=par_diffeo["alpha"], stochastic=True)

            self.color = RandomSmoothColor(cut=par_color["cut"], T=par_color["T"],
                                      freq_bandwidth=par_color["max_freqs"], stochastic=True)

            self.filt = RandomFilter(kernel_size=par_rand["kernel_size"],
                                sigma=par_rand["sigma"], stochastic=True)

    def run(self,Dataset):
        self.app.prepare(ctx_id=0)#det_size=(Dataset[0][0][0].size(1),Dataset[0][0][0].size(2))
        for img_tensor,_,sentiment,idx in tqdm(Dataset):
            img = img_tensor[0]#np.asarray(self.tf_input(img_tensor[0]))
            faces = self.app.get(img)
            img_PIL = Image.fromarray(img.astype(np.uint8))
            face_count = len(faces)

            if face_count > self.max_len:
                self.max_len = face_count

            if face_count == 0:
                self.fault += 1
                self.idx_fault.append(idx)

            else:

              if face_count>self.max_num:
                a = self.max_num
              else :
                a = face_count

              for i,f in zip(range(a),faces):
                  try: 
                    cons = np.linalg.norm(f['bbox'][0:2] - f['bbox'][2:4])//10
                    f['bbox'] += np.array([-cons,-cons,cons,cons])
                  except:
                     pass
                  face=img_PIL.crop(f['bbox'])
                  face_tf_org = self.tf_original(face)
                  face_tf_org.save(f"{self.root}/faceDataset/originalFace/{self.mode}/"+f'{idx}_{i}.jpg')
                  if self.aug:
                    tensor2PIL = transforms.ToPILImage()
                    face_tf = self.tf_aug(face)/255
                    # diffeo augmentation
                    aug_diffeo = self.diffeo(face_tf)
                    tensor2PIL(aug_diffeo).save(f"{self.root}/faceDataset/augmentationFace/{self.mode}/"+f'{idx}_{i}_0.jpg')
                    # color augmentation
                    aug_color = self.color(face_tf)
                    tensor2PIL(aug_color).save(f"{self.root}/faceDataset/augmentationFace/{self.mode}/"+f'{idx}_{i}_1.jpg')
                    # filter augmentation
                    aug_filt = self.filt(face_tf)
                    tensor2PIL(aug_filt).save(f"{self.root}/faceDataset/augmentationFace/{self.mode}/"+f'{idx}_{i}_2.jpg')

                     
                     
                     

                     
                
