from faceAnalysis.networks.dan import DAN
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn
from collections import OrderedDict
from faceAnalysis.networks.MixFace import MixFaceMLP
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

class mergeNetwork():
    def __init__(self, max_num: int=1, modelPath_face: str='.', modelPath_image: str='.', num_head: int=4) -> None:
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
        self.max_num = max_num

        self.face_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
                                                    

        self.image_transformer = transforms.Compose([transforms.ToTensor(), 
                                                     transforms.Resize((224,224))])

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pip_model = torch.load(modelPath_face)
        self.dan_model = DAN(num_head=num_head,num_class=3,pretrained=False)
        # dan_model.load_state_dict(pip_model['model_state_dict'])
        self.dan_model.requires_grad_(False)
        self.dan_model.to(self.device)

        self.Net_Total = MixFaceMLP(dim=6)
        self.Net_Total.load_state_dict(pip_model['Net_Total'])
        self.Net_Total.requires_grad_(False)
        self.Net_Total.to(self.device)
        

        # self.model_face = nn.Sequential(OrderedDict([
        #     ('DAN',dan_model),
        #     ('MixFaceMLP', Net_Total)
        # ]))

        # model_image = torch.load(modelPath_image)
        self.resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        fc_head = nn.Sequential(
            nn.Linear(2048, 256),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(256, 64),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(64, 16),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(16, 3),
        )
        self.resnet_model.fc = fc_head
        # self.resnet_model.load_state_dict(model_image)
        self.resnet_model.requires_grad_(False)
        self.resnet_model.to(self.device)
        self.app.prepare(ctx_id=0)

        self.prediction_model = nn.Linear(6,3).to(self.device)

    def faceExractor(self, img_tensor):
        transformer = transforms.PILToTensor()
        #det_size=(Dataset[0][0][0].size(1),Dataset[0][0][0].size(2))
        img = img_tensor[0]#np.asarray(self.tf_input(img_tensor[0]))
        faces = self.app.get(img)
        img_PIL = Image.fromarray(img.astype(np.uint8))
        face_count = len(faces)

        if face_count > self.max_num:
            a = self.max_num
        else :
            a = face_count

        face_tf_org = []
        for i,f in zip(range(a),faces):
            try: 
                cons = np.linalg.norm(f['bbox'][0:2] - f['bbox'][2:4])//10
                f['bbox'] += np.array([-cons,-cons,cons,cons])
            except:
                pass
            face=img_PIL.crop(f['bbox'])
            face_tf_org.append(transformer(face))

        if len(face_tf_org) == 0:
            # raise Exception("face extarctor could not find any faces")
            return torch.tensor([])
            
        return torch.cat(face_tf_org)
             
    def train(self, dataset_train, epochs, lr: float=1e-3):
        tf_face = self.face_transformer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.prediction_model.parameters(), lr=lr)
        avgLoss = 0
        avgCorrect = 0
        for epoch in range(epochs):
            print(f' Epoch {epoch} ----------------------------')
            for (img,text,sentiment,index) in tqdm(dataset_train):
                # Backpropagation
                optimizer.zero_grad()

                face = self.faceExractor(img)
                if len(face): 
                    face_tf = tf_face(face.float())
                    input_face = face_tf.repeat((2,1,1,1))
                    dan_output,_,_ = self.dan_model(input_face.to(self.device))
                    logits_modelpip = self.Net_Total(dan_output)
                else:
                    logits_modelpip = torch.tensor([0,0,0], device=self.device)
                
                logits_resnet = self.resnet_model(face_tf[None,:].to(self.device))[0]
                output_layer = torch.cat((logits_resnet, logits_modelpip))
                # print(output_layer)
                pred = self.prediction_model(output_layer)
                y = torch.tensor(int(sentiment[0])).to(self.device)
                loss = loss_fn(pred, y)
                avgLoss += loss.item()
                avgCorrect += 1 if pred.argmax(0).item() == int(sentiment[0]) else 0

                # Backpropagation
                loss.backward()
                optimizer.step()

            avgLoss /= dataset_train.__len__()
            avgCorrect /= dataset_train.__len__()
            print(f" -Training Accuracy (Avg) = {(100*avgCorrect):>0.1f}%, Avg loss = {avgLoss:>8f} \n")
            torch.save(self.prediction_model.state_dict(), 'checkpoints/prediction.pth')


    def test(self, dataset_test):
        # testloader = DataLoader(dataset_test, shuffle=True, batch_size=1, pin_memory=True, num_workers=2)
        tf_face = self.face_transformer
        avgCorrect = 0
        
        for (img,text,sentiment,index) in tqdm(dataset_test):
            face = self.faceExractor(img)
            if len(face): 
                face_tf = tf_face(face.float())
                input_face = face_tf.repeat((2,1,1,1))
                dan_output,_,_ = self.dan_model(input_face.to(self.device))
                logits_modelpip = self.Net_Total(dan_output)
            else:
                logits_modelpip = torch.tensor([0,0,0], device=self.device)
            
            logits_resnet = self.resnet_model(face_tf[None,:].to(self.device))[0]

            pred = logits_modelpip + logits_resnet
            avgCorrect += 1 if pred.argmax(0).item() == int(sentiment[0]) else 0
        
        return avgCorrect/dataset_test.__len__()


from libs.MSCTDdataset import MSCTD
if __name__ == "__main__":
    root_dir = '.'
    train_data = MSCTD(
            mode='validation',
            download=False,
            root_dir=root_dir,
            read_mode="single")
    
    test_data = MSCTD(
            mode='test',
            download=False,
            root_dir=root_dir,
            read_mode="single")
    mn = mergeNetwork(modelPath_face='checkpoints/facePipModel.pth')
    mn.train(train_data,1)
    print(mn.test(test_data))

 
