from faceAnalysis.networks.dan import DAN
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch 
import torch.nn as nn
from faceAnalysis.networks.MixFace import MixFaceMLP
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from textAnalysis.bert import TextDataset, TextClassifier, faceTextDataset
from transformers import AutoTokenizer


class mergeImageTextNetwork():
    def __init__(self, max_num: int=1, modelPath_face: str='checkpoints/facePipModel.pth',
                  modelPath_text: str='checkpoints/bert.pth', num_head: int=4) -> None:
        """
            combine face and image netwoek

            Parameters:
            ------------------------
            `max_num` : find maximum number of face in each image
            `modelPath_face` : path of face model
            `modelPath_image` : path of image model
            `num_head` : Number of attention head
        """
        # self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
        self.max_num = max_num

        self.face_transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
                                                    
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

        pip_model = torch.load(modelPath_face)
        self.dan_model = DAN(num_head=num_head,num_class=3,pretrained=False)
        self.dan_model.load_state_dict(pip_model['model_state_dict'])
        self.dan_model.requires_grad_(False)
        self.dan_model.to(self.device)
        
        text_model = torch.load(modelPath_text)
        self.bert_model = TextClassifier()
        self.bert_model.load_state_dict(text_model)
        self.bert_model.requires_grad_(False)
        self.bert_model.to(self.device)

        self.calssifier = MLP()
        self.calssifier.to(self.device)

        # self.app.prepare(ctx_id=0)


    def faceExractor(self, img_tensor):
        """
            face exctraction from image

            Parameters:
            ------------------------
            `img_tensor` : image with type=tensor
        """
        transformer = transforms.PILToTensor()
        img = img_tensor[0]
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
             
    def train(self, train_loader, valid_loader, epochs: int=1, lr: float=1e-3, modelPath_pred='checkpoints/textimageClassifier.pth'):
        """
            train prediction layer (classifier)

            Parameters:
            ------------------------
            `dataset_train` : train dataset
            `epochs` : number of epochs
            `lr` : Initial learning rate for sgd
            `modelPath_pred` : path of prediction model
        """
        tf_face = self.face_transformer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.calssifier.parameters(), lr=lr)
        

        for epoch in range(epochs):
            print(f' Epoch {epoch+1} ----------------------------')
            avgLoss = 0
            avgCorrect = 0
            self.calssifier.train()
            for (face,sentiment,text) in tqdm(train_loader):
                # Backpropagation
                
                if len(face): 
                    face_tf = tf_face(face)
                    # input_face = face_tf.repeat((2,1,1,1))
                    input_face = face_tf
                    _,_,dan_head = self.dan_model(input_face.to(self.device))
                    #TODO change code 
                    dan_head = dan_head.sum(dim=1)#.sum(dim=0)
                else:
                    dan_head = torch.zeros(512, device=self.device)
                
                text_encode = self.tokenizer(text, truncation=True, padding=True, max_length=128)
                item = {key: torch.tensor(val) for key, val in text_encode.items()}
                input_ids = item['input_ids'].to(self.device)
                attention_mask = item['attention_mask'].to(self.device)
                _,bert_head,_,_ = self.bert_model(input_ids, attention_mask=attention_mask)

                feature_layer = torch.cat((dan_head, bert_head),1) # .view(-1)
                pred = self.calssifier(feature_layer)
                # y = nn.functional.one_hot(torch.tensor(sentiment), num_classes=3).float().to(self.device).view(-1)
                y = sentiment.to(self.device)
                loss = loss_fn(pred, y)
                avgLoss += loss.item()
                # avgCorrect += 1 if pred.argmax(1).item() == sentiment else 0
                avgCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avgLoss /= train_loader.__len__()
            avgCorrect /= len(train_loader.dataset)
            print(f" -Training Accuracy (Avg) = {(100*avgCorrect):>0.1f}%, loss = {avgLoss:>8f} \n")

            avgLoss = 0
            avgCorrect = 0
            self.calssifier.eval()
            with torch.no_grad():
                for (face,sentiment,text) in tqdm(valid_loader):
                    # Compute prediction and loss
                    if len(face): 
                        face_tf = tf_face(face)
                        # input_face = face_tf.repeat((2,1,1,1))
                        input_face = face_tf
                        _,_,dan_head = self.dan_model(input_face.to(self.device))
                        #TODO change code 
                        dan_head = dan_head.sum(dim=1)#.sum(dim=0)
                    else:
                        dan_head = torch.zeros(512, device=self.device)
                                    
                    text_encode = self.tokenizer(text, truncation=True, padding=True, max_length=128)
                    item = {key: torch.tensor(val) for key, val in text_encode.items()}
                    input_ids = item['input_ids'].to(self.device)
                    attention_mask = item['attention_mask'].to(self.device)
                    _,bert_head,_,_ = self.bert_model(input_ids, attention_mask=attention_mask)

                    feature_layer = torch.cat((dan_head, bert_head),1) #.view(-1)
                    pred = self.calssifier(feature_layer)
                    # y = nn.functional.one_hot(torch.tensor(sentiment), num_classes=3).float().to(self.device).view(-1)
                    y = sentiment.to(self.device)
                    loss = loss_fn(pred, y)
                    avgLoss += loss.item()
                    # avgCorrect += 1 if pred.argmax(0).item() == sentiment else 0
                    avgCorrect += (pred.argmax(1) == y).type(torch.float64).sum().item()

            avgLoss /= valid_loader.__len__()
            avgCorrect /= len(valid_loader.dataset)
            print(f" -Validation Accuracy = {(100*avgCorrect):>0.1f}%, loss = {avgLoss:>8f} \n")

        torch.save(self.calssifier.state_dict(), modelPath_pred)

    def run(self, dataset_test, modelPath_predict='checkpoints/prediction.pth',pred_f=False):
        """
            test combining two network

            Parameters:
            ------------------------
            `dataset_test` : test dataset
            `modelPath_pred` : path of prediction model
        """
        prediction_sentiment = []
        tf_face = self.face_transformer
        avgCorrect = 0

        model_predict = torch.load(modelPath_predict)
        self.prediction_model.load_state_dict(model_predict)

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

            
            if pred_f:
                logits_modelpip1 = len(face)*logits_modelpip
                output_layer = torch.cat((logits_resnet, logits_modelpip1))
                pred = self.prediction_model(output_layer)
            else:
                pred = len(face)*logits_modelpip + logits_resnet

            avgCorrect += 1 if pred.argmax(0).item() == int(sentiment[0]) else 0
            prediction_sentiment.append(pred)

        print(f" - Test Accuracy = {(100*(avgCorrect/dataset_test.__len__())):>0.1f}\n")
        return prediction_sentiment


class MSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(16, 3),
        )
        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


