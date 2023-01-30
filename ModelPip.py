import os
import sys
import warnings
from tqdm import tqdm
import argparse
from torchvision.transforms import ToTensor, Resize, Compose
from libs.MSCTDdataset import MSCTD
from libs.faceDataset import faceNetwrokDataset
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score

from networks.dan import DAN
from networks.MixFace import MixFaceMLP
from PIL import Image



eps = sys.float_info.epsilon


class AffinityLoss(nn.Module):
    def __init__(self, device, num_class=3, feat_dim=512):
        super(AffinityLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.device = device

        self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(device))

    def forward(self, x, labels):
        x = self.gap(x).view(x.size(0), -1)

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_class).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        dist = dist / self.centers.var(dim=0).sum()

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class PartitionLoss(nn.Module):
    def __init__(self, ):
        super(PartitionLoss, self).__init__()
    
    def forward(self, x):
        num_head = x.size(1)

        if num_head > 1:
            var = x.var(dim=1).mean()
            ## add eps to avoid empty var case
            loss = torch.log(1+num_head/(var+eps))
        else:
            loss = 0
            
        return loss

def collate_fn(x):
  labels=[]
  data=[]
  index=[]
  for i in x:
    if   len(i[1]) != 0 and len(i[2]) != 0 and len(i[0]) != 0 :
      # print(len(i[0]),len(i[1]),len(i[2]))
      data.append(i[0])
      labels.append(i[1])
      index.append(i[2])

  if len(labels):
    return torch.concat(data),torch.concat(labels),torch.concat(index)
  else:
    return #torch.tensor([]),torch.tensor([]),torch.tensor([])
    
def f_size(x):
  a=[]
  index=[]
  for i,j in zip(range(len(x)),x):
    if j in a:
      pass
    else :
      a.append(j)
      index.append(i)
  index_n=index.copy()
  index_n.append(len(x))
  return (np.array(index_n[1::])-np.array(index)).tolist()

#TODO augmentation
def train(batch_size: int=64,
          MSCTD_path: str='.',
          num_head: int=4,
          workers: int=2,
          lr: float=0.1,
          epochs: int=40,
          augmentation: list=[]):
    
    """
        Parameters:
        ------------------------
        `batch_size` : size of batch
        `MSCTD_path` : MSCTD path ?????????????????
        `num_head` : Number of attention head
        `workers` : Number of data loading workers
        `lr` : Initial learning rate for sgd
        `epochs` : Total training epochs
        `augmentation` : list of augmentation
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=num_head,num_class=3,pretrained=False)
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.requires_grad_(False)
    model.to(device)

    Net5=MixFaceMLP(dim=6)
    Net4=MixFaceMLP(dim=5)
    Net3=MixFaceMLP(dim=4)
    Net2=MixFaceMLP(dim=3)
    Net1=MixFaceMLP(dim=2)
    Net_Total=MixFaceMLP(dim=6)
    
    Net1.to(device)
    Net2.to(device)
    Net3.to(device)
    Net4.to(device)
    Net5.to(device)
    Net_Total.to(device)
    # faceDetector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
    # faceDetector.prepare(ctx_id=0)
    
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    # train_dataset = MSCTD(root_dir=MSCTD_path,mode='train',transformer=Compose([]),read_mode='single')
    train_dataset = faceNetwrokDataset(root_dir=MSCTD_path,mode='train',transformer=data_transforms)

    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = True,
                                               collate_fn=collate_fn,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

       
    # val_dataset = MSCTD(root_dir=MSCTD_path,mode='validation',transformer=Compose([]),read_mode='single')
    val_dataset = faceNetwrokDataset(root_dir=MSCTD_path,mode='validation',transformer=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())
   
      
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = False,collate_fn= collate_fn, 
                                               pin_memory = True)



    # test_dataset = faceNetwrokDataset(FaceModel=faceDetector,root_dir=MSCTD_path,mode='test',transformer=data_transforms_val)

    # print('Test set size:', test_dataset.__len__())
    
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                            batch_size = 1,
    #                                            num_workers = workers,
    #                                            shuffle = False,  
    #                                            pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    # criterion_af = AffinityLoss(device,num_class=3,feat_dim=512)
    # criterion_pt = PartitionLoss()

    # params = list(model.parameters()) + list(criterion_af.parameters())
    params=list(Net1.parameters())+list(Net2.parameters())+list(Net3.parameters())+list(Net4.parameters())+list(Net5.parameters())
    params=Net_Total.parameters()

    # optimizer = torch.optim.SGD(params,lr=lr, weight_decay = 1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(params,lr=lr, weight_decay = 1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5,last_epoch=-1)

    best_acc = 0
    for epoch in tqdm(range(1, epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        
        Net1.train()
        Net2.train()
        Net3.train()
        Net4.train()
        Net5.train()
        Net_Total.train()
        for img,targets,index in train_loader:
            
            targets = targets.type(torch.LongTensor)
            targets=targets.to(device)
            optimizer.zero_grad()

            if len(img):  
              iter_cnt += 1
              faces = img.to(device)
              out_,feat,heads = model(faces)
              loss=0
              Output_total=[]
              for logits,label in zip(torch.split(out_,f_size(index)),targets):
                label = torch.tensor([label]).to(device)


                # out=Net_Total(logits)
                # v,_=torch.mode(torch.argmax(logits,1))
                # out = torch.zeros(size=(3,)).to(device)
                # out[v] = 1.0

                if len(logits)==6:
                    out=Net5(logits)
                elif len(logits)==5:
                    out=Net4(logits)
                elif len(logits)==4:
                    out=Net3(logits)
                elif len(logits) == 3:
                    out=Net2(logits)
                elif len(logits) == 2:
                    out=Net1(logits)
                elif len(logits) ==1:
                    out=logits[0]
                
                # try:
                # print(Output_total[0:1].shape)
                Output_total.append(out[None,:])
                # print(out.shape)

              # print(Output_total[0])
              O=torch.concat(Output_total)


              loss = criterion_cls(O,targets)
              
              _, predicts = torch.max(O, 1)
              correct_num = torch.eq(predicts, targets).sum()
              correct_sum += correct_num

              loss.backward()
              optimizer.step()
              running_loss += loss

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
          running_loss = 0.0
          iter_cnt = 0
          bingo_cnt = 0
          sample_cnt = 0
          
          ## for calculating balanced accuracy
          y_true = []
          y_pred = []

          Net1.eval()
          Net2.eval()
          Net3.eval()
          Net4.eval()
          Net5.eval()
          Net_Total.eval()
          
          for img,targets,index in val_loader:
            targets = targets.type(torch.LongTensor)
            targets=targets.to(device)
            if len(img):  
                iter_cnt+=1
                faces = img.to(device)
                out_,feat,heads = model(faces)
                Output_total=[]
                for logits,label in zip(torch.split(out_,f_size(index)),targets):

                    # out=Net_Total(logits)
                    # v,_=torch.mode(torch.argmax(logits,1))
                    # out = torch.zeros(size=(3,)).to(device)
                    # out[v] = 1.0




                    if len(logits)==6:
                        out=Net5(logits)
                    elif len(logits)==5:
                        out=Net4(logits)
                    elif len(logits)==4:
                        out=Net3(logits)
                    elif len(logits) == 3:
                        out=Net2(logits)

                    elif len(logits) == 2:
                        out=Net1(logits)
                    
                    elif len(logits) ==1:
                        out=logits[0]
                    Output_total.append(out[None,:])
                    sample_cnt += 1

                O=torch.concat(Output_total)
                running_loss += criterion_cls(O,targets)
                
                _, predicts = torch.max(O, 1)
                correct_num = torch.eq(predicts, targets).sum()
                bingo_cnt += correct_num.cpu()
                
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())
            
        running_loss = running_loss/iter_cnt   
        scheduler.step()

        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        best_acc = max(acc,best_acc)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred),4)

        tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
        tqdm.write("best_acc:" + str(best_acc))

        if acc > 0.38 and acc == best_acc:
            torch.save({'iter': epoch,
                        'model_state_dict': model.state_dict(),
                        'Net_1':Net1.state_dict(),
                        'Net_2':Net2.state_dict(),
                        'Net_3':Net3.state_dict(),
                        'Net_4':Net4.state_dict(),
                        'Net_5':Net5.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),},
                        os.path.join('checkpoints', "PipModel_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))
            tqdm.write('Model saved.')

def test(batch_size: int=64,
         model_path: str='.',
         Facedataset_path: str='.',
         num_head: int=4,
         workers: int=2):
    
    """
        Parameters:
        ------------------------
        `Facedataset_path` : Raf-DB dataset path ?????????????????
        `num_head` : Number of attention head
        `model_path` : path of model saved ????????????????????
        `batch_size` : size of batch
        `workers` : Number of data loading workers
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    saved_model=torch.load(model_path)

    model = DAN(num_head=num_head,num_class=3,pretrained=False)
    model.load_state_dict(saved_model['model_state_dict'])
    model.requires_grad_(False)
    model.to(device)

    Net_Total=MixFaceMLP(dim=6)
    Net_Total.to(device)
    # Net5=MixFaceMLP(dim=6)
    # Net5.load_state_dict(saved_model['Net_5'])

    # Net4=MixFaceMLP(dim=5)
    # Net4.load_state_dict(saved_model['Net_4'])

    # Net3=MixFaceMLP(dim=4)
    # Net3.load_state_dict(saved_model['Net_3'])

    # Net2=MixFaceMLP(dim=3)
    # Net2.load_state_dict(saved_model['Net_2'])

    # Net1=MixFaceMLP(dim=2)
    # Net1.load_state_dict(saved_model['Net_1'])


    
    # Net1.to(device)
    # Net2.to(device)
    # Net3.to(device)
    # Net4.to(device)
    # Net5.to(device)
    def collate_fn_test(x):
      labels=[]
      data=[]
      index=[]
      miss_face_label=[]
      for i in x:
        if   len(i[1]) != 0 and len(i[2]) != 0 and len(i[0]) != 0 :
          # print(len(i[0]),len(i[1]),len(i[2]))
          data.append(i[0])
          labels.append(i[1])
          index.append(i[2])
        else :
          miss_face_label.append(i[1])
          

      if len(labels):
        return torch.concat(data),torch.concat(labels),torch.concat(index),torch.concat(miss_face_label)
      else:
        return #torch.tensor([]),torch.tensor([]),torch.tensor([])
    
    data_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    test_dataset = faceNetwrokDataset(root_dir=Facedataset_path ,mode='test',transformer=data_transforms_val)

    print('Test set size:', test_dataset.__len__())
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size = batch_size,
                                            num_workers = workers,
                                            collate_fn=collate_fn_test,
                                            shuffle = False,  
                                            pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    y_true=[]
    y_pred=[]
    running_loss = 0.0
    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
    for img,targets,index,miss_face_label in tqdm(test_loader):
        targets = targets.type(torch.LongTensor)
        targets=targets.to(device)
        if len(img):  
            iter_cnt+=1
            faces = img.to(device)
            out_,feat,heads = model(faces)
            Output_total=[]
            for logits,label in zip(torch.split(out_,f_size(index)),targets):

                out=Net_Total(logits)
                # if len(logits)==6:
                #     out=Net5(logits)
                # elif len(logits)==5:
                #     out=Net4(logits)
                # elif len(logits)==4:
                #     out=Net3(logits)
                # elif len(logits) == 3:
                #     out=Net2(logits)

                # elif len(logits) == 2:
                #     out=Net1(logits)
                
                # elif len(logits) ==1:
                #     out=logits[0]
                Output_total.append(out[None,:])
               
            O=torch.concat(Output_total)
            running_loss += criterion_cls(O,targets)
            
            _, predicts = torch.max(O, 1)
            correct_num = torch.eq(predicts, targets).sum()
            bingo_cnt += correct_num.cpu()
            
            y_true.append(targets.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())
        if len(miss_face_label):
           
            O=torch.randn(size=(len(miss_face_label),3))
            running_loss += criterion_cls(O,miss_face_label)

            _, predicts = torch.max(O, 1)
            correct_num = torch.eq(predicts, miss_face_label).sum()
            bingo_cnt += correct_num.cpu()
            
            y_true.append(miss_face_label.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())

    running_loss = running_loss/iter_cnt   
    
    

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = bingo_cnt.float()/float(len(y_true))
    acc = np.around(acc.numpy(),4)

    balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred),4)

    tqdm.write("test accuracy:%.4f. bacc:%.4f. Loss:%.3f" % ( acc, balanced_acc, running_loss))
    tqdm.write("best_acc:" + str(acc))

if __name__ == "__main__":
   
    train()
    # path=os.path.join('checkpoints', "rafdb_epoch"+str(40)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))