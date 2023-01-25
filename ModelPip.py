import os
import sys
import warnings
from tqdm import tqdm
import argparse
from torchvision.transforms import ToTensor, Resize, Compose
from libs.MSCTDdataset import MSCTD,FaceNetwrok_Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
from networks.dan import DAN
from networks.MixFace import MixFaceMLP
from PIL import Image


eps = sys.float_info.epsilon

def warn(*args, **kwargs):
    pass
warnings.warn = warn



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MSCTD_path', type=str, default='datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--Model_path', type=str, default=4, help='set pretraind DAN Model Path')
    return parser.parse_args()

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
    data.append(i[0])
    labels.append(i[1])
    index.append(i[2])
  return torch.concat(data),torch.concat(labels),torch.concat(index)
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


def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=args.num_head,num_class=3,pretrained=False)
    # model.load_state_dict(torch.load(args.Model_path)['model_state_dict'])
    model.requires_grad_(False)
    model.to(device)

    Net5=MixFaceMLP(dim=6)
    Net4=MixFaceMLP(dim=5)
    Net3=MixFaceMLP(dim=4)
    Net2=MixFaceMLP(dim=3)
    Net1=MixFaceMLP(dim=2)

    
    Net1.to(device)
    Net2.to(device)
    Net3.to(device)
    Net4.to(device)
    Net5.to(device)
    # faceDetector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
    # faceDetector.prepare(ctx_id=0)
    
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    # train_dataset = MSCTD(root_dir=args.MSCTD_path,mode='train',transformer=Compose([]),read_mode='single')
    train_dataset = FaceNetwrok_Dataset(root_dir=args.MSCTD_path,mode='train',transformer=data_transforms)

    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,
                                               collate_fn=collate_fn,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

       
    # val_dataset = MSCTD(root_dir=args.MSCTD_path,mode='validation',transformer=Compose([]),read_mode='single')
    val_dataset = FaceNetwrok_Dataset(root_dir=args.MSCTD_path,mode='validation',transformer=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())
   
      
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,collate_fn= collate_fn, 
                                               pin_memory = True)



    # test_dataset = FaceNetwrok_Dataset(FaceModel=faceDetector,root_dir=args.MSCTD_path,mode='test',transformer=data_transforms_val)

    # print('Test set size:', test_dataset.__len__())
    
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                            batch_size = 1,
    #                                            num_workers = args.workers,
    #                                            shuffle = False,  
    #                                            pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    # criterion_af = AffinityLoss(device,num_class=3,feat_dim=512)
    # criterion_pt = PartitionLoss()

    # params = list(model.parameters()) + list(criterion_af.parameters())
    params=list(Net1.parameters())+list(Net2.parameters())+list(Net3.parameters())+list(Net4.parameters())+list(Net5.parameters())
    # optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(params,lr=args.lr, weight_decay = 1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5,last_epoch=-1)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        
        Net1.train()
        Net2.train()
        Net3.train()
        Net4.train()
        Net5.train()
        for img,targets,index in train_loader:
            targets=targets.to(device)
            
            optimizer.zero_grad()

            if len(img):  
                iter_cnt += 1
                faces = img.to(device)
                out_,feat,heads = model(faces)
                loss=0
                for logits,label in zip(torch.split(out_,f_size(index)),targets):
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
                        out=logits

                  # print(out[None,:],targets)
                    loss += criterion_cls(out[None,:],label)

                    _, predicts = torch.max(out[None,:], 1)
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
            
            for img,targets,index in val_loader:
                targets=targets.to(device)
                if len(img):  
                    iter_cnt+=1
                    faces = img.to(device)
                    out_,feat,heads = model(faces)
                    for logits,label in zip(torch.split(out_,f_size(index)),targets):
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
                            out=logits
                        # print(out[None,:],targets)
                        running_loss += criterion_cls(out[None,:],label)
                        
                        _, predicts = torch.max(out[None,:], 1)
                        correct_num = torch.eq(predicts, targets).sum()
                        bingo_cnt += correct_num.cpu()
                        sample_cnt += 1
 
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




if __name__ == "__main__":
    # train_Dataset=MSCTD(mode='train',download=True,root_dir='.',transformer=transforms.Compose([]))
    # val_Dataset=MSCTD(mode='validation',download=True,root_dir='.',transformer=transforms.Compose([]))
    # test_Dataset=MSCTD(mode='test',download=True,root_dir='.',transformer=transforms.Compose([]))
    # Models_path=os.listdir('checkpoints')
    run_training()
    # path=os.path.join('checkpoints', "rafdb_epoch"+str(40)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))