import os
import sys
import warnings
from tqdm import tqdm
import argparse

from libs.MSCTDdataset import MSCTD
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from networks.dan import DAN
from networks.MixFace import MixFaceMLP




def warn(*args, **kwargs):
    pass
warnings.warn = warn



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Facedataset_path', type=str, default='datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')
    parser.add_argument('--Model_path', type=str, default=4, help='set pretraind DAN Model Path')
    return parser.parse_args()

class ModelPip(nn.Module):

    def __init__(self,Model_path=None) -> None:
        super(MixFaceMLP,self).__init__()

        self.faceDetector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],allowed_modules=['detection'])
        self.faceDetector.prepare(ctx_id=0)
        self.DAN_Model = DAN(num_head=4,num_class=3,pretrained=False)
        self.DAN_Model.load_state_dict(torch.load())
        self.DAN_Model.requires_grad_(False)
        self.DAN_Model.to(device)
    def forward(self,x):
        pass

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=args.num_head,num_class=3,pretrained=False)
    model.load_state_dict(torch.load(args.Model_path))
    model.requires_grad_(False)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    train_dataset = FACE(root_dir=args.Facedataset_path,mode='train',transformer=data_transforms)
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

       
    val_dataset = FACE(root_dir=args.Facedataset_path,mode='validation',transformer=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    test_dataset = FACE(root_dir=args.Facedataset_path,mode='test',transformer=data_transforms_val)

    print('Test set size:', test_dataset.__len__())
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_af = AffinityLoss(device,num_class=3,feat_dim=512)
    criterion_pt = PartitionLoss()

    params = list(model.parameters()) + list(criterion_af.parameters())
    # optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(params,lr=args.lr, weight_decay = 1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-5,last_epoch=-1)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)

            loss = criterion_cls(out,targets) + 1* criterion_af(feat,targets) + 1*criterion_pt(heads)  #89.3 89.4

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

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

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                out,feat,heads = model(imgs)
                loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
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
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('checkpoints', "rafdb_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))
                tqdm.write('Model saved.')

    with torch.no_grad():

        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        
        ## for calculating balanced accuracy
        y_true = []
        y_pred = []

        model.eval()
        for (imgs, targets) in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)
            loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

            running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)
            
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

        tqdm.write("[Epoch %d] test accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
        tqdm.write("best_acc:" + str(best_acc))



if __name__ == "__main__":
    train_Dataset=MSCTD(mode='train',download=True,root_dir='.',transformer=transforms.Compose([]))
    val_Dataset=MSCTD(mode='validation',download=True,root_dir='.',transformer=transforms.Compose([]))
    test_Dataset=MSCTD(mode='test',download=True,root_dir='.',transformer=transforms.Compose([]))
    Models_path=os.listdir('checkpoints')
    # path=os.path.join('checkpoints', "rafdb_epoch"+str(40)+"_acc"+str(acc)+"_bacc"+str(balanced_acc)+".pth"))