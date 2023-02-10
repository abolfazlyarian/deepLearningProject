import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from libs.faceDataset import faceDataset
from sklearn.metrics import balanced_accuracy_score
from faceAnalysis.networks.dan import DAN


eps = sys.float_info.epsilon
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def train(model_path: str, 
          batch_size: int=64,
          Facedataset_path: str='.',
          num_head: int=4,
          workers: int=2,
          lr: float=0.1,
          epochs: int=40,
          augmentation: list=[]):
    
    """
        Parameters:
        ------------------------
        `batch_size` : size of batch
        `Facedataset_path` : face dataset path 
        `num_head` : Number of attention head
        `workers` : Number of data loading workers
        `lr` : Initial learning rate for sgd
        `epochs` : Total training epochs
        `augmentation` : list of augmentation
        `model_path` : path of model saved
    """
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DAN(num_head=num_head,num_class=3,pretrained=False)
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
    
    train_dataset = faceDataset(root_dir=Facedataset_path,augmentation=augmentation, mode='train',transformer=data_transforms)
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

       
    val_dataset = faceDataset(root_dir=Facedataset_path,augmentation=[], mode='validation',transformer=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = False,  
                                               pin_memory = True)
    
    # test_dataset = faceDataset(root_dir=Facedataset_path,augmentation=[],mode='test',transformer=data_transforms_val)

    # print('Test set size:', test_dataset.__len__())
    
    # test_loader = torch.utils.data.DataLoader(test_dataset,
    #                                            batch_size = batch_size,
    #                                            num_workers = workers,
    #                                            shuffle = False,  
    #                                            pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_af = AffinityLoss(device,num_class=3,feat_dim=512)
    criterion_pt = PartitionLoss()

    params = list(model.parameters()) + list(criterion_af.parameters())
    # optimizer = torch.optim.SGD(params,lr=lr, weight_decay = 1e-4, momentum=0.9)
    optimizer = torch.optim.Adam(params,lr=lr, weight_decay = 1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5,last_epoch=-1)

    best_acc = 0
    for epoch in tqdm(range(1, epochs + 1)):
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

            if not os.path.exists(model_path):
                os.mkdir(model_path)

            if acc > 0.30 and acc == best_acc:
                if len(augmentation):
                    torch.save({'iter': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                os.path.join(model_path,"faceDAN_aug.pth"))
                    
                    path_save = os.path.join(model_path,"faceDAN_aug.pth")
                    tqdm.write(f"Model saved in : {path_save}")
                
                else:
                    torch.save({'iter': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),},
                                os.path.join(model_path,"faceDAN.pth"))
                    
                    path_save = os.path.join(model_path,"faceDAN.pth")
                    tqdm.write(f"Model saved in : {path_save}")
    

def test(model_path: str,
         batch_size: int=64,
         Facedataset_path: str='.',
         num_head: int=4,
         workers: int=2,
         augmentation=[]):
    
    """
        Parameters:
        ------------------------
        `Facedataset_path` : face dataset path 
        `num_head` : Number of attention head
        `model_path` : path of model saved
        `batch_size` : size of batch
        `workers` : Number of data loading workers
        `augmentation` : list of augmentation
    """
    data_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]) 

    if len(augmentation):
        test_dataset = faceDataset(root_dir=Facedataset_path,
                                augmentation=augmentation,
                                mode='train',
                                transformer=data_transforms_test,just_aug=True)
    else:
        test_dataset = faceDataset(root_dir=Facedataset_path,
                                augmentation=[],
                                mode='test',
                                transformer=data_transforms_test)
    
    print('Test set size:', test_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = workers,
                                               shuffle = False,  
                                               pin_memory = True)
    with torch.no_grad():
        model = DAN(num_head=num_head, num_class=3, pretrained=False)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])
        model.requires_grad_(False).to(device)
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        best_acc = 0
        ## for calculating balanced accuracy
        y_true = []
        y_pred = []

        model.eval()
        for (imgs, targets) in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            out,feat,heads = model(imgs)
            # loss = criterion_cls(out,targets) + criterion_af(feat,targets) + criterion_pt(heads)

            # running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)
            
            y_true.append(targets.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())
    
        # running_loss = running_loss/iter_cnt   

        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        best_acc = max(acc,best_acc)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred),4)

        print("test accuracy:%.4f. bacc:%.4f" % (acc, balanced_acc))
        return y_true, y_pred
       

