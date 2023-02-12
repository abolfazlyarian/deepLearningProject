import collections
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from models.lxmert_utils import load_lxmert_qa
from optimizers.lamb import Lamb
from optimizers.lookahead import Lookahead
from pretrain.qa_answer_table import load_lxmert_qa

home = str(Path.home())
DataTuple = collections.namedtuple("DataTuple", "dataset loader evaluator")
load_lxmert_qa_path = "/content" + "/snap/pretrained/model"
load_lxmert_best_path="/content/drive/MyDrive/Transfomer/snap/BEST"

class Learner:
    def __init__(self, model, data_tuple_dict, config):
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        base_optim = Lamb(
            params=self.model.parameters(), lr=1e-5, weight_decay=1.2e-6, min_trust=0.25
        )
        self.optim = Lookahead(base_optimizer=base_optim, k=5, alpha=0.8)
        self.lr_scheduler = CyclicLR(
            self.optim, base_lr=1e-5, max_lr=5e-5, cycle_momentum=False
        )
        # self.train_tuple = data_tuple_dict["train_tuple"]
        # self.valid_tuple = data_tuple_dict["valid_tuple"]
        # self.test_tuple = data_tuple_dict["test_tuple"]
        self.train_Loader = data_tuple_dict["train"]
        self.valid_Loader = data_tuple_dict["validation"]
        self.test_Loader = data_tuple_dict["test"]

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.output = "/content/drive/MyDrive/Transfomer" + "/snap/"
        # os.makedirs(self.output, exist_ok=True)
        self.model.to(self.device)
        self.adaptive = config["adaptive_enable"]
        # self.measure_flops = config["measure_flops"]
        # if self.measure_flops:
        #     from thop import clever_format, profile
        self.sparse = config["sparse_enable"]

        if config["load_model"]:
            load_lxmert_qa(load_lxmert_qa_path, self.model, label2ans={0:'neutral',1:'negative',2:'positive'})

        elif config["load_best"]:
            print("loading best model")
            self.load(load_lxmert_best_path)
            # load_lxmert_qa(load_lxmert_best_path, self.model, label2ans={0:'neutral',1:'negative',2:'positive'})
        

    def train(self, num_epochs):
        loader = self.train_Loader
        best_valid = 0.0
        iter_wrapper = lambda x: tqdm(x, total=len(loader))
        loss_total=0
        correct = 0
        for epoch in range(num_epochs):
            loss_total=0
            correct = 0
            t0 = time.time()
            quesid2ans = {}
            for i, (boxes,feats, sent, target) in iter_wrapper(
                enumerate(loader)
            ):
                self.model.train()
                self.optim.zero_grad()
                feats, boxes, target = (
                    feats.to(self.device),
                    boxes.to(self.device),
                    target.to(self.device),
                )
                
                logit = self.model(feats, boxes, sent)
                # assert logit.dim() == target.dim() == 2
                loss = self.criterion(logit, target) * logit.size(1)
                loss_total+=loss
                correct += (logit.argmax(1) == target.argmax(1)).type(torch.float64).sum().item()

                if self.adaptive:

                    adapt_span_loss = 0.0
                    for l in self.model.lxrt_encoder.model.bert.encoder.layer:
                        adapt_span_loss += l.attention.self.adaptive_span.get_loss()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        adapt_span_loss += (
                            l.visual_attention.att.adaptive_span.get_loss()
                        )

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        adapt_span_loss += l.lang_self_att.self.adaptive_span.get_loss()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        adapt_span_loss += l.visn_self_att.self.adaptive_span.get_loss()

                    for l in self.model.lxrt_encoder.model.bert.encoder.r_layers:
                        adapt_span_loss += l.attention.self.adaptive_span.get_loss()

                    loss += adapt_span_loss
                #####################################################
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()
                self.lr_scheduler.step()
                #####################################################
                if self.adaptive:
                    for l in self.model.lxrt_encoder.model.bert.encoder.layer:
                        l.attention.self.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        l.visual_attention.att.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        l.lang_self_att.self.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.x_layers:
                        l.visn_self_att.self.adaptive_span.clamp_param()

                    for l in self.model.lxrt_encoder.model.bert.encoder.r_layers:
                        l.attention.self.adaptive_span.clamp_param()
            #####################################################
            print(f"Epoch {epoch}: Loss {loss_total/len(loader)} Accuracy : {100 * correct/len(loader.dataset)}",)
            if self.valid_Loader is not None:  # Do Validation
                valid_score = self.evaluate(self.valid_Loader)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str = "Epoch %d: Valid %0.2f\n" % (
                    epoch,
                    valid_score * 100.0,
                ) + "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.0)
            
            current_time = time.time() - t0
            print(current_time)
            log_str += "Time elpased for epoch %f\n" % (current_time)
            print(log_str, end="")

            print(log_str)
            

        self.save("LAST")

    def predict(self, eval_tuple):
        """
        Predict the sentiment .

        :param eval: The data  to be evaluated.
        :return: Accuracy over data.
        """
        self.model.eval()
        loader = self.valid_Loader
        iter_wrapper = lambda x: tqdm(x, total=len(loader))
        correct = 0
        print("Predict in progress")
        for i, datum_tuple in iter_wrapper(enumerate(loader)):
            boxes,feats, sent,target = datum_tuple[:4]  # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                logit = self.model(feats, boxes, sent)
                target = target.to(self.device)
                correct += (logit.argmax(1) == target.argmax(1)).type(torch.float64).sum().item()
                
        Acc = correct/len(loader.dataset)
        return Acc
    
    def evaluate(self, eval_tuple: DataTuple):
        """Evaluate all data in data_tuple.""" 
        return self.predict(eval_tuple)

    def predict_test(self):
      """
      Predict the test to test data .
      """
      self.model.eval()
      loader = self.test_Loader
      iter_wrapper = lambda x: tqdm(x, total=len(loader))
      correct = 0
      pred=[]
      y_true =[]
      print("Predict in progress")
      for i, datum_tuple in iter_wrapper(enumerate(loader)):
          boxes,feats, sent,target = datum_tuple[:4]  # Avoid seeing ground truth
          with torch.no_grad():
              feats, boxes = feats.to(self.device), boxes.to(self.device)
              logit = self.model(feats, boxes, sent)
              target = target.to(self.device)
              correct += (logit.argmax(1) == target.argmax(1)).type(torch.float64).sum().item()
              y_true.extend(target.argmax(1).detach().cpu().numpy().tolist())
              pred.extend(logit.argmax(1).detach().cpu().numpy().tolist())

              
      Acc = correct/len(loader.dataset)
      return Acc,y_true,pred
    def test(self):
      return self.predict_test()

    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        state_dict = torch.load(
            "%s.pth" % path,
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
