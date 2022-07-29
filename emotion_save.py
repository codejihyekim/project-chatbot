from random import random
from pip import main
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import random
#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split

from emotion_dataset import BERTDataset
from emotion_classifier import BERTClassifier

class Bert:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.max_len = 100
        self.batch_size = 16
        self.warmup_ratio = 0.1
        self.num_epochs = 30
        self.max_grad_norm = 1
        self.log_interval = 200
        self.learning_rate = 5e-5
        self.dataset_train, self.dataset_test = train_test_split(self.preprocessing(), test_size=0.25, random_state=0)
        self.tokenizer = get_tokenizer()
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def hook(self):
        self.model_train()
        self.save_model()

    def preprocessing(self):
        chatbot_data = pd.read_excel('./data/감성대화말뭉치(원시데이터)_Validation.xlsx')
        chatbot_data.loc[(chatbot_data['Emotion'] == "당황"), 'Emotion'] = 0  # 당황 => 0
        chatbot_data.loc[(chatbot_data['Emotion'] == "분노"), 'Emotion'] = 1  # 분노 => 1
        chatbot_data.loc[(chatbot_data['Emotion'] == "불안"), 'Emotion'] = 2  # 불안 => 2
        chatbot_data.loc[(chatbot_data['Emotion'] == "행복"), 'Emotion'] = 3  # 행복 => 3
        chatbot_data.loc[(chatbot_data['Emotion'] == "슬픔"), 'Emotion'] = 4  # 슬픔 => 4

        data_list = []
        for q, label in zip(chatbot_data['Sentence'], chatbot_data['Emotion']):
            data = []
            data.append(q)
            data.append(str(label))

            data_list.append(data)
        #print(data_list)
        return data_list

    # 정확도 측정을 위한 함수 정의
    def calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
        return train_acc

    def train_data(self):
        # 토큰화
        data_train = BERTDataset(self.dataset_train, 0, 1, self.tok, self.max_len, True, False)  # dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair 순서
        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, num_workers=0)
        return train_dataloader

    def test_data(self):
        data_test = BERTDataset(self.dataset_test, 0, 1, self.tok, self.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, num_workers=0)
        return test_dataloader

    def load_model(self):
        model = BERTClassifier(self.bertmodel, dr_rate=0.5).to(self.device)
        return model

    def optimizer(self):
        model = self.load_model()
        # optimizer와 schedule 설정
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        return optimizer

    def scheduler(self):
        optimizer = self.optimizer()
        train_dataloader = self.train_data()
        t_total = len(train_dataloader) * self.num_epochs
        warmup_step = int(t_total * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
        return scheduler

    def model_train(self):
        model = self.load_model()
        optimizer = self.optimizer()
        scheduler = self.scheduler()
        for e in range(self.num_epochs):
            train_acc = 0.0
            test_acc = 0.0
            model.train()
            device = self.device
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.train_data())):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                loss = self.loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_acc += self.calc_accuracy(out, label)
                if batch_id % self.log_interval == 0:
                    print(
                        "epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                           train_acc / (batch_id + 1)))
            print("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

            model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.test_data())):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                test_acc += self.calc_accuracy(out, label)
            print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))
        return model

    def save_model(self):
        model = self.model_train()
        torch.save(model, './save/chatbot.pt')

if __name__ == '__main__':
    Bert().hook()