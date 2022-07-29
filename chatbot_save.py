import urllib
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from tqdm import tqdm
from chatbot_dataset import ChatbotDataset

class Kogpt:
    def __init__(self):
        self.Q_TKN = "<usr>"
        self.A_TKN = "<sys>"
        self.BOS = '</s>'
        self.EOS = '</s>'
        self.MASK = '<unused0>'
        self.SENT = '<unused1>'
        self.PAD = '<pad>'
        self.epoch = 30
        self.Sneg = -1e18
        self.log_interval = 200
        self.device = torch.device('cuda:0')

    def hook(self):
        self.train()
        self.save_model()

    def crawling(self):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
            filename="data/ChatBotData.csv")

    def load_tokenizer(self):
        koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=self.BOS,
                                                                   eos_token=self.EOS, unk_token="<unk>",
                                                                   pad_token=self.PAD, mask_token=self.MASK)
        return koGPT2_TOKENIZER

    def load_model(self):
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(self.device)
        return model

    @staticmethod
    def collate_batch(batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train(self):
        Chatbot_Data = pd.read_csv("data/ChatBotData.csv")
        train_set = ChatbotDataset(Chatbot_Data, max_len=40)

        #윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
        train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=self.collate_batch)

        model = self.load_model()
        #model.to(device)
        model.train()

        learning_rate = 3e-5
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("start")
        for epoch in range(self.epoch):
            train_acc = 0.0
            for batch_idx, samples in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                token_ids, mask, label = samples
                token_ids = token_ids.long().to(self.device)
                mask = mask.long().to(self.device)
                label = label.long().to(self.device)
                out = model(token_ids)
                out = out.logits      #Returns a new tensor with the logit of the elements of input
                mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
                mask_out = torch.where(mask_3d == 1, out, self.Sneg * torch.ones_like(out))
                loss = criterion(mask_out.transpose(2, 1), label)
                # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
                avg_loss = loss.sum() / mask.sum()
                avg_loss.backward()
                # 학습 끝
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print("epoch {} batch id {}0: loss {} ".format(epoch+1, batch_idx+1, avg_loss))
        print("end")
        return model

    def save_model(self):
        model = self.train()
        PATH = './save/test.pt'
        torch.save(model, PATH)

if __name__ == '__main__':
    Kogpt().hook()
