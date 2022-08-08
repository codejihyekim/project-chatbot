from transformers import PreTrainedTokenizerFast
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
import random
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from tts_mibot import Tts
from stt_mibot import Stt
from playsound import playsound

class Chatbot:

    def __init__(self) -> None:
        self.Q_TKN = "<usr>"
        self.A_TKN = "<sys>"
        self.BOS = '</s>'
        self.EOS = '</s>'
        self.MASK = '<unused0>'
        self.SENT = '<unused1>'
        self.PAD = '<pad>'
        self.model_path = './model'

        self.file = './music_mp3'

    def small_chatbot(self):
        Q_TKN = self.Q_TKN
        SENT = self.SENT
        A_TKN = self.A_TKN
        BOS = self.BOS
        EOS = self.EOS
        PAD = self.PAD
        MASK = self.MASK
        device = torch.device("cpu")
        koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK)
        model = torch.load(f'{self.model_path}/chatbot.pt', map_location=device)
        with torch.no_grad():
            intro = "안녕하세요 저는 미봇이에요"
            print(f"mibot > {intro}")
            Tts.run(input_text=intro)
            while 1:
                print("마이크로 말씀해주세요")
                q = Stt.get_audio()
                text = f"나 > {q}".strip()
                print(text)
                if q == "대화 종료" or q == "종료":
                    end_text = "당신은 충분히 잘 하고 있어요, 남은 하루도 행복하게 보내길 바래요"
                    print(f"mibot > {end_text}")
                    Tts.run(input_text=end_text)
                    break
                elif q == "좋아 음악 추천해줘" or q == "음악 추천" or q == '음악 추천해줘':
                    self.music()
                    continue
                a = ""
                while 1:
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
                    pred = model(input_ids)
                    pred = pred.logits
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print("mibot > {}".format(a.strip()))
                Tts.run(input_text=a.strip())

    # emotion
    def music(self):
        max_len = 100
        batch_size = 16
        bertmodel, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
        device = torch.device("cuda:0")
        with torch.no_grad():
            end = 1
            while end == 1:
                bot = "mibot > 현재 기분을 알려주세요 \n"
                print(bot)
                Tts.run(input_text="현재 기분을 알려주세요")
                sentence = Stt.get_audio()
                text = f"나 > {sentence}".strip()
                print(text)
                model = torch.load('./model/emotion.pt')
                data = [sentence, '0']
                dataset_another = [data]

                another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
                test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

                model.eval()

                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
                    token_ids = token_ids.long().to(device)
                    segment_ids = segment_ids.long().to(device)

                    valid_length = valid_length
                    label = label.long().to(device)

                    out = model(token_ids, valid_length, segment_ids)

                    happy_music = ['나빌레라']
                    sad_music = ['어른']
                    # sad_music = ['나의 사춘기에게', '꽃길', '이별후회', '한숨', '어른']
                    # happy_music = ['나빌레라', '빨간 맛', '딩가딩가', '상상더하기', '마지막처럼']
                    angry_music = ['그건 니 생각이고', '대취타', '내가 제일 잘나가', '닥쳐줘요', '보여줄게']
                    unrest_music = ['도망가자', '혼자라고 생각말기', '괜찮아요', '마음을 드려요', '아프지 말고 아프지 말자']
                    embar_music = ['밤편지', '양화대교', '무릎', '서울의 잠 못 이루는 밤', '밝게 빛나는 별이 되어 비춰줄게']

                    test_eval = []
                    for i in out:
                        logits = i
                        logits = logits.detach().cpu().numpy()

                        if np.argmax(logits) == 0:
                            test_eval.append("당황스러우셨군요 ")
                        elif np.argmax(logits) == 1:
                            test_eval.append("화가나셨군요 ")
                        elif np.argmax(logits) == 2:
                            test_eval.append("불안하시군요 ")
                        elif np.argmax(logits) == 3:
                            test_eval.append("행복하시군요 ")
                        elif np.argmax(logits) == 4:
                            test_eval.append("슬프시군요 ")

                    res = ""
                    if test_eval[0] == "행복하시군요 ":
                        res = '행복한 기분을 위해 ' + random.choice(happy_music) + " 를 들려드릴게요"
                    elif test_eval[0] == "슬프시군요 ":
                        res = '슬픈 기분을 위해 ' + random.choice(sad_music) + " 를 들려드릴게요"
                    elif test_eval[0] == "화가나셨군요 ":
                        res = '화난 기분을 위해 ' + random.choice(angry_music) + " 를 들려드릴게요"
                    elif test_eval[0] == "불안하시군요 ":
                        res = '불안한 기분을 위해 ' + random.choice(unrest_music) + " 를 들려드릴게요"
                    elif test_eval[0] == "당황스러우셨군요 ":
                        res = '당황스러운 기분을 위해 ' + random.choice(embar_music) + " 를 들려드릴게요"

                    print("mibot > " + f'{res}')
                    Tts.run(input_text=res)
                    if test_eval[0] == "행복하시군요 ":
                        playsound(f'{self.file}/nabillera1.mp3')
                        print('음악이 재생 되는 중')
                    elif test_eval[0] == "슬프시군요 ":
                        playsound(f'{self.file}/adult1.mp3')
                        print('음악이 재생 되는 중')
                break


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=5,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))





if __name__ == '__main__':
    Chatbot().small_chatbot()