import torch
from transformers import PreTrainedTokenizerFast
from emotion import Emotion
from tts_mibot import Tts
from stt_mibot import Stt
from emotion import BERTClassifier


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

    def small_chatbot(self):
        Q_TKN = self.Q_TKN
        SENT = self.SENT
        A_TKN = self.A_TKN
        BOS = self.BOS
        EOS = self.BOS
        PAD = self.PAD 
        MASK = self.MASK
        device = torch.device("cpu")
        koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK)
        model = torch.load(f'{self.model_path}/chatbot1.pt', map_location=device)
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
                    end_text = "당신은 충분히 잘 하고 있어요 남은 하루도 행복하게 보내길 바래요"
                    print(f"mibot > {end_text}")
                    Tts.run(input_text=end_text)
                    break
                elif q == "좋아 음악 추천해줘" or q == "음악 추천" or q == '음악 추천해줘':
                    Emotion.music(self)
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

if __name__ == '__main__':
    Chatbot().small_chatbot()