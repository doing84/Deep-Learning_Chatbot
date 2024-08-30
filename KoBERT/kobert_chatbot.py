import torch
import torch.nn as nn
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp

# KoBERT 모델과 vocab 로드 및 텍스트 생성에 맞춘 레이어 정의
class KoBERTForTextGeneration(nn.Module):
    def __init__(self, bert, vocab_size):
        super(KoBERTForTextGeneration, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, vocab_size)  # vocab_size로 출력 크기를 맞춤

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # [CLS] 토큰의 마지막 히든 상태를 가져옴
        logits = self.classifier(pooled_output)
        return logits

def get_pytorch_kobert_model():
    # 기존에 사용 중인 KoBERT 모델 로드 함수
    from kobert.pytorch_kobert import get_pytorch_kobert_model
    return get_pytorch_kobert_model()

def get_tokenizer():
    from kobert.utils import get_tokenizer
    return get_tokenizer()

# 모델과 토크나이저를 불러오는 함수
def load_model_and_tokenizer(model_path="kobert_chatbot_model.pth", vocab_size=8002):
    bertmodel, vocab = get_pytorch_kobert_model()
    
    # KoBERTForTextGeneration 모델 로드
    model = KoBERTForTextGeneration(bertmodel, vocab_size)
    model.load_state_dict(torch.load(model_path))
    
    # KoBERT 전용 토크나이저 로드
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    
    return model, tok, vocab

# 예측 함수 정의
def predict(model, tok, vocab, question):
    max_len = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 입력 데이터 전처리
    input_ids = [tok.convert_tokens_to_ids(tok("[CLS] " + question + " [SEP]"))]
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = (input_ids != 0).long().to(device)
    model.to(device)

    # 모델 예측
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    predicted_token_ids = torch.argmax(logits, dim=-1).cpu().numpy()
    predicted_tokens = [vocab.idx_to_token[idx] for idx in predicted_token_ids]

    return ' '.join(predicted_tokens)

