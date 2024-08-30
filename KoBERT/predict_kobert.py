import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
from transformers import BertTokenizer, BertForSequenceClassification

# 모델, 토크나이저 로드
model = BertForSequenceClassification.from_pretrained('monologg/kobert')
model.load_state_dict(torch.load("kobert_chatbot_model.pth"))
model.eval()

# KoBERT 전용 토크나이저 로드
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(model, tok, question):
    max_len = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = [tok.convert_tokens_to_ids(tok("[CLS] " + question + " [SEP]"))]
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = (input_ids != 0).long().to(device)
    model.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    label_id = torch.argmax(outputs.logits, dim=1).item()
    return label_id

while True:
    question = input("질문: ")
    if question.lower() in ['exit', 'quit', 'q']:
        print("종료합니다.")
        break

    label_id = predict(model, tok, question)
    print(f"답변 라벨: {label_id}")
