import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model, get_tokenizer
import gluonnlp as nlp
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

import time
import pandas as pd

# GPU 또는 CPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저 로드
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# 데이터 로드
data = pd.read_csv('./data/ChatbotData.csv', encoding="utf-8")
chatbot_data_shuffled = data.sample(frac=1).reset_index(drop=True)

# 질문 및 라벨 추출
X = chatbot_data_shuffled['Q']
y = chatbot_data_shuffled['A']

# train/test 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 레이블 인코딩
label_encoder = {label: idx for idx, label in enumerate(set(y_train.tolist() + y_test.tolist()))}
train_labels = torch.tensor([label_encoder[label] for label in y_train])
test_labels = torch.tensor([label_encoder[label] for label in y_test])

# 입력 데이터 전처리 함수 정의
def preprocess_data(X, tok, max_len=64):
    input_ids = [tok.convert_tokens_to_ids(tok("[CLS] " + str(sentence) + " [SEP]")) for sentence in X]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]
    return torch.tensor(input_ids), torch.tensor(attention_masks)

train_inputs, train_masks = preprocess_data(X_train, tok)
test_inputs, test_masks = preprocess_data(X_test, tok)

# TensorDataset 생성
train_data = TensorDataset(train_inputs, train_masks, train_labels)
test_data = TensorDataset(test_inputs, test_masks, test_labels)

# DataLoader 생성
batch_size = 32
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

# 모델 학습 및 검증
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=len(label_encoder))  
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        logits = model(b_input_ids, attention_mask=b_input_mask).logits
        loss = loss_fn(logits, b_labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch + 1}/{epochs} | Step {step}/{len(train_dataloader)} | Loss: {loss.item()} | Elapsed: {elapsed:.2f}s")
            start_time = time.time()

    print(f"Epoch {epoch + 1}/{epochs} | Average training loss: {total_loss / len(train_dataloader)}")

# 모델 평가 및 정확도 계산
model.eval()
correct_predictions = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        logits = model(b_input_ids, attention_mask=b_input_mask).logits
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(predictions == b_labels).item()

accuracy = correct_predictions / len(test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 모델과 인코더 저장
torch.save(model.state_dict(), "kobert_chatbot_model.pth")
