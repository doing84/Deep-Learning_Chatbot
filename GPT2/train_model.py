import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

# 하이퍼파라미터 설정
class HyperParameters:
    def __init__(self, learning_rate=3e-5, batch_size=32, max_len=40, epochs=10, weight_decay=0.01, max_grad_norm=1.0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")  # GPU 사용 여부 출력

# 토크나이저 설정
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>',
    mask_token='<mask>'
)

# 데이터셋 정의
class ChatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.q_token = "<usr>"
        self.a_token = "<sys>"
        self.eos = "</s>"
        self.sent_token = "<unused1>"
        self.mask = "<unused0>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        turn = self.data.iloc[index]
        q = turn["Q"]
        a = turn["A"]

        # 전처리: 구두점 제거
        q = re.sub(r"([?.!,])", r" ", q)
        a = re.sub(r"([?.!,])", r" ", a)

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)

        q_len = len(q_toked)
        a_len = len(a_toked)

        if q_len > self.max_len:
            q_toked = q_toked[-self.max_len:]
            q_len = len(q_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]

        labels = [self.mask] * q_len + a_toked[1:]

        # Mask와 Padding
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)

        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return torch.LongTensor(token_ids), torch.LongTensor(mask), torch.LongTensor(labels_ids)

# 배치 처리 함수
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.stack(data), torch.stack(mask), torch.stack(label)

# 학습 함수 정의
def train_model(hyperparams, train_df, val_df):
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.to(hyperparams.device)

    # 데이터 로더
    train_dataset = ChatDataset(train_df, tokenizer, max_len=hyperparams.max_len)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_batch)

    val_dataset = ChatDataset(val_df, tokenizer, max_len=hyperparams.max_len)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, collate_fn=collate_batch)

    # 옵티마이저와 손실 함수 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # 학습 루프
    for epoch in range(hyperparams.epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{hyperparams.epochs}"):
            optimizer.zero_grad()
            token_ids, mask, labels = batch
            token_ids, mask, labels = token_ids.to(hyperparams.device), mask.to(hyperparams.device), labels.to(hyperparams.device)

            outputs = model(token_ids)
            logits = outputs.logits
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=logits.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, logits, torch.tensor(-1e18).to(hyperparams.device))

            loss = criterion(mask_out.transpose(2, 1), labels)
            avg_loss = loss.sum() / mask.sum()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams.max_grad_norm)
            optimizer.step()

            epoch_loss += avg_loss.item()

            # 정확도 계산
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()

        avg_train_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{hyperparams.epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy:.4f}")

        # 검증
        model.eval()
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch + 1}/{hyperparams.epochs}"):
                token_ids, mask, labels = batch
                token_ids, mask, labels = token_ids.to(hyperparams.device), mask.to(hyperparams.device), labels.to(hyperparams.device)

                outputs = model(token_ids)
                logits = outputs.logits
                mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=logits.shape[2], dim=2)
                mask_out = torch.where(mask_3d == 1, logits, torch.tensor(-1e18).to(hyperparams.device))

                loss = criterion(mask_out.transpose(2, 1), labels)
                avg_loss = loss.sum() / mask.sum()

                val_loss += avg_loss.item()

                # 검증 정확도 계산
                predictions = torch.argmax(logits, dim=-1)
                val_correct_predictions += (predictions == labels).sum().item()
                val_total_predictions += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions
        print(f"Epoch {epoch + 1}/{hyperparams.epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 모델 저장
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")
    return avg_val_loss

if __name__ == "__main__":
    # 데이터 로드 및 분할
    df = pd.read_csv('./data/ChatbotData.csv')
    
    # 감성대화말뭉치 엑셀 파일 로드 및 전처리
    emotion_df = pd.read_excel('./data/감성대화말뭉치(최종데이터)_Training.xlsx')

    # Q와 A에 해당하는 컬럼들을 모두 사용하여 데이터프레임 생성
    qa_pairs = []

    # 각 사람 문장과 대응하는 시스템 문장을 매핑
    for i, row in emotion_df.iterrows():
        for q_col, a_col in zip(["사람문장1", "사람문장2", "사람문장3"],
                                ["시스템문장1", "시스템문장2", "시스템문장3"]):
            if pd.notna(row[q_col]) and pd.notna(row[a_col]):
                qa_pairs.append([row[q_col], row[a_col]])

    # 데이터프레임으로 변환
    emotion_df_processed = pd.DataFrame(qa_pairs, columns=["Q", "A"])

    # 기존의 ChatbotData.csv와 병합
    combined_df = pd.concat([df, emotion_df_processed], ignore_index=True)

    # 데이터셋 분할
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    # 하이퍼파라미터 설정
    hyperparams = HyperParameters()

    # 최종 모델 학습
    final_loss = train_model(hyperparams, train_df, val_df)
    print("Final loss:", final_loss)

