import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
import optuna

# 하이퍼파라미터 설정 클래스
class HyperParameters:
    def __init__(self, learning_rate=3e-5, batch_size=32, max_len=40, epochs=10, weight_decay=0.01, max_grad_norm=1.0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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

    train_dataset = ChatDataset(train_df, tokenizer, max_len=hyperparams.max_len)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_batch)

    val_dataset = ChatDataset(val_df, tokenizer, max_len=hyperparams.max_len)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, collate_fn=collate_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

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
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.numel()

        avg_train_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch + 1}/{hyperparams.epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {accuracy:.4f}")

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

                predictions = torch.argmax(logits, dim=-1)
                val_correct_predictions += (predictions == labels).sum().item()
                val_total_predictions += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions
        print(f"Epoch {epoch + 1}/{hyperparams.epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    model.save_pretrained("./model_optuna")
    tokenizer.save_pretrained("./model_optuna")
    return avg_val_loss

# Optuna를 사용하여 최적의 하이퍼파라미터 찾기
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)  # 학습률 범위를 좁힘
    batch_size = trial.suggest_categorical('batch_size', [16, 24, 32])  # 배치 크기 범위 축소
    max_len = trial.suggest_int('max_len', 40, 80)  # max_len 범위 축소

    hyperparams = HyperParameters(
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_len=max_len,
        epochs=4  # 테스트 목적으로 에포크를 줄였습니다.
    )

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    avg_val_loss = train_model(hyperparams, train_df, val_df)
    return avg_val_loss

if __name__ == "__main__":
    df = pd.read_csv('./data/ChatbotData.csv')
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    best_params = study.best_params
    print("Best params:", best_params)

    final_hyperparams = HyperParameters(
        learning_rate=best_params['learning_rate'],
        batch_size=best_params['batch_size'],
        max_len=best_params['max_len'],
        epochs=10  # 최종 학습에 사용할 에포크 수
    )

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    final_loss = train_model(final_hyperparams, train_df, val_df)
    print("Final loss:", final_loss)
