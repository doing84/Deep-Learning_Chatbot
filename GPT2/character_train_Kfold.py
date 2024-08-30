import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import random
from sklearn.model_selection import KFold  # KFold 추가
from character_config import character_patterns, character_prompts 

# 캐릭터별 설정 (이름과 프롬프트)
character_names = ["준하", "해미", "순재"]

# 감정 표현 필터링 함수 정의
def filter_action_description(text):
    return re.sub(r'\((.*?)\)', r'\1', text)

# 대본 형식의 데이터를 전처리하여 학습에 사용할 수 있는 형식으로 변환하는 함수
def preprocess_script_for_dialogue(lines):
    processed_pairs = []
    current_speaker = None
    current_lines = []
    
    for line in lines:
        match = re.match(r'(\w+)\s*[:\t]\s*(.*)', line)
        if match:
            speaker, dialogue = match.groups()
            dialogue = re.sub(r'\s+', ' ', dialogue).strip()
            dialogue = filter_action_description(dialogue)

            if current_speaker:
                input_text = f"{current_speaker}: {current_lines[-1]}"
                output_text = f"{speaker}: {dialogue}"
                processed_pairs.append((input_text, output_text))
            
            current_speaker = speaker
            current_lines.append(dialogue)
    
    return processed_pairs

# 하이퍼파라미터 설정
class HyperParameters:
    def __init__(self, learning_rate, batch_size, max_len=60, epochs=6, weight_decay=0.01, max_grad_norm=0.5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

# 토크나이저 설정
tokenizer = PreTrainedTokenizerFast.from_pretrained('./conversation_sequences_model')

# 데이터셋 정의
class ScriptDataset(Dataset):
    def __init__(self, dialogue_pairs, tokenizer, max_len):
        self.dialogue_pairs = dialogue_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dialogue_pairs)

    def __getitem__(self, index):
        input_text, output_text = self.dialogue_pairs[index]
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=self.max_len, truncation=True, padding="max_length")
        output_ids = self.tokenizer.encode(output_text, return_tensors='pt', max_length=self.max_len, truncation=True, padding="max_length")
        return input_ids.squeeze(), output_ids.squeeze()

# 배치 처리 함수
def collate_batch(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return data, labels

# 학습 및 평가 함수
def train_and_evaluate(hyperparams, train_dataset, val_dataset):
    model = GPT2LMHeadModel.from_pretrained('./conversation_sequences_model')
    model.to(hyperparams.device)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams.batch_size, shuffle=False, collate_fn=collate_batch)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(hyperparams.epochs):
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{hyperparams.epochs}", leave=False):
            optimizer.zero_grad()
            input_ids, labels = batch
            input_ids, labels = input_ids.to(hyperparams.device), labels.to(hyperparams.device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams.max_grad_norm)
            optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids, labels = batch
            input_ids, labels = input_ids.to(hyperparams.device), labels.to(hyperparams.device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# 최적의 하이퍼파라미터 찾기
def find_best_hyperparameters(dialogue_pairs):
    learning_rates = [1e-5, 5e-5]
    batch_sizes = [4, 8]
    epoch_nums = [5, 10]
    
    best_loss = float('inf')
    best_hyperparams = None

    kf = KFold(n_splits=5)

    for lr in learning_rates:
        for bs in batch_sizes:
            for epochs in epoch_nums:
                total_loss = 0
                hyperparams = HyperParameters(learning_rate=lr, batch_size=bs, epochs=epochs)
                
                for train_index, val_index in tqdm(kf.split(dialogue_pairs), desc=f"LR={lr}, BS={bs}, Epochs={epochs}"):
                    train_pairs = [dialogue_pairs[i] for i in train_index]
                    val_pairs = [dialogue_pairs[i] for i in val_index]
                    
                    train_dataset = ScriptDataset(train_pairs, tokenizer, hyperparams.max_len)
                    val_dataset = ScriptDataset(val_pairs, tokenizer, hyperparams.max_len)
                    
                    loss = train_and_evaluate(hyperparams, train_dataset, val_dataset)
                    total_loss += loss
                
                avg_loss = total_loss / kf.get_n_splits()
                print(f"Avg Loss for LR={lr}, Batch Size={bs}, Epochs={epochs}: {avg_loss:.4f}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_hyperparams = (lr, bs, epochs)
    
    print(f"Best Hyperparameters: LR={best_hyperparams[0]}, Batch Size={best_hyperparams[1]}, Epochs={best_hyperparams[2]}, with Loss={best_loss:.4f}")
    return best_hyperparams

# 모델 최종 학습 함수
def fine_tune_model(hyperparams, train_dataset, character):
    model = GPT2LMHeadModel.from_pretrained('./conversation_sequences_model')
    model.to(hyperparams.device)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(hyperparams.epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Fine-tuning {character} - Epoch {epoch + 1}/{hyperparams.epochs}"):
            optimizer.zero_grad()
            input_ids, labels = batch
            input_ids, labels = input_ids.to(hyperparams.device), labels.to(hyperparams.device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams.max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{hyperparams.epochs}, Loss: {avg_loss:.4f}")
    
    model.save_pretrained(f"./character_model/{character}_model")
    tokenizer.save_pretrained(f"./character_model/{character}_model")

if __name__ == "__main__":
    with open('./data/merged.txt', 'r', encoding='euc-kr', errors='ignore') as f:
        lines = preprocess_script_for_dialogue(f.readlines())

    for character in character_names:
        print(f"Finding best hyperparameters for {character}...")
        dialogue_pairs = [pair for pair in lines if pair[0].startswith(character)]
        if not dialogue_pairs:
            print(f"No lines found for character {character}. Skipping...")
            continue

        best_hyperparams = find_best_hyperparameters(dialogue_pairs)

        # 최적의 하이퍼파라미터로 모델을 최종 학습
        final_hyperparams = HyperParameters(
            learning_rate=best_hyperparams[0], 
            batch_size=best_hyperparams[1], 
            epochs=best_hyperparams[2]
        )
        
        train_dataset = ScriptDataset(dialogue_pairs, tokenizer, final_hyperparams.max_len)
        fine_tune_model(final_hyperparams, train_dataset, character)
        print(f"Final model for {character} has been saved.")
