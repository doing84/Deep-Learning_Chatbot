import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
import random
from sklearn.metrics.pairwise import cosine_similarity
from character_config import character_patterns, character_prompts

# 캐릭터별 설정 (이름과 프롬프트)
character_names = ["준하", "해미", "순재"]

# KoBERT 모델과 토크나이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kobert_model = AutoModel.from_pretrained('skt/kobert-base-v1').to(device)
kobert_tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')

# 각 캐릭터별 최적의 하이퍼파라미터 설정
best_hyperparameters = {
    "준하": {"learning_rate": 1e-05, "batch_size": 8, "epochs": 6},
    "해미": {"learning_rate": 1e-05, "batch_size": 8, "epochs": 6},
    "순재": {"learning_rate": 1e-05, "batch_size": 8, "epochs": 6},
}

# 감정 표현 필터링 함수 정의
def filter_action_description(text):
    return re.sub(r'\((.*?)\)', r'\1', text)

# GPU 사용 여부를 처음에 한 번만 출력하기 위한 상태 변수
similarity_device_printed = False

# 문장의 유사도를 계산하는 함수 (GPU에서 배치 처리 적용)
def calculate_similarity(batch_texts1, batch_texts2):
    global similarity_device_printed
    batch_size = len(batch_texts1)
    
    # 디바이스 출력은 처음 한 번만 실행되도록 함
    if not similarity_device_printed:
        print(f"Using device for similarity calculation: {device}")
        similarity_device_printed = True

    # 텍스트를 한번에 토큰화하여 텐서로 변환
    inputs1 = kobert_tokenizer(batch_texts1, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)
    inputs2 = kobert_tokenizer(batch_texts2, return_tensors='pt', padding=True, truncation=True, max_length=64).to(device)

    # token_type_ids 제거
    inputs1.pop("token_type_ids", None)
    inputs2.pop("token_type_ids", None)

    try:
        # 배치 처리로 유사도 계산
        with torch.no_grad():
            outputs1 = kobert_model(**inputs1).last_hidden_state.mean(dim=1)
            outputs2 = kobert_model(**inputs2).last_hidden_state.mean(dim=1)
        similarities = cosine_similarity(outputs1.cpu().numpy(), outputs2.cpu().numpy())
        return similarities.diagonal()  # 대각 요소만 반환하여 각 쌍의 유사도 계산
    except RuntimeError as e:
        print(f"Error during similarity calculation: {e}. Trying to reduce batch size or reset the environment.")
        return [0] * batch_size  # 유사도가 계산되지 않을 때 기본값 반환

# 대본 형식의 데이터를 연속적인 대화 흐름으로 변환하는 함수
def preprocess_script_for_dialogue(lines, similarity_threshold=0.7, batch_size=10, context_window=3):
    processed_pairs = []
    current_speaker = None
    current_lines = []

    print("Starting to process script lines...")
    batch_texts1, batch_texts2 = [], []  # 유사도 계산을 위한 배치 처리 리스트
    indices = []  # 배치에서의 인덱스 저장
 
    for i, line in enumerate(tqdm(lines, desc="Processing Script Lines")):
        match = re.match(r'(\w+)\s*[:\t]\s*(.*)', line)
        if match:
            speaker, dialogue = match.groups()
            dialogue = re.sub(r'\s+', ' ', dialogue).strip()
            dialogue = filter_action_description(dialogue)

            if current_speaker:
                # 현재 대사 앞의 context_window 만큼의 대사를 모아 context로 사용
                context_dialogue = ' '.join(current_lines[-context_window:])  # 이전 대사 2~3개 합치기
                batch_texts1.append(context_dialogue)
                batch_texts2.append(dialogue)
                indices.append((current_speaker, speaker, context_dialogue, dialogue))

                if len(batch_texts1) == batch_size:
                    similarities = calculate_similarity(batch_texts1, batch_texts2)
                    for (curr_speaker, next_speaker, prev_dialogue, next_dialogue), sim in zip(indices, similarities):
                        if sim > similarity_threshold:
                            input_text = f"{curr_speaker}: {prev_dialogue}"
                            output_text = f"{next_speaker}: {next_dialogue}"
                            processed_pairs.append((input_text, output_text))
                    batch_texts1, batch_texts2, indices = [], [], []

            current_speaker = speaker
            current_lines.append(dialogue)

    # 남아있는 배치 처리
    if batch_texts1:
        similarities = calculate_similarity(batch_texts1, batch_texts2)
        for (curr_speaker, next_speaker, prev_dialogue, next_dialogue), sim in zip(indices, similarities):
            if sim > similarity_threshold:
                input_text = f"{curr_speaker}: {prev_dialogue}"
                output_text = f"{next_speaker}: {next_dialogue}"
                processed_pairs.append((input_text, output_text))

    print("Finished processing script lines.")
    return processed_pairs


    # 남아있는 배치 처리
    if batch_texts1:
        similarities = calculate_similarity(batch_texts1, batch_texts2)
        for (curr_speaker, next_speaker, prev_dialogue, next_dialogue), sim in zip(indices, similarities):
            if sim > similarity_threshold:
                input_text = f"{curr_speaker}: {prev_dialogue}"
                output_text = f"{next_speaker}: {next_dialogue}"
                processed_pairs.append((input_text, output_text))

    print("Finished processing script lines.")
    return processed_pairs

# 하이퍼파라미터 설정 클래스
class HyperParameters:
    def __init__(self, learning_rate, batch_size, max_len=60, epochs=7, weight_decay=0.01, max_grad_norm=0.5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for model training: {self.device}")

# 토크나이저 설정 (conversation_sequences_model 사용)
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
    data = torch.stack([item[0] for item in batch]).to(device)
    labels = torch.stack([item[1] for item in batch]).to(device)
    return data, labels

# 학습 함수 정의
def fine_tune_model(hyperparams, train_dataset, character):
    model = GPT2LMHeadModel.from_pretrained('./conversation_sequences_model')
    model.to(hyperparams.device)
    print(f"Starting training for {character} using device: {hyperparams.device}")
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(hyperparams.epochs):
        model.train()
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
    print(f"Finished training for {character}.")

if __name__ == "__main__":
    print("Loading script data...")
    with open('./data/merged.txt', 'r', encoding='euc-kr', errors='ignore') as f:
        lines = preprocess_script_for_dialogue(f.readlines(), similarity_threshold=0.6)

    print("Script processing completed. Starting model training...")
    for character in character_names:
        print(f"Fine-tuning model for {character}...")
        dialogue_pairs = [pair for pair in tqdm(lines, desc=f"Processing {character}'s lines") if pair[0].startswith(character)]
        if not dialogue_pairs:
            print(f"No lines found for character {character}. Skipping...")
            continue
        
        # 캐릭터별로 최적의 하이퍼파라미터를 설정
        params = best_hyperparameters[character]
        hyperparams = HyperParameters(
            learning_rate=params['learning_rate'], 
            batch_size=params['batch_size'], 
            epochs=params['epochs']
        )
        
        train_dataset = ScriptDataset(dialogue_pairs, tokenizer, hyperparams.max_len)
        fine_tune_model(hyperparams, train_dataset, character)
        print(f"Model for {character} has been saved.")
    print("Model training completed for all characters.")
