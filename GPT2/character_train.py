import torch
import re
import faiss
import numpy as np
import random
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from character_config import character_prompts
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Layer Freezing 설정 상수
num_layers_to_freeeze = 4  # 프리징할 레이어 수

# 감정 표현 필터링 함수 정의
def filter_action_description(text):
    emotion_pattern = r'\b(흥분|미소|화난|조용|불안|긴장|웃으며|울며|당황하며|슬프게|행복하게|걱정하며|환하게|열 받는 표정|한숨 쉬며|재밌다는 듯|겁에 질린)\b'

    def filter_actions(match):
        action = match.group(1)
        if re.search(emotion_pattern, action):
            return f"({action})"
        return ""

    return re.sub(r'\(([^)]*)\)', filter_actions, text)

# 하이퍼파라미터 설정 클래스
class HyperParameters:
    def __init__(self, learning_rate, batch_size, max_len=50, epochs=5):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_len = max_len
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

# 토크나이저 설정
tokenizer = PreTrainedTokenizerFast.from_pretrained('./conversation_sequences_model')

# 유사도 계산을 위한 모델 로드
similarity_model = SentenceTransformer(
    'BM-K/KoSimCSE-roberta-multitask',
    use_auth_token=True
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 데이터셋 정의
class ScriptDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len, character_prompt, prompt_usage_rate=0.3):
        """
        prompt_usage_rate: 프롬프트를 사용하여 학습할 데이터 샘플의 비율 (0.0 ~ 1.0)
        """
        self.tokenizer = tokenizer
        self.lines = lines
        self.max_len = max_len
        self.character_prompt = character_prompt
        self.prompt_usage_rate = prompt_usage_rate

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        line = filter_action_description(line)

        # 프롬프트 사용 여부 결정
        if random.random() < self.prompt_usage_rate:
            input_text = self.character_prompt + line
        else:
            input_text = line

        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=self.max_len, truncation=True, padding="max_length")
        return input_ids.squeeze(), input_ids.squeeze()

# 배치 처리 함수
def collate_batch(batch):
    data = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return data, labels

# Layer Freezing 적용: 하위 층을 고정하고 상위 층만 학습
def freeze_layers(model, num_layers_to_freeeze=num_layers_to_freeeze):
    """GPT-2 모델의 하위 층을 고정시키는 함수"""
    # 모든 층을 일단 학습 가능하게 설정
    for param in model.parameters():
        param.requires_grad = True

    # 설정된 수만큼 하위 층을 고정
    for i in range(num_layers_to_freeeze):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False  # 하위 층의 가중치를 고정

    print(f"Freezed first {num_layers_to_freeeze} layers of the model.")

# 학습 함수 정의
def fine_tune_model(hyperparams, train_dataset, character):
    model = GPT2LMHeadModel.from_pretrained('./conversation_sequences_model')
    model.to(hyperparams.device)

    # Layer Freezing 적용
    freeze_layers(model)  # num_layers_to_freeeze를 명시적으로 전달하지 않고, 상수를 사용

    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True, collate_fn=collate_batch)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparams.learning_rate, weight_decay=0.01)
    
    print(f"Starting training for {character} using device: {hyperparams.device}")
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{hyperparams.epochs}, Loss: {avg_loss:.4f}")

    model.save_pretrained(f"./character_model/{character}_model")
    tokenizer.save_pretrained(f"./character_model/{character}_model")
    print(f"Finished training for {character}.")

# FAISS 인덱스 생성 및 임베딩 사전 계산
def build_faiss_index(qa_pairs):
    questions = [pair['question'] for pair in qa_pairs]
    embeddings = similarity_model.encode(questions, convert_to_tensor=False)

    # FAISS 인덱스 생성 (내적 기반)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IndexFlatIP는 내적 기반 인덱스
    normalized_embeddings = np.array(embeddings).astype(np.float32)
    faiss.normalize_L2(normalized_embeddings)  # 코사인 유사도 계산을 위해 벡터를 정규화
    index.add(normalized_embeddings)
    return index, questions


if __name__ == "__main__":
    # 캐릭터별 최적의 하이퍼파라미터 설정
    best_hyperparameters = {
        "준하": {"learning_rate": 1e-05, "batch_size": 8, "epochs": 6},
        "해미": {"learning_rate": 1e-05, "batch_size": 8, "epochs": 6},
        "순재": {"learning_rate": 1e-05, "batch_size": 8, "epochs": 6}
    }

    print("Loading script data...")
    with open('./data/merged.txt', 'r', encoding='euc-kr', errors='ignore') as f:
        lines = f.readlines()

    # 데이터 리샘플링을 통한 학습 데이터 증강
    important_pairs = [("안녕하세요", "안녕하세요, 반갑습니다."), ("여자친구와 헤어졌어요", "힘드시겠네요. 잘 이겨내세요.")]
    resampled_lines = lines + [pair[0] + '\t' + pair[1] for pair in important_pairs for _ in range(10)]  # 중요 쌍을 10번씩 추가

    print("Script processing completed. Starting model training...")
    character_names = ["준하", "해미", "순재"]
    for character in character_names:
        print(f"Fine-tuning model for {character}...")

        # 캐릭터에 해당하는 대사만 필터링
        character_lines = []
        qa_pairs = []  # 유사도 계산을 위해 대사와 응답을 저장하는 리스트
        current_line = ""
        for line in resampled_lines:
            parts = line.split('\t')
            if len(parts) == 2 and parts[0].startswith(character):
                dialogue = filter_action_description(parts[1]).strip()
                if dialogue:
                    if current_line:
                        qa_pairs.append({'question': current_line.strip(), 'response': dialogue})
                        character_lines.append(current_line.strip())
                    current_line = dialogue
                else:
                    current_line += " " + line.strip()

        # 마지막 남은 대사 추가
        if current_line:
            character_lines.append(current_line.strip())

        if not character_lines:
            print(f"No lines found for character {character}. Skipping...")
            continue

        # 캐릭터별로 최적의 하이퍼파라미터 설정 적용
        params = best_hyperparameters.get(character, {"learning_rate": 1e-05, "batch_size": 8, "epochs": 5})  # 기본값 설정
        hyperparams = HyperParameters(
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            epochs=params['epochs']
        )

        # 캐릭터별 데이터셋 생성
        train_dataset = ScriptDataset(character_lines, tokenizer, hyperparams.max_len, character_prompts[character])

        # 모델 파인튜닝
        fine_tune_model(hyperparams, train_dataset, character)
        print(f"Model for {character} has been saved.")
        
        # FAISS 인덱스 생성 및 저장
        index, questions = build_faiss_index(qa_pairs)
        faiss.write_index(index, f"./character_model/{character}_index.faiss")
        print(f"FAISS index for {character} saved.")

    print("Model training completed for all characters.")
