import os
import torch
import faiss
import numpy as np
import random
import re
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from character_config import character_patterns, character_prompts

# 모델 캐싱을 위한 전역 딕셔너리
model_cache = {}

# 캐릭터별 최적의 모델 경로 설정
character_models = {
    "준하": './character_model/준하_model',
    "해미": './character_model/해미_model',
    "순재": './character_model/순재_model',
    "상담챗봇": './conversation_sequences_model',
}

# 유사도 계산을 위한 모델 로드
similarity_model = SentenceTransformer(
    'BM-K/KoSimCSE-roberta-multitask',
    use_auth_token=True
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 유사한 응답 찾기 함수
def find_similar_response(input_text, index, questions, threshold=0.6):
    input_embedding = similarity_model.encode([input_text], convert_to_tensor=False).astype(np.float32)
    faiss.normalize_L2(input_embedding)

    D, I = index.search(input_embedding, 1)

    if len(D[0]) > 0:
        best_similarity = D[0][0]
        best_response = questions[I[0][0]]
        print(f"유사한 문장: '{best_response}', 유사도: {best_similarity:.2f}")

        if best_similarity > threshold:
            return best_response, best_similarity
    else:
        print("검색 결과가 없습니다. 임베딩 계산 오류일 수 있습니다.")

    return None, 0

def load_model(character):
    model_path = character_models.get(character, None)
    if not model_path:
        print(f"{character}에 해당하는 모델 경로가 없습니다.")
        return None, None, [], None, []

    if model_path in model_cache:
        print(f"캐싱된 {character} 모델을 불러옵니다.")
        return model_cache[model_path]

    if not os.path.exists(model_path):
        print(f"모델 경로가 존재하지 않습니다: {model_path}")
        return None, None, [], None, []

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        print(f"Loaded {character} model from {model_path} using device: {device}")

        # 상담챗봇에 대해서는 유사도 계산을 생략
        if character == "상담챗봇":
            print(f"{character} 모델을 로드합니다.")
            common_patterns = character_patterns.get(character, [])
            model_cache[model_path] = (model, tokenizer, common_patterns, None, None)
            return model, tokenizer, common_patterns, None, None

        # 유사도 계산이 필요한 다른 캐릭터의 경우
        index = faiss.read_index(f"./character_model/{character}_index.faiss")
        with open(f"./character_model/{character}_questions.txt", 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]

        common_patterns = character_patterns.get(character, [])
        model_cache[model_path] = (model, tokenizer, common_patterns, index, questions)

        return model, tokenizer, common_patterns, index, questions
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None, None, [], None, []

def filter_repeated_phrases(text, max_repeats=2):
    phrases = text.split()
    seen_phrases = {}
    filtered_phrases = []

    for phrase in phrases:
        if phrase in seen_phrases:
            seen_phrases[phrase] += 1
            if seen_phrases[phrase] <= max_repeats:
                filtered_phrases.append(phrase)
        else:
            seen_phrases[phrase] = 1
            filtered_phrases.append(phrase)

    return ' '.join(filtered_phrases)

def remove_redundant_sentences(text):
    sentences = text.split('. ')
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    return '. '.join(unique_sentences)

def convert_to_informal(text):
    replacements = {
        "요?": "냐?",
        "요.": ".",
        "어요": "어",
        "세요": "어",
        "워요": "워",
        "있나요": "있냐",
        "있으셨나요": "있었냐",
        "가세요": "가라",
        "했어요": "했다",
        "이세요": "이냐",
        "신가요": "냐",
        "보이세요": "보여",
        "바랄게요": "바랄게",
        "있으신가요": "있냐",
        "이시네요": "이네"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def remove_partial_question_from_response(response, question):
    question_words = set(question.split())
    response_words = response.split()
    filtered_response = [word for word in response_words if word not in question_words]
    return ' '.join(filtered_response)

def generate_response(model, tokenizer, question, common_patterns, character, index, questions):
    # 상담 챗봇은 유사도 기반 응답 생성 과정을 건너뛰도록 설정
    if character == "상담챗봇":
        input_text = question
    else:
        similar_response, similarity = find_similar_response(question, index, questions)

        if similar_response:
            print(f"유사한 응답 발견 (유사도: {similarity:.2f}): {similar_response}")
            input_text = similar_response
        else:
            input_text = question

    response = generate_character_response(model, tokenizer, input_text, character, max_length=150)

    # 60% 확률로 캐릭터의 추임새를 응답에 추가
    if random.random() < 0.6 and common_patterns:
        random_pattern = random.choice(common_patterns)
        response = f"{response} {random_pattern}"

    response = remove_partial_question_from_response(response, question)
    response = remove_redundant_sentences(response)
    if character == "순재":
        response = convert_to_informal(response)

    response = filter_repeated_phrases(response)

    if not response.strip() and common_patterns:
        random_pattern = random.choice(common_patterns)
        response = random_pattern

    return response

def generate_character_response(model, tokenizer, input_text, character, max_length=100, use_prompt=True):
    if use_prompt:
        character_prompt = character_prompts.get(character, "")
        input_text = character_prompt + input_text

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
            repetition_penalty=1.5,
            top_k=50,
            top_p=0.85,
            temperature=0.85,
            num_beams=5,
            early_stopping=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # 응답에서 프롬프트 제거
    if use_prompt and character_prompt in response:
        response = response.replace(character_prompt, "").strip()

    # 괄호 안의 내용을 제거 (모든 종류의 괄호를 포함)
    response = re.sub(r'\([^)]*\)', '', response).strip()
    response = re.sub(r'\[[^]]*\]', '', response).strip()
    response = re.sub(r'\{[^}]*\}', '', response).strip()

    return response

def chat():
    character = None
    model = None
    tokenizer = None
    common_patterns = []
    index = None
    questions = []

    while True:
        if not model or not tokenizer:
            print("사용할 캐릭터를 선택하세요: 준하, 해미, 순재, 상담챗봇")
            character = input("캐릭터: ").strip()
            model, tokenizer, common_patterns, index, questions = load_model(character)
            if model is None:
                print("모델을 불러올 수 없습니다. 다른 캐릭터를 선택해주세요.")
                continue

        question = input("질문 ('exit' to quit, 'change' to switch character): ").strip()
        if question.lower() == "exit":
            print("안녕히계세요!")
            break
        elif question.lower() == "change":
            model, tokenizer, common_patterns, index, questions = None, None, [], None, []
            print("캐릭터를 변경합니다.")
            continue

        answer = generate_response(model, tokenizer, question, common_patterns, character, index, questions)  
        print(f"답변 ({character}): {answer}")

if __name__ == "__main__":
    print("안녕하세요 '챗봇'이에요! 종료하려면 'exit'를 입력하세요, 캐릭터를 변경하려면 'change'를 입력하세요.")
    chat()
