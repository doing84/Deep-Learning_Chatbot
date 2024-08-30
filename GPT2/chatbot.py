import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

def load_model():
    model = GPT2LMHeadModel.from_pretrained('./conversation_sequences_model').to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained('./conversation_sequences_model')
    return model, tokenizer

def generate_response(model, tokenizer, question):
    model.eval()
    input_text = f"<usr> {question.strip()} <sys>"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
            repetition_penalty=2.0,
            top_k=50,
            top_p=0.95
        )
    
    # 디코딩 및 특수 토큰 제거
    decoded_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # <usr> 태그와 질문을 확실히 제거
    response = decoded_response.replace(f"<usr> {question.strip()} <sys>", "").strip()
    
    # 만약 질문이 포함된 부분이 여전히 있다면 제거
    if question.strip() in response:
        response = response.replace(question.strip(), "").strip()
    
    return response

def chat():
    model, tokenizer = load_model()
    print("안녕하세요 '챗봇'이에요! 종료하려면 'exit'를 입력하세요!")

    while True:
        try:
            question = input("질문: ")
            if question.lower() == "exit":
                print("안녕히계세요!")
                break
            
            answer = generate_response(model, tokenizer, question)
            print(f"답변: {answer}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue

if __name__ == "__main__":
    chat()
