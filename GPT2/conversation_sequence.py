import pandas as pd

# CSV 파일 병합
files = [
    './data/건강_및_식음료.csv',
    './data/문화_생활_및_여가.csv',
    './data/여행_관광_및_명소.csv',
    './data/경제_및_사회.csv'
]

all_data = []

for file in files:
    df = pd.read_csv(file)
    all_data.append(df)

merged_df = pd.concat(all_data, ignore_index=True)

# 대화 데이터를 시퀀스로 변환
def create_conversation_sequence(df):
    conversations = []
    current_conversation_id = None
    conversation = ""
    
    for _, row in df.iterrows():
        conversation_id = row['대화ID']

        # 새로운 대화 ID가 시작되면 이전 대화를 저장하고 초기화
        if conversation_id != current_conversation_id:
            if conversation:
                conversations.append(conversation.strip())
            conversation = ""
            current_conversation_id = conversation_id

        # A 또는 B 발화자를 구분하여 대화 시퀀스 생성
        if pd.notna(row['발화']) and pd.notna(row['발화자']):
            if row['발화자'] == 'A':
                conversation += f"<usr> {row['발화']} "
            elif row['발화자'] == 'B':
                conversation += f"<sys> {row['발화']} "

    # 마지막 대화 추가
    if conversation:
        conversations.append(conversation.strip())

    return conversations

# 병합된 데이터로부터 시퀀스를 생성
conversation_sequences = create_conversation_sequence(merged_df)

# DataFrame으로 변환
sequence_df = pd.DataFrame(conversation_sequences, columns=["conversation"])

# 병합된 데이터 저장
sequence_df.to_csv('./data/merged_conversation_sequences.csv', index=False)

print(sequence_df)
