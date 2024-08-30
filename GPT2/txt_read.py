import os
import chardet

# folder_path = './data/kick'  # 폴더 경로
# output_file = './data/merged.txt'  # 합칠 파일 이름

# # 모든 .txt 파일을 합칠 파일을 엽니다.
# with open(output_file, 'w', encoding='euc-kr') as outfile:
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(folder_path, filename)
#             with open(file_path, 'rb') as f:
#                 raw_data = f.read()
#                 result = chardet.detect(raw_data)
#                 detected_encoding = result['encoding']
                
#             try:
#                 with open(file_path, 'r', encoding=detected_encoding) as infile:
#                     outfile.write(infile.read() + '\n')  # 각 파일 내용을 추가하고 개행을 넣습니다.
#             except UnicodeDecodeError:
#                 print(f"Failed to decode file: {filename} with detected encoding: {detected_encoding}.")
