import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel  # KoGPT2를 위한 import
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split  # train_test_split import
from tqdm import tqdm
import optuna

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU를 사용합니다.")
else:
    device = torch.device("cpu")
    print("CPU를 사용합니다.")


# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


# pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116