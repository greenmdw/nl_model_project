# IMDB 영화 리뷰 데이터셋 사용하여 리뷰 문장이 긍정인지 부정인지 분류
# LSTM 기반 자연어 처리 모델을 PyTorch Lightening 으로 학습
"""
설치 필요:
    pip install pytorch-lightning
    pip install torchtext==0.10.0
전체 실행 흐름
    1. 필요 패키지 불러오기
    2. torchtext field 객체 생성
    3. IMDB 데이터셋 로드
    4. 단어 사전 생성
    5. BucketIterator로 데이터 로더 생성
    6. Pytorch Lightening 기반 LSTM 모델 정의
    7. 모델 객체 생성
    8. Trainer 이용하여 모델 학습
"""

# ---------------------------------------------------------------------
# 1. 필요 라이브러리 불러오기
# ---------------------------------------------------------------------

import re
from collections import Counter

import torch
import torch.nn as nn       # 신경망 계층 만들 때
import torch.nn.functional as F     # 활성함수나 손실함수등을 함수 형태로 사용
from torch.optim import Adam        # Adma은 딥러닝 모델의 가중치를 업데이트하는 최적화 알고리즘
from torch.nn.utils.rnn import pad_sequence     # 길이가 제각각인 데이터 길이를 맞추는 기능
from torch.utils.data import DataLoader         # 우리가 지정한 크기만큼 잘라 모델에 공급

import pytorch_lightning as pl        # 학습 코드를 더 구조적으로 작성하게 해줌.
from torchtext.datasets import IMDB
from torchmetrics import F1Score, Precision, Recall

# 모든 라이브러리의 시드 42로 통일
pl.seed_everything(42, workers=True)

# ---------------------------------------------------------------------
# 2. 기본 설정 (하이퍼 파라미터)
# ---------------------------------------------------------------------
max_length = 200
batch_size = 32
embedding_dim = 300         # 하나의 단어를 몇 차원 숫자로 표현할지
hidden_size = 100            # LSTM 내부의 기억장치 크기
output_size = 2             # 긍정 부정2가지로 분류하니까
min_freq = 2                # 데이터셋 전체에서 최소 2번 이상 등장한 단어만 사전에 등록

pad_token = "<pad>"         # 문장 길이를 맞추기 위한 빈 칸 채우기용 토큰
unk_token = '<unk>'         # 사전에 없는 모르는 단어 대체할 토큰


# ---------------------------------------------------------------------
# 3. 텍스트 전처리 함수 정의
# ---------------------------------------------------------------------
def tokenizer(text):
    text = text.lower()     # 소문자로 통일
    return re.findall(r"\b\w+\b", text)     # 정규식을 이용해 알파벳/숫자 단위로 단어 추출

def label_to_index(label):
    """
    Pytorch의 Crossentropyloss는 정답 라벨이 반드시 0부터 시작하는 정수여야 함
    IBDB는 데이터가 1부터 되어 있어서 0으로 보정
    """
    return int(label) - 1       # 1(부정) , 1(긍정)


# ---------------------------------------------------------------------
# 4. IMDB 데이터셋 로드
# ---------------------------------------------------------------------
# 데이터셋 다운로드 하고, Generator 형태의 데이터를 반복문으로 쓰기 편하게 list로 변환
train_iter = list(IMDB(split='train'))
test_iter = list(IMDB(split='test'))
print('SAMPLE OF train_iter: ', train_iter[:1])
print('SAMPLE OF test_iter: ',test_iter[:1])

print(f'훈련 데이터 총 개수: {len(train_iter)}개')
print(f'테스트 데이터 총 개수: {len(test_iter)}개')

print(f'\n')
print('[실제 데이터 샘플 들여다 보기]')
pos_sample = None
neg_sample = None
for label, text in train_iter:
    if label == 2 and pos_sample is None:
        pos_sample = text
    if label == 1 and neg_sample is None:
        neg_sample = text
    if pos_sample and neg_sample:
        break
print(f"\n🟢 [긍정 리뷰 라벨]: 2 (Positive)")
print(f"💬 [리뷰 본문 (앞부분 300자)]: \n{pos_sample[:300]}...")

print(f"\n🔴 [부정 리뷰 라벨]: 1 (Negative)")
print(f"💬 [리뷰 본문 (앞부분 300자)]: \n{neg_sample[:300]}...")
# ---------------------------------------------------------------------
# 5. Vocabulary 생성
# ---------------------------------------------------------------------
counter = Counter()
for label, text in train_iter:
    # counter.update는 기존에 세어둔 단어 총합에 새로 들어온 단어 개수를 누적해 더해줌
    counter.update(tokenizer(text))     # 모든 훈련 데이터 문장을 쪼개어 단어의 빈도수 측정

# 특수 토큰(<pad>: 0 번, <unk>: 1번)을 가장 앞에 배치
# index to string 약자.
itos = [pad_token, unk_token]

# 설정한 min_frequent(2) 이상 나온 단어만 사저에 등록(노이즈 제거 및 메모리 절약)
for word, freq in counter.items():
    if freq >= min_freq:
        itos.append(word)

# 단어 -> 인덱스 형태로 매핑하는 딕셔너리 생성(컴퓨터가 읽을 수 있는 숫자로 바꾸기 위함)
# string to index 약자. 단어가 몇 번방에 있는지 물어볼 때 쓰는 도구
# enumerate(itos)는 itos 리스트를 돌면서
# (0, "<pad>"), (1 "<unk>"), (2, 'the') 처럼 방 번호랑 단어를 쌍으로 해서 꺼내줌
stoi = {word: index for index, word in enumerate(itos)}

pad_index = stoi[pad_token]         # 결과 : 0
unk_index = stoi[unk_token]         # 결과 : 1

def text_to_tensor(text):
    tokens = tokenizer(text)[:max_length]       # 설정한 최대 길이만큼 자름
    # 사전에 있으면 그 번호를, 없으면 <unk> 번호(1)을 부여
    indices = [stoi.get(token, unk_index) for token in tokens]

    if len(indices) == 0:           # 혹시 문장에 아예 아는 단어가 없으면 UNK 하나만 넣어줌
        indices = [unk_index]

    return torch.tensor(indices, dtype=torch.long)

# ---------------------------------------------------------------------
# 6. DataLoader 생성 (미니배치 구성 및 패딩)
# ---------------------------------------------------------------------
def collate_batch(batch):
    # 가장 긴 문장을 기준으로 모자란 부분을 <pad> 토큰(0)으로 채워줌
    label_list = []
    text_list = []

    # 위에서 구성한 함수대로 라벨과 문장을 이어 배치를 구성하는데,
    # 라벨은 -1 맥여서 배치에 넣고, 모르는 단어는 <unk> 번호 1 부여해 tensor값으로 넣음
    for label, text in batch:
        label_list.append(label_to_index(label))
        text_list.append(text_to_tensor(text))

    # 파이썬의 단순 정수 리스트를 PyTorch가 연산할 수 있게 1차원 행렬로 변환
    labels = torch.tensor(label_list, dtype=torch.long)

    # pad_sequence가 자동으로 가변 길이의 텐서들을 똑같은 길이로 맞춰 정렬
    # batch_first=True이면 결과 차원이 (Batch_size, Sequence_length)가 됩니다.
    texts = pad_sequence(
        text_list,
        batch_first=True,       #
        padding_value=pad_index
    )

    # 혹시 패딩된 길이가 max_length를 넘어가면 안전하게 다시 잘라줌
    if texts.size(1) > max_length:
        # : 는 전체 배치, :max_length 는 0 부터 200번째 열까지 잘라낸다는 뜻
        texts = texts[:, :max_length]

    return texts, labels

# 딥러닝 모델에 데이터를 효율적으로 나눠서 공급해주는 Dataloader 정의
train_loader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# ---------------------------------------------------------------------
# 7. LSTM 모델 클래스 정의 (PyTorch Lightening)
# ---------------------------------------------------------------------
class RNNModel(pl.LightningModule):
    def __init__(self, vocab_size, pad_index, embedding_dim=300, lstm_hidden_size=100, output_size=2 ):
        super().__init__()

        # 1) embedding layer: 단어의 정수 인덱스를 의미를 담은 밀집 벡터로 변환
        # padding_idx를 지정하면 해당 인덱스(<pad>=0)는 미분(학습)되지 않고 항상 0 벡터로 유지
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index
        )

        # 2) LSTM 레이어: 텍스트의 순서 정보를 기억하며 처리
        # batch_first=True 레이아웃 필수 설정 (입력 차원의 첫 번째가 batch size임을 명시
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        # 3) Linear 레이어: LSTM의 출력을 받아 최종 클래스 개수: (2개)
        self.lin = nn.Linear(lstm_hidden_size, output_size)

        # 다중 분류를 위한 CrossEntropyLoss (출력이 2개이므로 이진 분류 형태이지만 다중 분류로 처리 중)
        self.loss_function = nn.CrossEntropyLoss()

        # 평가 지표 설정을 위한 torchmetrics 라이브러리 사용
        from torchmetrics import Accuracy
        self.train_accuracy = Accuracy(task='multiclass', num_classes=output_size)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=output_size)
        self.train_f1 = F1Score(task='multiclass', num_classes=output_size)
        self.val_f1 = F1Score(task='multiclass', num_classes=output_size)

    def forward(self, x: torch.Tensor):
        # 1. Embedding 통과 후
        # 차원 변화 (32, 200) -> (32, 200, 300) [Batch_size, seq_len, Embedding_dim]
        x = self.embedding(x)

        # 2. LSTM 통과 후
        # LSTM은 (출력값, (최종 은닉상태, 최종 셀상태))를 반환하므로 출력값인 x만 받고 "_"는 버림.
        # 차원변화(32, 200, 300) -> (32, 200, 100) [Batch_size, seq_len, Hidden_size]
        x, _ = self.lstm(x)

        # 3. 마지막 단어 기억만 추출
        # 차원 변화 (32, 200, 100) -> (32, 100) [Batch_size, Hidden_size]
        x = x[:, -1, :]

        # 4. 활성함수 ELU 적용 (비선형성 추가)
        x = F.elu(x)

        # 4. Linear레이어 통과
        # Hidden_size 받아서  output_size(2개)로 내뱉는 함수
        x = self.lin(x)

        return x

    def training_step(self, batch, batch_idx):
        """lightening 이 제공하는 학습한 걸 자동화 함수

        """
        x, y = batch
        y_hat = self(x)     # forward 함수 호출해서 예측값 뽑음

        loss = self.loss_function(y_hat, y)     # 예측값과(y_hat) 실제값(y) 비교해 loss계산
        train_acc = self.train_accuracy(y_hat, y)       # 정확도 계산
        train_f1 = self.train_f1(y_hat, y)  # F1-score 계산 추가


        # 진행바 (prog_bar)에 로그 출력
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=True)
        self.log('train_f1_score', train_f1, prog_bar=True)

        return loss         # 이 loss값을 반환하면 뼈대 내부에서 알아서 역전파 수행


    def validation_step(self, batch, batch_idx):
        """
        검증 단계에서만 호출되는 함수. 학습은 징행되지 않고 오직 평가만 수행
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        val_acc = self.val_accuracy(y_hat, y)
        val_f1 = self.val_f1(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1_score', val_f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        어떤 최적화 알고리즘을 사용하여 가중치를 업데이트 하는지 설정
        """















