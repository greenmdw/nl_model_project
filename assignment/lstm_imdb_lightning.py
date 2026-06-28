# -*- coding: utf-8 -*-
"""
동원 assignment
"""
# 0. import
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightening as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
from collections import Counter

# 1. 기본 설정 (하이퍼파라미터 및 경로)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ratings.txt')

BATCH_SIZE = 65
MAX_LENGTH = 50
EMBEDDING_DIM = 200
LSTM_HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 5

# 2. 한글 데이터 전처리 및 전용 DATASET 클래스 정으
class RatingsDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_vocab_size=20000):
        self.labels = []
        self.texts = []

        # 파일 읽기 ('\t' 탭 구분자)
        with open(data_path, 'r', encoding='utf-8') as f:
            next(f)             # 헤더 줄 skip
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:         # 3개 조각으로 쪼개진 것이 정상 데이터(id, text, label)
                    _, document, label = parts
                    if document.strip():        # 빈 문자열 제외
                        self.labels.append(int(label))
                        self.texts.append(document.split())         # 기본 공백 토큰화

        # 단어 사전(vocab) 구축
        if vocab is None:
            # self.text에 담긴 모든 문장을 1차원 리스트로 쫙 펼침
            all_tokens = [token for text in self.texts for token in text]
            # Counter를 써서 각 단어가 전체에 몇 번 등장했는지 빈도수 체크
            token_counts = Counter(all_tokens)
            vocab_list = [token for token, _ in token_counts.most_common(max_vocab_size)]

            # 인덱스 부여하는 딕셔너리
            # 0, 1은 특수 토큰용이므로 일반 단어는 +2로 저장
            self.vocab = {token: idx + 2 for idx, token in enumerate(vocab_list)}
            # 특수 토큰 지정 (<pad>=0 : 가짜단어 , <unk>=1: 사전에 없는 단어)
            self.vocab['<pad>'] = 0
            self.vocab['<unk>'] = 1

        else:
            # 검증 테스트 할 때 사용
            self.vocab = vocab

        # 자주 쓰는 특수 토큰의 번호는 꺼내기 쉽게 미리 변수로 등록
        self.pad_index = self.vocab['<pad>']
        self.unk_index = self.vocab['<unk>']

    def __len__(self):
        """총 리뷰가 몇 개인지 개수 반환"""
        return len(self.labels)

    def text_to_tensor(self, tokenized_text):
        """vocab에 있으면 고유번호, 없으면 unk"""
        indexed = [self.vocab.get(token, self.unk_index) for token in tokenized_text]

        # 파이썬 리스트를 PyTorch 모델이 연산할 수 있게 정수형 텐서로 변환
        return torch.tensor(indexed, dtype=torch.long)

    def __getitem__(self, idx):
        """DataLoader가 요청한 idx에 있는 값 꺼내주는 것"""
        return self.labels[idx], self.text_to_tensor(self.texts[idx])

# 3. DataLoader를 위한 collate_fn (동적 패딩)
def collate_batch(batch):
    """
    DataLoader가 64개 샘플을 무작위로 뽑아와서 이 함수에 실행
    64개의 데이터를 행렬로 조립
    """
    label_list = []
    text_list = []

    # 1. 튜플 데이터를 각각의 리스트로 분리
    for label, text_tensor in batch:
        label_list.append(label)
        text_list.append(text_tensor)

    # 2. 라벨을 행렬로 통합
    # 파이썬 리스트였던 라벨들을 1차원 텐서로
    labels = torch.tensor(label_list, dtype=torch.long)

    # 3. 동적 패딩(Dynamic Padding)
    # 64개 문장중 (이번 배치내에서) 가장 긴 문장을 찾음
    # 보다 짧은 문장들은 <pad> 로 채움
    texts = pad_sequence(text_list, batch_first=True, padding_value=pad_index)

    # 4. 과도하게 긴 문장 차단
    if texts.size(1) > MAX_LENGTH:
        texts = texts[:, :MAX_LENGTH]

    # 완성된 텍스트 행렬과 정답 행렬을 튜플 형태로 리턴
    return texts, labels

# 공급망 배치 인프라 구축
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
# 검증용 데이터 로더
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# 4. RatingsLSTMModel 정의
class RatingsLSTMModel(pl.LighteningModule):
    def __init__(self, vocab_size, pad_index, embedding_dim=200, lstm_hidden_size=100, output_size=2):
        super().__init__()

        # 하이퍼파라미터를 모델 내부에 자동으로 저장하여 나중에 꺼내쓰기 쉽게
        self.save_hyperparameters()

        # 1. embedding 레이어: 단어 번호를 의미를 가진 밀집 벡터로 변환
        # padding_idx=pad_index를 주어 0번 토큰은 학습 과정에서 무시하고 항상 0 벡터로 유지
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index
        )

        # 2. LSTM 레이어: 문장을 앞에서부터 읽으며 어순과 문맥 정보를 기억 공간에 누적
        # Batch_first=true 레이아웃 필수. 입력 데이터의 첫 번째 축이 배치 크기임을 명시
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        # 3. Linear 레이어: LSTM이 최종 요약한 100 차원 문맥을 받아 클래스 개체로 변환
        self.lin = nn.Linear(lstm_hidden_size, output_size)

        # 오차를 계산할 손실 함수 (이진 분류이므로 CrossEntorpyLoss)
        self.loss_function = nn.CrossEntropyLoss()

        # 4. 훈련 및 검증용 지표 설정
        # task='multiclass'와 클래스 개수(2)를 지정하여 0과 1 분류하는 지표 세팅
        self.train_accuracy = Accuracy(task='multiclass', num_classes=output_size)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=output_size)
        self.val_f1 = F1Score(task='multiclass', num_classes=output_size)
        self.val_precision = Precision(task='multiclass', num_classes=output_size)
        self.val_recall = Recall(task='multiclass', num_classes=output_size)

    def forward(self, x: torch.Tensor):

        x = self.embedding(x)
        # (64, 50) -> (64, 50, 200)  [batch_size, seq_len, embedding_dim]

        # 50개 단어 순서대로 훑으며 주변 문맥을 반영한 100차원 문맥으로 변경
        # _ 자리는 최종 기억 상태(call state)인데 여기선 안써서 버림
        # (64, 50, 200) - > (64, 50, 100)  [Batch, seq_len, hidden_size]
        x, _ = self.lstm(x)

        # 출력 슬라이싱
        # 모든 맥락이 최종 압축된 맨 마지막 단어만 슬라이싱
        # (64, 50, 100) -> (64, 100) [batch, hidden_size]
        x = x[:, -1, :]

        # ELU 적용
        # 차원 유지, 알맹이 값만 필터링
        x = F.elu(x)

        # 최종 출력층 linear 통과
        # 100개의 문맥 특징을 조합해 최종 2개의 숫자로 결론
        # (64, 100) -> (64, 2) [Batch, output_size]
        x = self.lin(x)

        return x

# 5. Refectored 다중 지표 및 마지막 시점 추출 RNN Model
    def training_step(self, batch, batch_idx):
        x, y = batch
        # 1. 배치에서 x와 y를 분리

        # 2. forward 함수를 호출하며 yhat 계산
        y_hat = self(x)         # 크기 (64, 2)

        # 3. 예측 점수와 정답을 비교하여 틀린만큼 오차 계산
        loss = self.loss_function(y_hat, y)

        # 4. 이번 배치에서 훈련 정확도 계산
        train_acc = self.train














