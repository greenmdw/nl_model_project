# -*- coding: utf-8 -*-
"""
    IMDB 영화 리뷰 데이터셋을 사용하여 리뷰 문장이 긍정(pos)인지 부정(neg)인지 분류하는
    LSTM 기반 자연어 처리 모델을 PyTorch Lightning으로 학습합니다.
"""

import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchtext.datasets import IMDB
from torchmetrics import F1Score, Precision, Recall

# ---------------------------------------------------------------------
# 2. 기본 설정 (하이퍼파라미터)
# ---------------------------------------------------------------------
# 🔥 [주의 깊게 볼 점] 이 숫자들을 바꿨을 때 모델의 성능이나 메모리에 어떤 영향을 줄지 고민해보세요.
max_length = 200  # 한 문장당 최대 단어 개수 (넘으면 자르고, 모자라면 패딩)
batch_size = 32  # 한 번에 학습할 데이터 묶음 크기
embedding_dim = 300  # 하나의 단어를 몇 차원의 숫자로 표현할 것인가 (단어 벡터의 크기)
hidden_size = 200  # LSTM 내부의 기억 장치(Hidden State)의 크기
output_size = 2  # 긍정/부정 2가지로 분류하므로 2
min_freq = 2  # 데이터셋 전체에서 최소 2번 이상 등장한 단어만 사전에 등록

pad_token = "<pad>"  # 문장 길이를 맞추기 위한 빈 칸 채우기용 토큰
unk_token = "<unk>"  # 사전에 없는 모르는 단어를 대체할 토큰


# ---------------------------------------------------------------------
# 3. 텍스트 전처리 함수 정의
# ---------------------------------------------------------------------

def tokenizer(text):
    """
    🔥 [주의 깊게 볼 점] 토큰화(Tokenization)의 중요성
    영어는 공백 기준 분리가 쉽지만, 한국어는 조사/어미 때문에 이 방식이 불가능합니다.
    자연어 처리의 첫 단추인 '텍스트 쪼개기'가 어떻게 일어나는지 보세요.
    """
    text = text.lower()  # 소문자로 통일 (Apple과 apple을 같은 단어로 취급하기 위함)
    return re.findall(r"\b\w+\b", text)  # 정규식을 이용해 알파벳/숫자 단위로 단어 추출


def label_to_index(label):
    """
    🔥 [주의 깊게 볼 점] Loss 함수와 라벨 인덱스의 관계
    PyTorch의 CrossEntropyLoss는 정답 라벨이 반드시 0부터 시작하는 정수(0, 1, 2...)여야 합니다.
    기존 IMDB 데이터가 1, 2로 되어 있어서 0, 1로 보정해주는 필수적인 작업입니다.
    """
    return int(label) - 1  # 1(부정) -> 0, 2(긍정) -> 1


# ---------------------------------------------------------------------
# 4. IMDB 데이터셋 로드
# ---------------------------------------------------------------------
# 데이터셋을 다운로드하고, Generator 형태의 데이터를 반복문으로 쓰기 편하게 list로 변환합니다.
train_iter = list(IMDB(split="train"))
test_iter = list(IMDB(split="test"))

# ---------------------------------------------------------------------
# 5. Vocabulary(어휘 사전) 생성
# ---------------------------------------------------------------------

counter = Counter()
for label, text in train_iter:
    counter.update(tokenizer(text))  # 모든 훈련 데이터 문장을 쪼개어 단어의 빈도수를 측정

# 특수 토큰(<pad>: 0번, <unk>: 1번)을 가장 앞에 배치합니다.
itos = [pad_token, unk_token]

# 설정한 min_freq(2번) 이상 나온 단어만 사전에 등록 (노이즈 제거 및 메모리 절약)
for word, freq in counter.items():
    if freq >= min_freq:
        itos.append(word)

# 단어 -> 인덱스 형태로 매핑하는 딕셔너리 생성 (컴퓨터가 읽을 수 있는 숫자로 바꾸기 위함)
stoi = {word: index for index, word in enumerate(itos)}

pad_index = stoi[pad_token]  # 주로 0
unk_index = stoi[unk_token]  # 주로 1


def text_to_tensor(text):
    """
    🔥 [주의 깊게 볼 점] 문장이 숫자의 배열(Tensor)로 바뀌는 순간
    "I love this movie" -> [tokenizer] -> ['i', 'love', 'this', 'movie']
                        -> [stoi 매핑]  -> [12, 45, 9, 123] -> Tensor 변환
    """
    tokens = tokenizer(text)[:max_length]  # 설정한 최대 길이만큼 자름
    # 사전에 있으면 그 번호를, 없으면 <unk> 번호(1)를 부여
    indices = [stoi.get(token, unk_index) for token in tokens]

    if len(indices) == 0:  # 혹시 문장에 아예 아는 단어가 없으면 unk 하나만 넣어줌
        indices = [unk_index]

    return torch.tensor(indices, dtype=torch.long)


# ---------------------------------------------------------------------
# 6. DataLoader 생성 (미니배치 구성 및 패딩)
# ---------------------------------------------------------------------

def collate_batch(batch):
    """
    🔥 [NLP 공부의 핵심 코드] 배치를 만들 때 길이가 다른 문장들을 어떻게 처리하는가?
    32개의 문장이 들어왔을 때, 각각 길이가 다르면 하나의 행렬(Tensor)로 묶을 수 없습니다.
    따라서 배치 내에서 가장 긴 문장을 기준으로 모자란 부분을 <pad> 토큰(0)으로 채워줍니다.
    """
    label_list = []
    text_list = []

    for label, text in batch:
        label_list.append(label_to_index(label))
        text_list.append(text_to_tensor(text))

    labels = torch.tensor(label_list, dtype=torch.long)

    # pad_sequence가 자동으로 가변 길이의 텐서들을 똑같은 길이로 맞춰 정렬해줍니다.
    # batch_first=True이면 결과 차원이 (Batch_Size, Sequence_Length)가 됩니다.
    texts = pad_sequence(
        text_list,
        batch_first=True,
        padding_value=pad_index
    )

    # 혹시 패딩된 길이가 max_length를 넘어가면 안전하게 다시 한번 잘라줍니다.
    if texts.size(1) > max_length:
        texts = texts[:, :max_length]

    return texts, labels


# 딥러닝 모델에 데이터를 효율적으로 나눠서 공급해주는 DataLoader 정의
train_loader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


# ---------------------------------------------------------------------
# 7. LSTM 모델 클래스 정의 (PyTorch Lightning)
# ---------------------------------------------------------------------

class RNNModel(pl.LightningModule):

    def __init__(self, vocab_size, pad_index, embedding_dim=300, lstm_hidden_size=100, output_size=2):
        super().__init__()

        # 1) Embedding 레이어: 단어의 정수 인덱스를 의미를 담은 밀집 벡터로 변환
        # padding_idx를 지정하면 해당 인덱스(<pad>=0)는 미분(학습)되지 않고 항상 0 벡터로 유지됩니다. (중요)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index
        )

        # 2) LSTM 레이어: 텍스트의 시퀀스(순서) 정보를 기억하며 처리
        # batch_first=True 레이아웃 필수 설정 (입력 차원의 첫 번째가 Batch Size임을 명시)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        # 3) Linear 레이어: LSTM의 출력을 받아 최종 클래스 개수(2개: 부/긍)로 변환하는 최종 출력층
        self.lin = nn.Linear(lstm_hidden_size, output_size)


        # 다중 분류를 위한 CrossEntropyLoss (출력이 2개이므로 이진 분류 형태이지만 다중 분류로 처리 중)
        self.loss_function = nn.CrossEntropyLoss()

        # 평가 지표 설정을 위한 torchmetrics 라이브러리 사용
        from torchmetrics import Accuracy
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.train_f1 = F1Score(task="multiclass", num_classes=output_size)
        self.val_f1 = F1Score(task="multiclass", num_classes=output_size)

    def forward(self, x: torch.Tensor):
        """
        🔥 [모델 공부의 핵심] 데이터의 차원(Shape) 변화 추적하기
        입력 x의 크기: (Batch_Size, Sequence_Length) -> 예: (32, 200)
        """

        # 1. Embedding 통과 후
        # 차원 변화: (32, 200) -> (32, 200, 300) [Batch, Seq_Len, Embedding_Dim]
        x = self.embedding(x)

        # 2. LSTM 통과 후
        # LSTM은 (출력값, (최종 은닉상태, 최종 셀상태))를 반환하므로 출력값인 x만 받습니다. '_'는 버림.
        # 차원 변화: (32, 200, 300) -> (32, 200, 100) [Batch, Seq_Len, Hidden_Size]
        # 300 차원의 단어 벡터를 순서대로 읽으면서 앞뒤 문맥을 반영한 100차원 문맥 벡터로 변환
        x, _ = self.lstm(x)

        # ------------
        # 신규 방법: 모든 단어 점수 계산전에 마지막 단어 기억만 추출
        # 차원 변화: (32, 200, 100) -> (32, 200)  [Batch_size, Hidden_size]
        x = x[:, -1, :]
        # ------------

        # 3. 활성화 함수 ELU 적용 (비선형성 추가) Exponentional Linear Unit
        # ReLU가 음수를 0으로만 보내는 단점을 보완한 함수
        # 함수를 변경하는게 아닌, 행렬 안의 값을 확인하며 양수이면 그대로 통과, 음수이면 부드러운 음수로 변환
        x = F.elu(x)

        # 4. Linear 레이어 통과
        # 차원 변화: (32, 200, 100) -> (32, 200, 2) [Batch, Seq_Len, Output_Size]
        # 100개의 문맥 점수를 바탕으로 각 단어 위치마다 [부정확률, 긍정확률]이라는 숫자로 요약
        x = self.lin(x)

        # 5. 문장 길이 방향(dim=1)으로 모든 단어의 출력값을 더함 (Pooling 작업)
        # 🔥 [주의 깊게 볼 점] 이 부분이 이 코드의 특이한 점입니다.
        # 일반적인 텍스트 분류에서는 LSTM의 '마지막 시점 출력'만 사용하거나 Attention을 쓰는데,
        # 여기서는 모든 단어의 출력을 그냥 합산(sum)했습니다. 왜 이렇게 했을지, 단점은 없을지 꼭 고민해보세요!
        # 차원 변화: (32, 200, 2) -> (32, 2) [Batch, Output_Size]
        # x = x.sum(dim=1)          => 각 단계별로 나온 기억들을 그냥 다 더해버려 노이즈 쌓임
        # 이미 (32, 2) 형상이 되었으므로 sum 해줄 필요 없음.

        return x

    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning이 제공하는 자동화 함수
        1개 배치(Batch)의 데이터를 가지고 모델을 딱 한 번 업데이트하는 과정
        기존 PyTorch의 loss.backward(), optimizer.step() 등을 내부적으로 알아서 처리해줍니다.
        """
        x, y = batch
        y_hat = self(x)  # forward 함수 호출 (예측값 계산)

        loss = self.loss_function(y_hat, y)  # 손실(오차) 계산
        train_acc = self.train_accuracy(y_hat, y)  # 정확도 계산
        train_f1 = self.train_f1(y_hat, y)  # F1-score 계산 추가

        # 진행바(prog_bar)에 로그 출력 설정
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)
        self.log("train_f1", train_f1, prog_bar=True)  # 화면에 출력됨

        return loss  # 이 loss 값을 반환하면 뼈대 내부에서 알아서 역전파(Backpropagation)를 수행합니다.

    def validation_step(self, batch, batch_idx):
        """
        검증(Test/Validation) 단계에서 호출되는 함수. (학습은 진행되지 않고 오직 평가만 수행)
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        val_acc = self.val_accuracy(y_hat, y)
        val_f1 = self.val_f1(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log('val_f1_score', val_f1, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        어떤 최적화 알고리즘(Optimizer)을 사용하여 가중치를 업데이트할지 설정합니다.
        가장 무난하고 성능이 좋은 Adam 오프티마이저를 사용 중이며, 학습률(Learning Rate)은 0.001입니다.
        """
        return Adam(self.parameters(), lr=0.001)


# ---------------------------------------------------------------------
# 8. 모델 객체 생성
# ---------------------------------------------------------------------
# 위에서 정의한 하이퍼파라미터 값들을 주입하며 모델 인스턴스를 만듭니다.
model = RNNModel(
    vocab_size=len(stoi),
    pad_index=pad_index,
    embedding_dim=embedding_dim,
    lstm_hidden_size=hidden_size,
    output_size=output_size
)

# ---------------------------------------------------------------------
# 9. Trainer 생성 및 모델 학습 시작
# ---------------------------------------------------------------------
# Trainer는 하드웨어 제어 및 전체 학습 루프를 총괄합니다.
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",  # 내 컴퓨터에 GPU(CUDA)가 있으면 GPU 사용, 없으면 CPU 사용
    devices=1,  # 사용할 GPU/CPU 개수
    max_epochs=4  # 전체 데이터셋을 총 3번 반복해서 학습하겠다는 의미
)

# 학습 시작! (훈련 데이터로 학습하며 동시에 검증 데이터를 넣어 오버피팅 여부를 확인)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


# =====================================================================
# [추가 코드] 10. 학습 결과 최종 평가 (Test)
# =====================================================================
print("\n" + "=" * 50)
print("[단계 10] 학습 완료 후 test_loader를 이용한 최종 평가")
print("=" * 50)

# trainer.test()는 검증용 데이터셋을 사용해 모델의 최종 성능(Loss, Accuracy)을 리포트해줍니다.
# 내부적으로는 모델의 validation_step 또는 test_step을 호출합니다.
val_results = trainer.validate(model, dataloaders=test_loader)
print(f"최종 검증 결과: {val_results}")


# =====================================================================
# [추가 코드] 11. 새로운 샘플 데이터로 추론(Inference) 확인하기
# =====================================================================
print("\n" + "=" * 50)
print("[단계 11] 새로운 문장으로 긍정/부정 추론 예측하기")
print("=" * 50)


def predict_sentiment(model, sentence):
    """
    🔥 [주의 깊게 볼 점] 추론(Inference) 시 전처리 파이프라인의 일치성
    학습할 때 사용했던 토큰화(tokenizer), 어휘 사전(stoi), 패딩 방식을
    새로운 문장에도 '똑같이' 적용해야 모델이 올바르게 이해합니다.
    """
    # 1) 모델을 평가 모드(Evaluation Mode)로 전환
    # Dropout이나 Batch Normalization 같은 레이어가 있다면 학습 때와 다르게 동작해야 하므로 필수입니다.
    model.eval()

    # 2) 예측할 때는 역전파(Gradient) 계산이 필요 없으므로 메모리를 절약하기 위해 설정을 끕니다.
    with torch.no_grad():
        # 학습 때 만들었던 전처리 함수를 그대로 활용해 문장을 숫자로 변환합니다.
        # 차원 변화: ['i', 'love', 'it'] -> [12, 45, 9] -> Tensor [3] (1차원 벡터)
        tensor_input = text_to_tensor(sentence)

        # 🔥 [주의 깊게 볼 점] 배치 차원 추가 (Unsqueeze)
        # 딥러닝 모델은 항상 '배치(Batch)' 단위의 입력을 받도록 설계되어 있습니다.
        # 방금 만든 텐서는 문장 1개짜리 1차원이므로, 맨 앞에 '1개짜리 배치'라는 차원을 추가해줍니다.
        # 차원 변화: [Seq_Len] -> (1, Seq_Len)  예: [45] -> (1, 45)
        tensor_input = tensor_input.unsqueeze(0)

        # 모델의 연산 장치(CPU 혹은 GPU)와 입력 데이터의 위치를 일치시킵니다.
        tensor_input = tensor_input.to(model.device)

        # 3) 모델에 입력하여 예측값(Logits)을 얻습니다.
        # 출력 차원: (1, 2) -> [[부정점수, 긍정점수]]
        logits = model(tensor_input)

        # 4) Softmax를 적용하여 확률 값(0~1 사이)으로 변환합니다.
        probs = F.softmax(logits, dim=1)

        # 5) 두 개의 확률 중 더 높은 쪽의 인덱스(0 또는 1)를 뽑아냅니다.
        prediction = torch.argmax(probs, dim=1).item()

        # 확률 값 표기를 위해 소수점 추출
        neg_prob = probs[0][0].item() * 100
        pos_prob = probs[0][1].item() * 100

    # 결과 출력
    sentiment = "긍정 (Positive)" if prediction == 1 else "부정 (Negative)"
    print(f"\n입력 문장: '{sentence}'")
    print(f"해석 결과: {sentiment}")
    print(f"상세 확률: [부정] {neg_prob:.2f}% | [긍정] {pos_prob:.2f}%")


# ---------------------------------------------------------------------
# 12. 실제 새로운 문장 테스트 실행
# ---------------------------------------------------------------------

# 예측해볼 임의의 영화 리뷰 샘플 문장 리스트
sample_reviews = [
    "This movie was absolutely amazing! The acting was great and the plot was perfect.",
    "I hated this movie. It was a complete waste of my time and money. So boring.",
    "The visual effects were stunning, but the story was weak and disappointed me."
]

# 반복문을 돌며 한 문장씩 모델에게 긍정/부정 퀴즈를 냅니다.
for review in sample_reviews:
    predict_sentiment(model, review)