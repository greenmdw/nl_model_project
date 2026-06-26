# -*- coding: utf-8 -*-
"""
    IMDB 영화 리뷰 데이터셋을 사용하여 리뷰 문장이 긍정(pos)인지 부정(neg)인지 분류하는
    LSTM 기반 자연어 처리 모델을 PyTorch Lightning으로 학습합니다.

주의:
    이 원본 코드는 torchtext.legacy API를 사용합니다.
    torchtext.legacy는 최신 torchtext 버전에서는 제거되었을 수 있으므로,
    실행 환경에 따라 torchtext 구버전 설치가 필요할 수 있습니다.

예시 설치:
    pip install pytorch-lightning
    pip install torchtext==0.10.0

전체 실행 흐름:
    1. 필요한 패키지 불러오기
    2. torchtext Field 객체 생성
    3. IMDB 데이터셋 로드
    4. 단어 사전 vocabulary 생성
    5. BucketIterator로 데이터 로더 생성
    6. PyTorch Lightning 기반 LSTM 모델 정의
    7. 모델 객체 생성
    8. Trainer를 이용하여 모델 학습
"""

# ---------------------------------------------------------------------
# 1. 필요한 라이브러리 불러오기
# ---------------------------------------------------------------------

# torch는 PyTorch의 핵심 라이브러리입니다.
# 텐서 연산, GPU 사용, 딥러닝 모델 학습 등에 사용됩니다.
import torch

# torch.nn은 신경망 계층을 만들 때 사용하는 모듈입니다.
# LSTM, Linear, CrossEntropyLoss 같은 딥러닝 구성요소를 제공합니다.
import torch.nn as nn

# torch.nn.functional은 활성화 함수나 손실 함수 등을 함수 형태로 사용할 수 있게 해 줍니다.
# 여기서는 ELU 활성화 함수 F.elu()를 사용합니다.
import torch.nn.functional as F

# Adam은 딥러닝 모델의 가중치를 업데이트하는 최적화 알고리즘입니다.
# 학습 과정에서 손실값을 줄이는 방향으로 파라미터를 조정합니다.
from torch.optim import Adam

# PyTorch Lightning은 PyTorch 학습 코드를 더 구조적으로 작성하게 해 주는 라이브러리입니다.
# 반복 학습문, 검증 과정, GPU 설정 등을 간결하게 관리할 수 있습니다.
import pytorch_lightning as pl

# Field는 torchtext.legacy에서 텍스트 데이터와 라벨 데이터를 전처리하기 위한 객체입니다.
# 텍스트를 토큰화하고, 길이를 맞추고, vocabulary를 만드는 데 사용됩니다.
from torchtext.legacy.data import Field

# BucketIterator는 문장 길이가 비슷한 데이터끼리 배치로 묶어 주는 데이터 로더입니다.
# RNN/LSTM 계열 모델에서는 문장 길이 차이가 크면 패딩이 많이 생기므로,
# 비슷한 길이끼리 묶으면 학습 효율을 높일 수 있습니다.
from torchtext.legacy.data import BucketIterator

# IMDB는 torchtext.legacy.datasets에서 제공하는 영화 리뷰 감성 분석 데이터셋입니다.
# 리뷰 문장과 긍정/부정 라벨로 구성되어 있습니다.
from torchtext.legacy.datasets import IMDB


# ---------------------------------------------------------------------
# 2. 데이터 전처리 Field 객체 생성
# ---------------------------------------------------------------------

# text_field는 영화 리뷰 문장 데이터를 처리하기 위한 Field 객체입니다.
# sequential=True는 입력 데이터가 단어들이 순서대로 나열된 문장 데이터라는 뜻입니다.
# include_lengths=True는 각 문장의 실제 길이 정보를 함께 저장하겠다는 뜻입니다.
# fix_length=200은 모든 문장을 최대 200개 토큰 길이로 맞추겠다는 뜻입니다.
# 길이가 200보다 짧은 문장은 패딩되고, 200보다 긴 문장은 잘릴 수 있습니다.
text_field = Field(
    sequential=True,
    include_lengths=True,
    fix_length=200
)

# label_field는 긍정/부정 라벨 데이터를 처리하기 위한 Field 객체입니다.
# sequential=False는 라벨이 문장처럼 순서가 있는 데이터가 아니라 단일 값이라는 뜻입니다.
# 예를 들어 "pos" 또는 "neg" 하나의 값만 사용합니다.
label_field = Field(
    sequential=False
)


# ---------------------------------------------------------------------
# 3. IMDB 데이터셋 로드
# ---------------------------------------------------------------------

# IMDB.splits()는 IMDB 영화 리뷰 데이터셋을 훈련 데이터와 테스트 데이터로 나누어 불러옵니다.
# 첫 번째 인자인 text_field는 리뷰 문장을 어떻게 처리할지 정의합니다.
# 두 번째 인자인 label_field는 라벨을 어떻게 처리할지 정의합니다.
train, test = IMDB.splits(
    text_field,
    label_field
)

# train.examples[0]은 훈련 데이터셋의 첫 번째 샘플입니다.
# vars()를 사용하면 해당 샘플의 text와 label 정보를 딕셔너리 형태로 확인할 수 있습니다.
print("[훈련 데이터 첫 번째 샘플 전체 정보]")
print(vars(train.examples[0]))

# 첫 번째 샘플의 label만 출력합니다.
# 보통 'pos' 또는 'neg' 중 하나가 출력됩니다.
print("\n[훈련 데이터 첫 번째 샘플 라벨]")
print(vars(train.examples[0])["label"])


# ---------------------------------------------------------------------
# 4. Vocabulary 생성
# ---------------------------------------------------------------------

# build_vocab()은 훈련 데이터에 등장한 단어들을 모아 단어 사전을 생성합니다.
# 단어 사전은 단어를 정수 인덱스로 바꾸기 위해 필요합니다.
# vectors='fasttext.simple.300d'는 사전 학습된 FastText 300차원 임베딩 벡터를 사용하겠다는 의미입니다.
# 즉, 각 단어를 300차원 숫자 벡터로 표현합니다.
text_field.build_vocab(
    train,
    vectors="fasttext.simple.300d"
)

# label_field.build_vocab()은 라벨 문자열도 정수 인덱스로 바꾸기 위한 사전을 만듭니다.
# 예를 들어 'neg'와 'pos'가 각각 특정 정수 번호로 매핑됩니다.
label_field.build_vocab(train)

# 생성된 단어 사전의 크기를 출력합니다.
# 이 값은 모델의 입력 단어 종류 수를 의미합니다.
print("\n[텍스트 vocabulary 크기]")
print(len(text_field.vocab))

# 생성된 라벨 사전을 출력합니다.
# 어떤 라벨이 어떤 인덱스로 변환되는지 확인할 수 있습니다.
print("\n[라벨 vocabulary 정보]")
print(label_field.vocab.stoi)


# ---------------------------------------------------------------------
# 5. 데이터 로더 생성
# ---------------------------------------------------------------------

# torch.cuda.is_available()는 현재 컴퓨터에서 CUDA GPU를 사용할 수 있는지 확인합니다.
# GPU가 있으면 'cuda'를 사용하고, 없으면 CPU를 사용합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# batch_size는 한 번에 모델에 입력할 데이터 샘플 개수입니다.
# 32라면 한 번의 학습 단계에서 리뷰 32개를 묶어서 학습합니다.
batch_size = 32

# BucketIterator.splits()는 train과 test 데이터셋을 배치 단위로 공급하는 반복자를 만듭니다.
# train_iter는 훈련용 데이터 로더이고, test_iter는 검증 또는 평가용 데이터 로더입니다.
# device=device는 배치 데이터를 CPU 또는 GPU에 올려서 사용하겠다는 의미입니다.
train_iter, test_iter = BucketIterator.splits(
    (train, test),
    batch_size=batch_size,
    device=device
)


# ---------------------------------------------------------------------
# 6. LSTM 모델 클래스 정의
# ---------------------------------------------------------------------

# RNNModel 클래스는 PyTorch Lightning의 LightningModule을 상속합니다.
# LightningModule을 사용하면 모델 구조, 학습 단계, 검증 단계, 최적화 설정을 한 클래스 안에 정리할 수 있습니다.
class RNNModel(pl.LightningModule):

    # __init__()은 모델 객체가 생성될 때 한 번 실행되는 생성자 함수입니다.
    # embedding은 text_field.vocab.vectors로부터 전달되는 사전 학습 단어 임베딩 행렬입니다.
    # lstm_input_size=300은 FastText 임베딩 차원이 300차원이기 때문에 입력 크기를 300으로 설정한 것입니다.
    # lstm_hidden_size=100은 LSTM이 내부적으로 저장하는 은닉 상태 벡터 크기입니다.
    # output_size=3은 출력 클래스 개수입니다.
    # 단, IMDB는 보통 긍정/부정 2개 클래스이므로 실제 환경에서는 output_size=2가 더 자연스럽습니다.
    def __init__(
        self,
        embedding,
        lstm_input_size=300,
        lstm_hidden_size=100,
        output_size=3
    ):
        # 부모 클래스인 pl.LightningModule의 초기화 기능을 실행합니다.
        # Lightning 내부 기능을 정상적으로 사용하기 위해 반드시 호출해야 합니다.
        super().__init__()

        # 사전 학습된 임베딩 벡터 행렬을 모델의 멤버 변수로 저장합니다.
        # 이 코드는 nn.Embedding 레이어가 아니라 이미 만들어진 벡터 행렬을 직접 인덱싱하는 방식입니다.
        self.embedding = embedding

        # nn.LSTM은 순서가 있는 데이터, 즉 문장이나 시계열 데이터를 처리하는 계층입니다.
        # 입력 크기는 300차원 임베딩 벡터이고, 출력 은닉 상태 크기는 100입니다.
        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size
        )

        # nn.Linear는 LSTM에서 나온 100차원 특징을 클래스 개수만큼의 출력값으로 변환합니다.
        # 최종적으로 각 클래스에 대한 점수(logit)를 출력합니다.
        self.lin = nn.Linear(
            lstm_hidden_size,
            output_size
        )

        # CrossEntropyLoss는 다중 클래스 분류에서 많이 사용하는 손실 함수입니다.
        # 모델이 예측한 클래스 점수와 실제 정답 라벨의 차이를 계산합니다.
        self.loss_function = nn.CrossEntropyLoss()

        # PyTorch Lightning 버전에 따라 pl.metrics.Accuracy()가 동작하지 않을 수 있습니다.
        # 최신 환경에서는 torchmetrics.Accuracy 사용이 권장됩니다.
        try:
            # 구버전 PyTorch Lightning에서 정확도 계산 객체를 생성합니다.
            self.train_accuracy = pl.metrics.Accuracy()

            # 검증 정확도 계산 객체를 생성합니다.
            self.val_accuracy = pl.metrics.Accuracy()
        except AttributeError:
            # 최신 PyTorch Lightning에서는 pl.metrics가 제거되었을 수 있으므로 torchmetrics를 사용합니다.
            from torchmetrics import Accuracy

            # 다중 클래스 정확도 계산 객체를 생성합니다.
            self.train_accuracy = Accuracy(task="multiclass", num_classes=output_size)

            # 검증 정확도 계산 객체를 생성합니다.
            self.val_accuracy = Accuracy(task="multiclass", num_classes=output_size)

    # forward()는 모델의 순전파를 정의하는 함수입니다.
    # 입력 데이터 X가 들어오면 임베딩, LSTM, 활성화 함수, 선형층을 거쳐 예측값을 반환합니다.
    def forward(self, X: torch.Tensor):

        # X는 정수 인덱스로 변환된 문장 데이터입니다.
        # self.embedding[X]는 각 단어 인덱스에 해당하는 300차원 임베딩 벡터를 가져옵니다.
        # to(self.device)는 해당 텐서를 현재 모델이 사용하는 장치(CPU 또는 GPU)로 이동시킵니다.
        # permute(1, 0, 2)는 텐서 차원 순서를 바꿉니다.
        # LSTM은 기본적으로 (문장길이, 배치크기, 임베딩차원) 형태를 기대하므로 차원을 맞춥니다.
        x = self.embedding[X].to(self.device).permute(1, 0, 2)

        # LSTM 계층에 임베딩 벡터 시퀀스를 입력합니다.
        # x는 각 시점별 LSTM 출력입니다.
        # _에는 마지막 hidden state와 cell state가 들어가지만 여기서는 사용하지 않습니다.
        x, _ = self.lstm(x)

        # LSTM 출력의 차원을 다시 (배치크기, 문장길이, 은닉크기) 형태로 변경합니다.
        # F.elu()는 ELU 활성화 함수를 적용하여 비선형성을 추가합니다.
        x = F.elu(x.permute(1, 0, 2))

        # 각 단어 위치마다 출력층을 적용하여 클래스 점수로 변환합니다.
        # 결과 형태는 대략 (배치크기, 문장길이, 클래스개수)가 됩니다.
        x = self.lin(x)

        # 문장길이 방향(dim=1)으로 모든 위치의 점수를 합산합니다.
        # 이렇게 하면 문장 전체에 대한 하나의 클래스 점수 벡터가 만들어집니다.
        x = x.sum(dim=1)

        # 최종 예측 점수(logits)를 반환합니다.
        # CrossEntropyLoss는 softmax를 내부적으로 처리하므로 여기서는 softmax를 직접 적용하지 않습니다.
        return x

    # training_step()은 훈련 배치 하나에 대해 수행할 연산을 정의합니다.
    # PyTorch Lightning은 학습 중 이 함수를 자동으로 반복 호출합니다.
    def training_step(self, batch, batch_idx):

        # batch.text[0]에는 정수 인코딩된 문장 텐서가 들어 있습니다.
        # .T는 텐서를 전치하여 모델 입력 형태에 맞춥니다.
        # batch.label에는 각 문장의 실제 정답 라벨이 들어 있습니다.
        x, y = batch.text[0].T, batch.label

        # self(x)는 forward(x)를 호출하는 것과 같습니다.
        # 모델이 각 문장에 대한 클래스 점수를 예측합니다.
        y_hat = self(x)

        # 손실 함수를 사용하여 예측값 y_hat과 실제 정답 y의 차이를 계산합니다.
        loss = self.loss_function(y_hat, y)

        # 훈련 정확도를 계산합니다.
        train_acc = self.train_accuracy(y_hat, y)

        # self.log()는 학습 과정에서 지표를 기록합니다.
        # prog_bar=True는 진행률 표시줄에 train_acc 값을 보여 주겠다는 의미입니다.
        self.log(
            "train_acc",
            train_acc,
            prog_bar=True
        )

        # Lightning은 반환된 loss를 이용하여 역전파와 파라미터 업데이트를 수행합니다.
        return {
            "loss": loss
        }

    # validation_step()은 검증 배치 하나에 대해 수행할 연산을 정의합니다.
    # 모델 학습 중 검증 데이터에 대한 성능을 확인할 때 사용됩니다.
    def validation_step(self, batch, batch_idx):

        # 검증 배치에서 입력 문장과 정답 라벨을 분리합니다.
        x, y = batch.text[0].T, batch.label

        # 검증 데이터에 대한 예측값을 계산합니다.
        y_hat = self(x)

        # 검증 손실을 계산합니다.
        loss = self.loss_function(y_hat, y)

        # 검증 정확도를 계산합니다.
        val_acc = self.val_accuracy(y_hat, y)

        # 검증 정확도를 진행률 표시줄에 기록합니다.
        self.log(
            "val_acc",
            val_acc,
            prog_bar=True
        )

        # 검증 손실을 반환합니다.
        return {
            "validation_loss": loss
        }

    # train_dataloader()는 훈련에 사용할 데이터 로더를 반환합니다.
    # PyTorch Lightning의 trainer.fit()은 이 함수를 호출하여 훈련 데이터를 가져옵니다.
    def train_dataloader(self):

        # 앞에서 만든 train_iter를 훈련 데이터 로더로 반환합니다.
        return train_iter

    # val_dataloader()는 검증에 사용할 데이터 로더를 반환합니다.
    def val_dataloader(self):

        # 앞에서 만든 test_iter를 검증 데이터 로더로 반환합니다.
        return test_iter

    # configure_optimizers()는 모델 학습에 사용할 최적화 알고리즘을 정의합니다.
    # PyTorch Lightning은 이 함수를 호출하여 optimizer를 자동으로 설정합니다.
    def configure_optimizers(self):

        # Adam optimizer는 학습률을 적응적으로 조절하는 최적화 알고리즘입니다.
        # self.parameters()는 학습 가능한 모델 파라미터 전체를 의미합니다.
        # lr=0.01은 학습률을 0.01로 설정한다는 의미입니다.
        return Adam(
            self.parameters(),
            lr=0.01
        )


# ---------------------------------------------------------------------
# 7. 모델 객체 생성
# ---------------------------------------------------------------------

# text_field.vocab.vectors는 build_vocab()에서 생성된 사전 학습 임베딩 벡터 행렬입니다.
# 이 벡터 행렬을 RNNModel에 전달하여 모델이 단어 임베딩을 사용할 수 있게 합니다.
model = RNNModel(
    text_field.vocab.vectors
)


# ---------------------------------------------------------------------
# 8. Trainer 생성 및 모델 학습
# ---------------------------------------------------------------------

# PyTorch Lightning의 버전에 따라 GPU 설정 인자가 달라질 수 있습니다.
# 구버전에서는 gpus=1을 사용하고, 최신 버전에서는 accelerator/devices를 사용하는 경우가 많습니다.
try:
    # 구버전 PyTorch Lightning 방식입니다.
    # gpus=1은 GPU 1개를 사용하겠다는 의미입니다.
    # max_epochs=3은 전체 훈련 데이터를 3번 반복해서 학습하겠다는 의미입니다.
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=3
    )
except TypeError:
    # 최신 PyTorch Lightning 방식입니다.
    # accelerator는 GPU가 있으면 "gpu", 없으면 "cpu"로 설정합니다.
    # devices=1은 사용할 장치 개수를 1개로 지정합니다.
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=3
    )

# trainer.fit()은 모델 학습을 시작합니다.
# Lightning은 내부적으로 train_dataloader(), validation_step(), configure_optimizers() 등을 호출합니다.
trainer.fit(model)
