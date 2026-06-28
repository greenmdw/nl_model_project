import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall
from collections import Counter

# ---------------------------------------------------------------------
# 1. 하이퍼파라미터 및 경로 설정
# ---------------------------------------------------------------------
# 현재 파일 위치 기준으로 상위 혹은 다른 폴더에 있는 data/ratings.txt를 안전하게 가리킵니다.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ratings.txt')

BATCH_SIZE = 64
MAX_LENGTH = 50  # 한글 리뷰는 보통 문장이 짧으므로 50 정도가 적당합니다.
EMBEDDING_DIM = 200
LSTM_HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 5


# ---------------------------------------------------------------------
# 2. 한글 데이터 전처리 및 전용 Dataset 클래스 정의
# ---------------------------------------------------------------------
class RatingsDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_vocab_size=20000):
        self.labels = []
        self.texts = []

        # 파일 읽기 (탭 구분자 \t 사용)
        with open(data_path, 'r', encoding='utf-8') as f:
            next(f)  # 헤더 줄 skip (id, document, label)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:  # 정상적인 데이터만 파싱
                    _, document, label = parts
                    if document.strip():  # 빈 문자열 제외
                        self.labels.append(int(label))
                        self.texts.append(document.split())  # 기본 공백 토큰화

        # 단어 사전(Vocabulary) 구축
        if vocab is None:
            all_tokens = [token for text in self.texts for token in text]
            token_counts = Counter(all_tokens)
            vocab_list = [token for token, _ in token_counts.most_common(max_vocab_size)]

            # 특수 토큰 지정 (<pad>=0, <unk>=1)
            self.vocab = {token: idx + 2 for idx, token in enumerate(vocab_list)}
            self.vocab['<pad>'] = 0
            self.vocab['<unk>'] = 1
        else:
            self.vocab = vocab

        self.pad_index = self.vocab['<pad>']
        self.unk_index = self.vocab['<unk>']

    def __len__(self):
        return len(self.labels)

    def text_to_tensor(self, tokenized_text):
        # 단어 사전에 없으면 <unk>(1)로 매핑
        indexed = [self.vocab.get(token, self.unk_index) for token in tokenized_text]
        return torch.tensor(indexed, dtype=torch.long)

    def __getitem__(self, idx):
        return self.labels[idx], self.text_to_tensor(self.texts[idx])


# 데이터셋 로드 및 분할
full_dataset = RatingsDataset(DATA_PATH)
vocab_size = len(full_dataset.vocab)
pad_index = full_dataset.pad_index

# 훈련용 / 검증용 8:2 분할
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])


# ---------------------------------------------------------------------
# 3. DataLoader를 위한 collate_fn 구현 (동적 패딩)
# ---------------------------------------------------------------------
def collate_batch(batch):
    label_list, text_list = [], []
    for label, text_tensor in batch:
        label_list.append(label)
        text_list.append(text_tensor)

    labels = torch.tensor(label_list, dtype=torch.long)

    # 배치 내 가장 긴 문장 기준으로 pad 값(0) 부여
    texts = pad_sequence(text_list, batch_first=True, padding_value=pad_index)

    # 지정한 max_length를 넘어가면 잘라내어 안전성 확보
    if texts.size(1) > MAX_LENGTH:
        texts = texts[:, :MAX_LENGTH]

    return texts, labels


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# ---------------------------------------------------------------------
# 4. Refactored 다중 지표 및 마지막 시점 추출 RNN Model
# ---------------------------------------------------------------------
class RatingsLSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, pad_index, embedding_dim=200, lstm_hidden_size=100, output_size=2):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

        # 은닉 상태(Hidden Size) 크기를 받아 최종 긍부정 분류
        self.lin = nn.Linear(lstm_hidden_size, output_size)
        self.loss_function = nn.CrossEntropyLoss()

        # 다중 평가지표 정의 (정확도, F1, 정밀도, 재현율)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=output_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_size)

        self.val_f1 = F1Score(task="multiclass", num_classes=output_size)
        self.val_precision = Precision(task="multiclass", num_classes=output_size)
        self.val_recall = Recall(task="multiclass", num_classes=output_size)

    def forward(self, x: torch.Tensor):
        # 1. Embedding 레이어 통과 -> (Batch, Seq_Len, Embedding_Dim)
        x = self.embedding(x)

        # 2. LSTM 레이어 통과 -> (Batch, Seq_Len, Hidden_Size)
        x, _ = self.lstm(x)

        # ⭐ [수정 핵심] sum 방식을 버리고, 문맥이 모두 압축된 '맨 마지막 단어 시점'만 도려냅니다.
        # 차원 변화: (Batch, Seq_Len, Hidden_Size) -> (Batch, Hidden_Size)
        x = x[:, -1, :]

        # 3. 활성화 함수 거치며 비선형 특징 부각
        x = F.elu(x)

        # 4. 최종 출력층 통과 -> (Batch, Output_Size)
        x = self.lin(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)
        train_acc = self.train_accuracy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_function(y_hat, y)

        # 다중 지표 기록 수행
        val_acc = self.val_accuracy(y_hat, y)
        val_f1 = self.val_f1(y_hat, y)
        val_prec = self.val_precision(y_hat, y)
        val_rec = self.val_recall(y_hat, y)

        # 모니터링 진행바와 로그 테이블에 함께 기록
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        self.log("val_precision", val_prec, prog_bar=False)
        self.log("val_recall", val_rec, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------
# 5. 실행 및 GPU 가속 시작
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # 데이터 경로가 잘 잡혔는지 검증
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"지정된 위치에 파일이 없습니다. 경로를 확인해 주세요: {DATA_PATH}")

    model = RatingsLSTMModel(vocab_size=vocab_size, pad_index=pad_index)

    # GPU 가속(accelerator="gpu") 장치 자동 할당 설정 
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10
    )

    print(f"🚀 네이버 영화 리뷰 LSTM 학습을 시작합니다. (대상 파일: {DATA_PATH})")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)