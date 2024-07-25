"""
GENERAL TODO:

1) UnrolledRNN과 UnrolledDiagonalRNN이 코드를 공유하도록 만들기 (init을 제외하고 동일함)
2) 모든 디코더 레이어가 (logits), (preds)를 출력하도록 만들기 (혹은 preds를 사용하지 않으므로 제거하기)
"""

import torch
import torch.nn as nn

from typing import Union

from .dictionaries import *

class AlphabetDecoder(nn.Module):
    """
    `AlphabetDict`를 기반으로 한 원-핫 디코더입니다. 클래스 로짓(logits)과 `argmax` 예측을 반환합니다.
    """
    def __init__(self, dictionary: AlphabetDict, hidden_size):
        super(AlphabetDecoder, self).__init__()
        # hidden_size를 받아서 projection layer를 초기화
        self._projection = nn.Linear(hidden_size, len(dictionary))

    def forward(self, hidden, **_):
        # hidden 상태를 projection layer를 통해 logits 계산
        logits = self._projection(hidden)
        # logits에서 argmax를 사용하여 예측값 계산, 이때 gradients는 사용하지 않음
        return logits, torch.argmax(logits, dim=-1).detach()

class UnrolledRNNDecoder(nn.Module):
    """음절의 각 세 하위 문자를 예측하기 위해 RNN 루프의 3 틱을 계산하는 unrolled RNN입니다.

    이 코드는 전체 크기의 RNN을 사용하고 `tanh(h*Wh + x*Wx)` 항목을 세 번 수동으로 계산합니다.
    """
    def __init__(
        self,
        dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering],
        hidden_size,
    ):
        """Unrolled RNN 디코더 초기화

        Args:
            dictionary: `ThreeHotDict` 또는 `ThreeHotDictArbitraryOrdering`
            hidden_size: 모델의 hidden 차원
        """
        super(UnrolledRNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        # 딕셔너리에서 각 하위 문자의 사이즈를 가져옴
        size_i, size_v, size_f = dictionary.sizes()
        self.size_i, self.size_v, self.size_f = size_i, size_v, size_f
        # 딕셔너리에서 패딩 인덱스를 가져옴
        pad_i, pad_v, _ = dictionary.pad()

        # 각 하위 문자를 임베딩하기 위한 임베딩 레이어 초기화
        self._reembed_i = nn.Embedding(size_i, hidden_size, padding_idx=pad_i)
        self._reembed_v = nn.Embedding(size_v, hidden_size, padding_idx=pad_v)

        # 각 하위 문자의 출력을 위한 fully connected 레이어 초기화
        self._fc_i = nn.Linear(hidden_size, size_i)
        self._fc_v = nn.Linear(hidden_size, size_v)
        self._fc_f = nn.Linear(hidden_size, size_f)

        # RNN의 가중치 행렬을 초기화
        self._A = nn.Linear(hidden_size, hidden_size)
        self._B = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, force=None):
        """Threehot triplet 예측

        이 함수는 훈련을 위해 설계되었으며 teacher forcing을 사용하여 음절을 예측합니다. 
        `force` 파라미터는 예측할 음절의 threehot 하위 문자 레이블을 포함하므로 (seq, batch, 3)의 형태를 가져야 합니다.
        인덱스 [..., 0]는 첫 번째 하위 문자 클래스 등을 의미합니다.

        `force`가 `None`이면, 모델 자체의 예측을 사용하여 다음 하위 문자를 생성합니다.

        Args:
            hidden: hidden context 벡터
            force: 예측을 위한 teacher forcing 값을 포함하는 threehot 텐서

        Returns:
            logits: 각 하위 문자(i, v, f)에 대한 정규화되지 않은 로짓의 3-튜플
            preds: 각 로짓 벡터의 argmax
        """
        if force is not None:
            # hidden은 (seq, batch, H)
            # force는 (seq, batch, 3)

            # force로부터 각 하위 문자 레이블을 가져옴
            i_labels = force[..., 0]  # (seq, batch)
            v_labels = force[..., 1]  # (seq, batch)

            # 임베딩 레이어를 통해 레이블 임베딩
            emb_i = self._reembed_i(i_labels)  # (seq, batch, h)
            emb_v = self._reembed_v(v_labels)  # (seq, batch, h)

            # RNN unroll
            h_0 = torch.zeros_like(hidden)  # 초기 hidden 상태
            h_1 = torch.tanh(self._A(hidden) + self._B(h_0))
            h_2 = torch.tanh(self._A(emb_i) + self._B(h_1))
            h_3 = torch.tanh(self._A(emb_v) + self._B(h_2))

            # 각 하위 문자에 대한 로짓 계산
            logits_i = self._fc_i(h_1)
            logits_v = self._fc_v(h_2)
            logits_f = self._fc_f(h_3)

            # 각 로짓에 대해 argmax를 사용하여 예측값 계산
            pred_i = torch.argmax(logits_i, dim=-1).detach()
            pred_v = torch.argmax(logits_v, dim=-1).detach()
            pred_f = torch.argmax(logits_f, dim=-1).detach()

            return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)

        else:
            # hidden은 (seq, batch, H)
            # force는 (seq, batch, 3)

            # 임베딩 레이어를 통해 레이블 임베딩
            emb_i = self._reembed_i(i_labels)  # (seq, batch, h)
            emb_v = self._reembed_v(v_labels)  # (seq, batch, h)

            # RNN unroll
            h_0 = torch.zeros_like(hidden)  # 초기 hidden 상태

            h_1 = torch.tanh(self._A(hidden) + self._B(h_0))
            logits_i = self._fc_i(h_1)
            pred_i = torch.argmax(logits_i, dim=-1).detach()
            emb_i = self._reembed_i(pred_i)

            h_2 = torch.tanh(self._A(emb_i) + self._B(h_1))
            logits_v = self._fc_v(h_2)
            pred_v = torch.argmax(logits_v, dim=-1).detach()
            emb_v = self._reembed_v(pred_v)

            h_3 = torch.tanh(self._A(emb_v) + self._B(h_2))
            logits_f = self._fc_f(h_3)
            pred_f = torch.argmax(logits_f, dim=-1).detach()

            return (logits_i, logits_v, logits_f), (pred_i, pred_v, pred_f)

class DiagonalLinear(nn.Module):
    def __init__(self, hidden_size):
        super(DiagonalLinear, self).__init__()
        # 대각선 가중치와 바이어스를 파라미터로 초기화
        self._diag = nn.Parameter(torch.randn(hidden_size))
        self._bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input):
        # 대각선 가중치와 입력을 element-wise 곱하고 바이어스를 더함
        return torch.addcmul(self._bias, input, self._diag)

# TODO: 이 클래스를 UnrolledRNNDecoder의 서브클래스로 만들기 (대각선 레이어를 제외하고 동일함)
class UnrolledDiagonalRNNDecoder(nn.Module):
    def __init__(
        self,
        dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering],
        hidden_size,
    ):
        super(UnrolledDiagonalRNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        # 딕셔너리에서 각 하위 문자의 사이즈를 가져옴
        size_i, size_v, size_f = dictionary.sizes()
        # 딕셔너리에서 패딩 인덱스를 가져옴
        pad_i, pad_v, _ = dictionary.pad()

        # 각 하위 문자를 임베딩하기 위한 임베딩 레이어 초기화
        self._reembed_i = nn.Embedding(size_i, hidden_size, padding_idx=pad_i)
        self._reembed_v = nn.Embedding(size_v, hidden_size, padding_idx=pad_v)

        # 각 하위 문자의 출력을 위한 fully connected 레이어 초기화
        self._fc_i = nn.Linear(hidden_size, size_i)
        self._fc_v = nn.Linear(hidden_size, size_v)
        self._fc_f = nn.Linear(hidden_size, size_f)

        # 대각선 RNN 논문(`DIAGONAL RNNS IN SYMBOLIC MUSIC MODELING`)은
        # 임베딩에서 내부 hidden 차원으로 가기 위해 전체 dense 레이어를 사용하지만,
        # 내부 hidden 상태 업데이트를 위해 대각선 레이어를 사용합니다.
        # 현재 우리는 항상 hid_dim = emb_dim을 사용합니다.
        self._A = DiagonalLinear
