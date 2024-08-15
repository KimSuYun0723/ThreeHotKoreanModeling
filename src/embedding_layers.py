"""
GENERAL TODO:
1) Replace _emb_i,v,f with an EmbeddingBag for efficiency
"""

import math  # 수학적 연산을 위해 math 모듈을 임포트
import torch  # PyTorch 모듈을 임포트
from torch import Tensor  # Tensor 타입을 임포트
import torch.nn as nn  # PyTorch의 신경망 모듈을 임포트
from typing import Union  # 여러 타입을 허용하기 위해 Union을 임포트

from .dictionaries import ThreeHotDict, AlphabetDict, ThreeHotDictArbitraryOrdering  # 필요한 딕셔너리 클래스들을 임포트

class AlphabetEmbedding(nn.Module):
    """
    원-핫 인코딩 딕셔너리를 위한 임베딩 레이어. `nn.Embedding`을 감싸서 `AlphabetDict`로 직접 초기화할 수 있게 함.
    """
    def __init__(self, alphabet_dict: AlphabetDict, emb_size):
        """Alphabet 임베딩 테이블 초기화

        Args:
            dictionary: AlphabetDict
            emb_size: 임베딩 크기
        """
        super(AlphabetEmbedding, self).__init__()  # nn.Module 초기화
        self.alphabet_dict = alphabet_dict  # AlphabetDict 저장
        self._pad_idx = self.alphabet_dict.pad()  # 패딩 인덱스 저장
        self.emb_size = emb_size  # 임베딩 크기 저장
        self.embedding = nn.Embedding(
            len(self.alphabet_dict), emb_size, padding_idx=self._pad_idx
        )  # 임베딩 레이어 초기화

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)  # 임베딩 후 sqrt(임베딩 크기)로 스케일링


class ThreeHotEmbedding(nn.Module): # threehot 임베딩 레이어
    """
    여러 패딩 인덱스를 허용하는 `nn.EmbeddingBag`처럼 동작.
    자모 임베딩 합산 -> 평균 --> `sqrt(emb_size)`로 스케일링 --> 벡터 생성
    """
    def __init__(
        self, dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering], emb_size
    ):
        """ThreeHot 임베딩 테이블 초기화

        Args:
            dictionary: `ThreeHotDict` 또는 `ThreeHotDictArbitraryOrdering`
            emb_size: 임베딩 크기
        """
        super(ThreeHotEmbedding, self).__init__()  # nn.Module 초기화

        size_i, size_v, size_f = dictionary.sizes()  # 각 하위 문자의 사이즈 가져오기
        pad_i, pad_v, pad_f = dictionary.pad()  # 각 하위 문자의 패딩 인덱스 가져오기
        self.pad_f = pad_f  # 종성 패딩 인덱스 저장

        self._emb_i = nn.Embedding(size_i, emb_size, padding_idx=pad_i)  # 초성 임베딩 레이어 초기화
        self._emb_v = nn.Embedding(size_v, emb_size, padding_idx=pad_v)  # 중성 임베딩 레이어 초기화
        self._emb_f = nn.Embedding(size_f, emb_size, padding_idx=pad_f)  # 종성 임베딩 레이어 초기화
        self.emb_size = emb_size  # 임베딩 크기 저장

    def forward(self, tokens: Tensor):
        """입력 텐서를 임베딩. `tokens`는 threehot 형태여야 함.

        Args:
            tokens: threehot 텐서. 형태는 (X, Y, ... , 3)이며,
            [..., 0]은 초성, [..., 1]은 중성, [..., 2]는 종성을 의미함.

        Returns:
            입력 임베딩의 결합된 표현 반환
        """
        assert len(tokens.size()) > 1 and tokens.size(-1) == 3, "must be threehot"  # tokens가 threehot 형태인지 확인

        i, v, f = tokens[..., 0], tokens[..., 1], tokens[..., 2]  # 초성, 중성, 종성 분리
        return (
            (self._emb_i(i) + self._emb_v(v) + self._emb_f(f))  # 각 자모 임베딩 합산
            * math.sqrt(self.emb_size)  # 임베딩 크기의 제곱근으로 스케일링
            / 3  # 평균 계산
        )


class ThreeHotConcatEmbedding(nn.Module):
    """
    `ThreeHotEmbedding`과는 달리 
    하위 임베딩을 연결한 후 --> Linear 레이어를 통해 최종 `emb_size` 출력을 생성.
    """
    def __init__(
        self, dictionary: Union[ThreeHotDict, ThreeHotDictArbitraryOrdering], emb_size
    ):
        """ThreeHot Concat 임베딩 테이블 초기화

        Args:
            dictionary: `ThreeHotDict` 또는 `ThreeHotDictArbitraryOrdering`
            emb_size: 임베딩 크기
        """
        super(ThreeHotConcatEmbedding, self).__init__()  # nn.Module 초기화

        size_i, size_v, size_f = dictionary.sizes()  # 각 하위 문자의 사이즈 가져오기
        pad_i, pad_v, pad_f = dictionary.pad()  # 각 하위 문자의 패딩 인덱스 가져오기
        self.pad_f = pad_f  # 종성 패딩 인덱스 저장

        self._emb_i = nn.Embedding(size_i, emb_size, padding_idx=pad_i)  # 초성 임베딩 레이어 초기화
        self._emb_v = nn.Embedding(size_v, emb_size, padding_idx=pad_v)  # 중성 임베딩 레이어 초기화
        self._emb_f = nn.Embedding(size_f, emb_size, padding_idx=pad_f)  # 종성 임베딩 레이어 초기화

        self._combiner = nn.Linear(3 * emb_size, emb_size)  # 연결된 임베딩을 결합하기 위한 Linear 레이어 초기화

        self.emb_size = emb_size  # 임베딩 크기 저장

    def forward(self, tokens: Tensor):
        """입력 텐서를 임베딩. `tokens`는 threehot 형태여야 함.

        Args:
            tokens: threehot 텐서. 형태는 (X, Y, ... , 3)이며,
            [..., 0]은 초성, [..., 1]은 중성, [..., 2]는 종성을 의미함.

        Returns:
            입력 임베딩의 결합된 표현 반환
        """
        assert len(tokens.size()) > 1 and tokens.size(-1) == 3, "must be threehot"  # tokens가 threehot 형태인지 확인

        i, v, f = tokens[..., 0], tokens[..., 1], tokens[..., 2]  # 초성, 중성, 종성 분리
        i, v, f = self._emb_i(i), self._emb_v(v), self._emb_f(f)  # 각 자모를 임베딩

        return self._combiner(torch.cat([i, v, f], axis=-1)) * math.sqrt(self.emb_size)  # 임베딩을 연결하고 결합한 후 스케일링