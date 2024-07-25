"""
The contents of this file were derived directly from PyTorch's Seq2Seq Transformer tutorial.

https://pytorch.org/tutorials/beginner/translation_transformer.html
"""

import torch  # PyTorch 모듈 임포트
from torch import Tensor  # Tensor 클래스 임포트
import torch.nn as nn  # 신경망 모듈 임포트
from torch.nn import Transformer  # Transformer 클래스 임포트
import torch.nn.functional as F  # 함수형 API 임포트

from .decoding_layers import (  # 필요한 디코딩 레이어 임포트
    AlphabetDecoder,
    ThreeHotIndependentDecoder,
    UnrolledDiagonalRNNDecoder,
    UnrolledRNNDecoder,
)

from .embedding_layers import AlphabetEmbedding, ThreeHotEmbedding  # 필요한 임베딩 레이어 임포트

from .dictionaries import *  # dictionaries 모듈에서 모든 것을 임포트

import math  # 수학 모듈 임포트

def create_mask_threehot(src, tgt, src_dict, tgt_dict):
    tgt_pad = torch.tensor([*tgt_dict.pad()], device=None, dtype=torch.long)  # 타겟 패딩 텐서 생성

    src_seq_len = src.shape[0]  # 소스 시퀀스 길이
    tgt_seq_len = tgt.shape[0]  # 타겟 시퀀스 길이

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)  # 타겟 마스크 생성
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=None).type(torch.bool)  # 소스 마스크 생성

    src_padding_mask = (src == src_dict.pad()).transpose(0, 1)  # 소스 패딩 마스크 생성

    tgt_padding_mask = (torch.all(tgt == tgt_pad, dim=-1)).transpose(0, 1)  # 타겟 패딩 마스크 생성
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  # 마스크 반환

def generate_square_subsequent_mask(sz, device=None):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)  # 상삼각행렬 마스크 생성
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )  # 마스크를 float형으로 변환하고 값 채우기
    return mask  # 마스크 반환

def create_mask(src, tgt, src_dict, tgt_dict, device=None):
    src_seq_len = src.shape[0]  # 소스 시퀀스 길이
    tgt_seq_len = tgt.shape[0]  # 타겟 시퀀스 길이

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)  # 타겟 마스크 생성
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # 소스 마스크 생성

    src_padding_mask = (src == src_dict.pad()).transpose(0, 1)  # 소스 패딩 마스크 생성
    tgt_padding_mask = (tgt == tgt_dict.pad()).transpose(0, 1)  # 타겟 패딩 마스크 생성
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  # 마스크 반환

def create_mask_threehot(src, tgt, src_dict, tgt_dict, device=None):
    tgt_pad = torch.tensor([*tgt_dict.pad()], device=device, dtype=torch.long)  # 타겟 패딩 텐서 생성

    src_seq_len = src.shape[0]  # 소스 시퀀스 길이
    tgt_seq_len = tgt.shape[0]  # 타겟 시퀀스 길이

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)  # 타겟 마스크 생성
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # 소스 마스크 생성

    src_padding_mask = (src == src_dict.pad()).transpose(0, 1)  # 소스 패딩 마스크 생성

    tgt_padding_mask = (torch.all(tgt == tgt_pad, dim=-1)).transpose(0, 1)  # 타겟 패딩 마스크 생성
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask  # 마스크 반환

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)  # 지수 함수 계산
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)  # 위치 텐서 생성
        pos_embedding = torch.zeros((maxlen, emb_size))  # 위치 임베딩 초기화
        pos_embedding[:, 0::2] = torch.sin(pos * den)  # 위치 임베딩의 짝수 인덱스에 대해 사인 함수 적용
        pos_embedding[:, 1::2] = torch.cos(pos * den)  # 위치 임베딩의 홀수 인덱스에 대해 코사인 함수 적용
        pos_embedding = pos_embedding.unsqueeze(-2)  # 위치 임베딩에 차원 추가

        self.dropout = nn.Dropout(dropout)  # 드롭아웃 설정
        self.register_buffer("pos_embedding", pos_embedding)  # 위치 임베딩을 버퍼로 등록

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )  # 토큰 임베딩에 위치 임베딩을 더한 후 드롭아웃 적용

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        src_dict,
        tgt_dict,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len=300,
        tie_tgt_embeddings=False,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
        )  # Transformer 모델 초기화
        self.tie_tgt_embeddings = tie_tgt_embeddings  # 타겟 임베딩 공유 여부
        self.src_tok_emb = AlphabetEmbedding(src_dict, emb_size)  # 소스 토큰 임베딩
        self.tgt_tok_emb = AlphabetEmbedding(tgt_dict, emb_size)  # 타겟 토큰 임베딩
        self.generator = AlphabetDecoder(tgt_dict, emb_size)  # 디코더 생성

        if self.tie_tgt_embeddings:
            self._shared_weights = nn.Parameter(
                torch.randn(self.tgt_tok_emb.embedding.weight.shape)
            )  # 공유 가중치 생성

            del self.generator._projection.weight
            del self.tgt_tok_emb.embedding.weight

            self.generator._projection.weight = self._shared_weights
            self.tgt_tok_emb.embedding.weight = self._shared_weights

        else:
            self._shared_weights = nn.Parameter()

        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout, maxlen=max_len
        )  # 위치 인코딩 초기화

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))  # 소스 임베딩에 위치 인코딩 적용
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))  # 타겟 임베딩에 위치 인코딩 적용
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )  # Transformer 모델 적용
        return self.generator(outs)  # 디코더를 통해 출력 생성

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )  # 인코더를 통해 소스 인코딩

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )  # 디코더를 통해 타겟 디코딩

class Seq2SeqTransformerThreeHot(nn.Module):
    def __init__(
        self,
        src_dict,
        tgt_dict: ThreeHotDict,
        decoder_cls,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        decoder_hidden_size: int = 512,
        dropout: float = 0.1,
        max_len=300,
        tgt_embedding_cls=ThreeHotEmbedding,
        tie_tgt_embeddings=False,
    ):
        super(Seq2SeqTransformerThreeHot, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
        )  # Transformer 모델 초기화

        self.tie_tgt_embeddings = tie_tgt_embeddings  # 타겟 임베딩 공유 여부

        self._src_dict = src_dict
        self._tgt_input_dict = tgt_dict
        self._tgt_input_embed_cls = tgt_embedding_cls
        self._tgt_dict = tgt_dict
        self.generator = decoder_cls(tgt_dict, emb_size, decoder_hidden_size)  # 디코더 생성
        self.src_tok_emb = AlphabetEmbedding(src_dict, emb_size)  # 소스 토큰 임베딩
        self.tgt_tok_emb = self._tgt_input_embed_cls(self._tgt_input_dict, emb_size)  # 타겟 토큰 임베딩

        if self.tie_tgt_embeddings:
            self._shared_i = nn.Parameter(
                torch.randn(self.tgt_tok_emb._emb_i.weight.data.shape)
            )  # 공유 가중치 생성
            self._shared_v = nn.Parameter(
                torch.randn(self.tgt_tok_emb._emb_v.weight.data.shape)
            )
            self._shared_f = nn.Parameter(
                torch.randn(self.tgt_tok_emb._emb_f.weight.data.shape)
            )

            del self.tgt_tok_emb._emb_i.weight
            del self.tgt_tok_emb._emb_v.weight
            del self.tgt_tok_emb._emb_f.weight

            self.tgt_tok_emb._emb_i.weight = self._shared_i
            self.tgt_tok_emb._emb_v.weight = self._shared_v
            self.tgt_tok_emb._emb_f.weight = self._shared_f

            if decoder_cls in [UnrolledRNNDecoder, UnrolledDiagonalRNNDecoder]:
                del self.generator._reembed_i.weight
                del self.generator._reembed_v.weight

                self.generator._reembed_i.weight = self._shared_i
                self.generator._reembed_v.weight = self._shared_v

                del self.generator._fc_i.weight
                del self.generator._fc_v.weight
                del self.generator._fc_f.weight

                self.generator._fc_i.weight = self._shared_i
                self.generator._fc_v.weight = self._shared_v
                self.generator._fc_f.weight = self._shared_f

            elif decoder_cls in [ThreeHotIndependentDecoder]:
                del self.generator._fc_i.weight
                del self.generator._fc_v.weight
                del self.generator._fc_f.weight

                self.generator._fc_i.weight = self._shared_i
                self.generator._fc_v.weight = self._shared_v
                self.generator._fc_f.weight = self._shared_f

        else:
            self._shared_i = nn.Parameter()
            self._shared_v = nn.Parameter()
            self._shared_f = nn.Parameter()

        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout, maxlen=max_len
        )  # 위치 인코딩 초기화

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
        teacher_force=False,
        force=None,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))  # 소스 임베딩에 위치 인코딩 적용
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))  # 타겟 임베딩에 위치 인코딩 적용
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )  # Transformer 모델 적용
        return self.generator(outs, force=force if teacher_force else None)  # 디코더를 통해 출력 생성

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )  # 인코더를 통해 소스 인코딩

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )  # 디코더를 통해 타겟 디코딩