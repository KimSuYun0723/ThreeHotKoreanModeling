"""
TODO: 모델을 직렬화할 때 모든 인스턴스 변수를 직렬화하여 모델을 재구성할 수 있도록 함
"""

import torch  # PyTorch 모듈 임포트
import torch.nn as nn  # 신경망 모듈 임포트

import logging  # 로깅 모듈 임포트

from .transformers import Seq2SeqTransformer, Seq2SeqTransformerThreeHot  # 필요한 Transformer 클래스 임포트
from .decoding_layers import (  # 디코딩 레이어 임포트
    ThreeHotIndependentDecoder,
    UnrolledDiagonalRNNDecoder,
    UnrolledRNNDecoder,
)

DEVICE = None  # 디바이스 초기화

def build_transformer(
    src_dict,
    tgt_dict,
    max_len=300,
    FFN_HID_DIM=512,
    ENCODER_LAYERS=6,
    DECODER_LAYERS=6,
    NHEADS=8,
    EMB_SIZE=512,
    tie_tgt_embeddings=False,
    device=None
):
    torch.manual_seed(0)  # 랜덤 시드 설정

    transformer = Seq2SeqTransformer(  # Seq2SeqTransformer 모델 초기화
        src_dict,
        tgt_dict,
        ENCODER_LAYERS,
        DECODER_LAYERS,
        EMB_SIZE,
        NHEADS,
        FFN_HID_DIM,
        max_len=max_len,
        tie_tgt_embeddings=tie_tgt_embeddings,
    )

    for p in transformer.parameters():  # 모델의 파라미터 초기화
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # Xavier 유니폼 분포로 초기화

    transformer = transformer.to(device)  # 모델을 지정된 디바이스로 이동

    optimizer = torch.optim.Adam(  # Adam 옵티마이저 초기화
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-8
    )
    return src_dict, tgt_dict, transformer, optimizer  # 소스 딕셔너리, 타겟 딕셔너리, 모델, 옵티마이저 반환

def build_transformer_threehot(
    src_dict,
    tgt_dict,
    FFN_HID_DIM=512,
    EMB_SIZE=512,
    ENCODER_LAYERS=6,
    DECODER_LAYERS=6,
    NHEADS=8,
    decoder_cls=UnrolledRNNDecoder,
    tie_tgt_embeddings=False,
    max_len=300,
    device=None,
):
    torch.manual_seed(0)  # 랜덤 시드 설정

    transformer = Seq2SeqTransformerThreeHot(  # Seq2SeqTransformerThreeHot 모델 초기화
        src_dict,
        tgt_dict,
        decoder_cls,
        ENCODER_LAYERS,
        DECODER_LAYERS,
        EMB_SIZE,
        NHEADS,
        FFN_HID_DIM,
        FFN_HID_DIM,
        tie_tgt_embeddings=tie_tgt_embeddings,
        max_len=max_len,
    )

    for p in transformer.parameters():  # 모델의 파라미터 초기화
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # Xavier 유니폼 분포로 초기화

    transformer = transformer.to(device)  # 모델을 지정된 디바이스로 이동

    optimizer = torch.optim.Adam(  # Adam 옵티마이저 초기화
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-8
    )
    return src_dict, tgt_dict, transformer, optimizer  # 소스 딕셔너리, 타겟 딕셔너리, 모델, 옵티마이저 반환

three_hot_cond_builder = (
    lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer_threehot(
        src_dict,
        tgt_dict,
        decoder_cls=UnrolledRNNDecoder,
        tie_tgt_embeddings=tie_target_embeddings,
    )
)  # UnrolledRNNDecoder를 사용하는 three hot 조건부 빌더

three_hot_cond_diag_builder = (
    lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer_threehot(
        src_dict,
        tgt_dict,
        decoder_cls=UnrolledDiagonalRNNDecoder,
        tie_tgt_embeddings=tie_target_embeddings,
    )
)  # UnrolledDiagonalRNNDecoder를 사용하는 three hot 조건부 빌더

three_hot_ind_builder = (
    lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer_threehot(
        src_dict,
        tgt_dict,
        decoder_cls=ThreeHotIndependentDecoder,
        tie_tgt_embeddings=tie_target_embeddings,
    )
)  # ThreeHotIndependentDecoder를 사용하는 three hot 독립형 빌더

jamo_builder = lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer(
    src_dict, tgt_dict, max_len=500, tie_tgt_embeddings=tie_target_embeddings
)  # 자모 빌더

syllable_builder = lambda src_dict, tgt_dict, tie_target_embeddings: build_transformer(
    src_dict, tgt_dict, tie_tgt_embeddings=tie_target_embeddings
)  # 음절 빌더

parameter_reallocation_builder = lambda src_dict, tgt_dict, tie_target_embedings: build_transformer_threehot(
    src_dict,
    tgt_dict,
    ENCODER_LAYERS=7,
    DECODER_LAYERS=7,
    FFN_HID_DIM=1024,
    decoder_cls=UnrolledDiagonalRNNDecoder,
)  # 파라미터 재할당 빌더

def load_model(path, model_builder):
    checkpoint = torch.load(path, map_location='cpu')  # 체크포인트 로드
    epoch = checkpoint["epoch"]  # 에폭 로드
    tie_embeddings = checkpoint["tie_embeddings"]  # 임베딩 공유 여부 로드
    en_dict = checkpoint["en_dict"]  # 영어 딕셔너리 로드
    kr_dict = checkpoint["kr_dict"]  # 한국어 딕셔너리 로드
    src_dict, tgt_dict, model, opt = model_builder(en_dict, kr_dict, tie_embeddings)  # 모델 빌더로 모델 생성
    model.load_state_dict(checkpoint["model_state_dict"])  # 모델 상태 로드
    opt.load_state_dict(checkpoint["optimizer_state_dict"])  # 옵티마이저 상태 로드

    return src_dict, tgt_dict, model, opt, epoch  # 소스 딕셔너리, 타겟 딕셔너리, 모델, 옵티마이저, 에폭 반환

def serialize_model_and_opt(path, model, opt, src_dict, tgt_dict, epoch):
    logging.info(f"serializing {path}")  # 직렬화 정보 로깅

    torch.save(
        {
            "tie_embeddings": model.tie_tgt_embeddings,  # 임베딩 공유 여부 저장
            "epoch": epoch,  # 에폭 저장
            "model_state_dict": model.state_dict(),  # 모델 상태 저장
            "optimizer_state_dict": opt.state_dict(),  # 옵티마이저 상태 저장
            "en_dict": src_dict,  # 영어 딕셔너리 저장
            "kr_dict": tgt_dict,  # 한국어 딕셔너리 저장
        },
        path,  # 저장 경로
    )