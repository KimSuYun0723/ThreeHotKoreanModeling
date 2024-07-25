import torch  # PyTorch 모듈을 임포트
import torch.nn as nn  # PyTorch의 신경망 모듈 임포트
import torch.nn.functional as F  # PyTorch의 함수형 API 임포트

from .common import *  # common 모듈에서 모든 것을 임포트
from data_batching import *  # data_batching 모듈에서 모든 것을 임포트
from .transformers import create_mask, create_mask_threehot  # 필요한 함수 임포트
from .utils import load_model, serialize_model_and_opt  # 필요한 유틸리티 함수 임포트
from .perplexity_functions import (  # 필요한 퍼플렉시티 함수 임포트
    perplexity_syllable_batched,
    perplexity_jamo_batched,
    perplexity_threehot_per_class_batched,
)
from .loss_functions import (  # 필요한 손실 함수 임포트
    threehot_loss,
)

import os, logging, random, time  # 운영 체제 인터페이스, 로깅, 무작위, 시간 모듈 임포트

def train_epoch_target_token_batch(
    model, optimizer, train_data, src_dict, tgt_dict, device=None
):
    """하나의 에폭 동안 모델을 학습합니다 (음절 및 자모 모델 전용).
    배치는 타겟 측의 토큰 수에 의해 형성됩니다.

    `train_epoch_target_token_batch_threehot`의 threehot 버전을 참조하십시오.

    Args:
        model: 모델
        optimizer: 옵티마이저
        train_data: (src, tgt) 쌍의 집합. threehot/syllable의 경우 tgt 측은
                    단순히 음절 시퀀스입니다. 자모의 경우 자모 시퀀스입니다.
                    `filtered_bpe_translation_train_pairs`와 `filtered_bpe_jamo_translation_train_pairs`를 참조하십시오.
        src_dict: 소스 언어 딕셔너리
        tgt_dict: 타겟 언어 딕셔너리
        batch_size: 각 배치의 토큰 수 (기본값 4000)
    """
    model.train()  # 모델을 학습 모드로 설정
    losses = 0  # 손실 초기화
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad())  # 패딩 인덱스를 무시하는 크로스 엔트로피 손실 함수 설정

    random.shuffle(train_data)  # 학습 데이터를 무작위로 섞음

    for batch in train_data:  # 각 배치에 대해 반복
        src, tgt = batch  # 소스와 타겟 데이터를 배치에서 가져옴

        src = src.transpose(0, 1)  # 소스 텐서의 차원 교환
        tgt = tgt.transpose(0, 1)  # 타겟 텐서의 차원 교환

        tgt_input = tgt[:-1, :]  # 타겟 입력 설정

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(  # 마스크 생성
            src, tgt_input, src_dict, tgt_dict, device=device
        )

        logits, _ = model(  # 모델에 소스와 타겟 입력, 마스크를 입력하여 logits 계산
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()  # 옵티마이저의 기울기를 초기화
        tgt_out = tgt[1:, :]  # 타겟 출력 설정
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))  # 손실 계산
        loss.backward()  # 역전파 수행

        optimizer.step()  # 옵티마이저를 사용하여 파라미터 업데이트
        losses += loss.item()  # 총 손실에 현재 손실 추가

        del loss  # 손실 변수 삭제

    return losses / len(train_data)  # 평균 손실 반환

def train_epoch_target_token_batch_threehot(
    model,
    optimizer,
    train_data,
    src_dict,
    tgt_dict,
    loss_fn=threehot_loss,
    device=None,
):
    """하나의 에폭 동안 모델을 학습합니다 (threehot 전용).
    배치는 타겟 측의 토큰 수(음절 수)에 의해 형성됩니다.

    Args:
        model: 모델
        optimizer: 옵티마이저
        train_data: (src, tgt) 쌍의 집합. threehot/syllable의 경우 tgt 측은
                    단순히 음절 시퀀스입니다. 자모의 경우 자모 시퀀스입니다.
                    `filtered_bpe_translation_train_pairs`와 `filtered_bpe_jamo_translation_train_pairs`를 참조하십시오.
        src_dict: 소스 언어 딕셔너리
        tgt_dict: 타겟 언어 딕셔너리
    """
    model.train()  # 모델을 학습 모드로 설정
    losses = 0  # 손실 초기화
    random.shuffle(train_data)  # 학습 데이터를 무작위로 섞음

    for batch in train_data:  # 각 배치에 대해 반복
        src, tgt = batch  # 소스와 타겟 데이터를 배치에서 가져옴

        src = src.transpose(0, 1)  # 소스 텐서의 차원 교환
        tgt = tgt.transpose(0, 1)  # 타겟 텐서의 차원 교환

        tgt_input = tgt[:-1, :]  # 타겟 입력 설정

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask_threehot(  # threehot 마스크 생성
            src, tgt_input, src_dict, tgt_dict, device=device,
        )

        tgt_out = tgt[1:, :]  # 타겟 출력 설정

        logits, _ = model(  # 모델에 소스와 타겟 입력, 마스크를 입력하여 logits 계산
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
            teacher_force=True,
            force=tgt_out,
        )
        i, v, f = logits  # logits에서 초성, 중성, 종성 분리
        optimizer.zero_grad()  # 옵티마이저의 기울기를 초기화

        loss = loss_fn(  # 손실 함수 호출하여 손실 계산
            i.reshape(-1, i.shape[-1]),
            v.reshape(-1, v.shape[-1]),
            f.reshape(-1, f.shape[-1]),
            tgt_out[..., 0].reshape(-1),
            tgt_out[..., 1].reshape(-1),
            tgt_out[..., 2].reshape(-1),
            tgt_dict,
        )

        loss.backward()  # 역전파 수행

        optimizer.step()  # 옵티마이저를 사용하여 파라미터 업데이트
        losses += loss.item()  # 총 손실에 현재 손실 추가

        del loss  # 손실 변수 삭제

    return losses / len(train_data)  # 평균 손실 반환

def train_syllable(
    epochs,
    model,
    optimizer,
    src_dict,
    tgt_dict,
    train_data,
    test_data,
    output_dir,
    output_name,
    batch_size=4000,
    output_every=1,
    device=None,
):
    """주어진 에폭 수 동안 음절 모델을 학습합니다.
    모델은 주기적으로 직렬화되며, 각 에폭에서 메트릭 딕셔너리가 기록됩니다.

    Args:
        epochs: 학습할 에폭 수
        model: 모델
        optimizer: 옵티마이저
        src_dict: 소스 언어 딕셔너리
        tgt_dict: 타겟 언어 딕셔너리
        batch_size: 각 배치의 토큰 수 (기본값 4000)
        train_data: 학습을 위한 (src, tgt) 쌍의 집합. `filtered_bpe_translation_train_pairs` 참조
        test_data: 테스트를 위한 (src, tgt) 쌍의 집합. `filtered_bpe_translation_valid_pairs` 참조
        output_dir: 직렬화된 모델을 저장할 디렉터리
        output_name: 모델 이름의 접두사 (예: `SYLLABLE`). 에폭이 모델 이름에 추가됨 (예: SYLLABLE_EPOCH_30.mod)
        output_every: `N` 에폭마다 모델 저장 및 직렬화
    """
    results = {
        "ACCURACY": [],  # 정확도 기록
        "LOSS_PER_SUBCHAR_TEST": [],  # 테스트 손실 기록
        "LOSS_PER_SUBCHAR_TRAIN": [],  # 학습 손실 기록
        "PERPLEXITY_TEST": [],  # 테스트 퍼플렉시티 기록
        "PERPLEXITY_TRAIN": [],  # 학습 퍼플렉시티 기록
        "TIME": [],  # 학습 시간 기록
    }

    path = os.path.join(output_dir, output_name)  # 출력 경로 설정

    batched_data = batch_by_target_size(train_data, batch_size)  # 데이터를 배치 크기에 맞게 나눔
    random.shuffle(batched_data)  # 배치를 무작위로 섞음

    tensor_batches = []

    for b in batched_data:  # 각 배치에 대해 반복
        s, t = collate_fn(b, src_dict, tgt_dict)  # 배치 데이터를 텐서로 변환

        tensor_batches.append(
            (torch.tensor(s, device=device), torch.tensor(t, device=device))
        )

    for epoch in range(1, epochs + 1):  # 각 에폭에 대해 반복
        start = time.time()  # 시작 시간 기록
        train_epoch_target_token_batch(
            model,
            optimizer,
            tensor_batches,
            src_dict,
            tgt_dict,
            batch_size=batch_size,
            device=device,
        )
        results["TIME"].append(time.time() - start)  # 학습 시간 기록

        perp_test = perplexity_syllable_batched(
            model, test_data, src_dict, tgt_dict, device=device
        )
        perp_train = perplexity_syllable_batched(
            model, random.sample(train_data, 5000), src_dict, tgt_dict, device=device
        )

        results["PERPLEXITY_TEST"].append(perp_test)  # 테스트 퍼플렉시티 기록
        results["PERPLEXITY_TRAIN"].append(perp_train)  # 학습 퍼플렉시티 기록

        logging.info(f"name={output_name}, epoch={epoch}, metrics:{results}")  # 로깅

        if epoch % output_every == 0:  # 지정된 에폭마다 모델 저장 및 직렬화
            serialize_model_and_opt(
                f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
            )

    if epoch % output_every != 0:  # 마지막 에폭에서 모델 저장 및 직렬화
        serialize_model_and_opt(
            f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
        )

    return model, optimizer, results  # 학습된 모델, 옵티마이저, 결과 반환

def train_jamo(
    epochs,
    model,
    optimizer,
    src_dict,
    tgt_dict,
    train_data,
    test_data,
    output_dir,
    output_name,
    batch_size=10000,
    output_every=1,
    device=None,
):
    """주어진 에폭 수 동안 자모 모델을 학습합니다.
    모델은 주기적으로 직렬화되며, 각 에폭에서 메트릭 딕셔너리가 기록됩니다.

    Args:
        epochs: 학습할 에폭 수
        model: 모델
        optimizer: 옵티마이저
        src_dict: 소스 언어 딕셔너리
        tgt_dict: 타겟 언어 딕셔너리
        batch_size: 각 배치의 토큰 수 (기본값 10000)
        train_data: 학습을 위한 (src, tgt) 쌍의 집합. `filtered_bpe_jamo_translation_train_pairs` 참조
        test_data: 테스트를 위한 (src, tgt) 쌍의 집합. `filtered_bpe_jamo_translation_valid_pairs` 참조
        output_dir: 직렬화된 모델을 저장할 디렉터리
        output_name: 모델 이름의 접두사 (예: `JAMO`). 에폭이 모델 이름에 추가됨 (예: JAMO_EPOCH_30.mod)
        output_every: `N` 에폭마다 모델 저장 및 직렬화
    """
    results = {
        "ACCURACY": [],  # 정확도 기록
        "LOSS_PER_SUBCHAR_TEST": [],  # 테스트 손실 기록
        "LOSS_PER_SUBCHAR_TRAIN": [],  # 학습 손실 기록
        "PERPLEXITY_TEST": [],  # 테스트 퍼플렉시티 기록
        "PERPLEXITY_TRAIN": [],  # 학습 퍼플렉시티 기록
        "TIME": [],  # 학습 시간 기록
    }

    path = os.path.join(output_dir, output_name)  # 출력 경로 설정

    batched_data = batch_by_target_size(train_data, batch_size)  # 데이터를 배치 크기에 맞게 나눔
    random.shuffle(batched_data)  # 배치를 무작위로 섞음

    tensor_batches = []

    for b in batched_data:  # 각 배치에 대해 반복
        s, t = collate_fn(b, src_dict, tgt_dict)  # 배치 데이터를 텐서로 변환

        tensor_batches.append(
            (torch.tensor(s, device=device), torch.tensor(t, device=device))
        )

    for epoch in range(1, epochs + 1):  # 각 에폭에 대해 반복
        start = time.time()  # 시작 시간 기록
        train_epoch_target_token_batch(
            model,
            optimizer,
            tensor_batches,
            src_dict,
            tgt_dict,
            batch_size=batch_size,
            device=device
        )
        results["TIME"].append(time.time() - start)  # 학습 시간 기록

        perp_test = perplexity_jamo_batched(model, test_data, src_dict, tgt_dict, device=device)  # 테스트 퍼플렉시티 계산
        perp_train = perplexity_jamo_batched(model, random.sample(train_data, 6000), src_dict, tgt_dict, device=device)  # 학습 퍼플렉시티 계산

        results["PERPLEXITY_TEST"].append(perp_test)  # 테스트 퍼플렉시티 기록
        results["PERPLEXITY_TRAIN"].append(perp_train)  # 학습 퍼플렉시티 기록

        logging.info(f"name={output_name}, epoch={epoch}, metrics:{results}")  # 로깅

        if epoch % output_every == 0:  # 지정된 에폭마다 모델 저장 및 직렬화
            serialize_model_and_opt(
                f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
            )

    if epoch % output_every != 0:  # 마지막 에폭에서 모델 저장 및 직렬화
        serialize_model_and_opt(
            f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
        )

    return model, optimizer, results  # 학습된 모델, 옵티마이저, 결과 반환

def train_threehot(
    epochs,
    model,
    optimizer,
    src_dict,
    tgt_dict,
    train_data,
    test_data,
    output_dir,
    output_name,
    batch_size=4000,
    output_every=1,
    device=None,
    checkpoint_offset=0,
):
    """주어진 에폭 수 동안 threehot 모델을 학습합니다. 이 함수는 모든 threehot 모델
    (조건부 또는 독립형, 모든 출력 순서)에 대해 작동합니다.

    모델은 주기적으로 직렬화되며, 각 에폭에서 메트릭 딕셔너리가 기록됩니다.

    Args:
        epochs: 학습할 에폭 수
        model: 모델
        optimizer: 옵티마이저
        src_dict: 소스 언어 딕셔너리
        tgt_dict: 타겟 언어 딕셔너리
        batch_size: 각 배치의 토큰 수 (기본값 4000)
        train_data: 학습을 위한 (src, tgt) 쌍의 집합. `filtered_bpe_translation_train_pairs` 참조
        test_data: 테스트를 위한 (src, tgt) 쌍의 집합. `filtered_bpe_translation_valid_pairs` 참조
        output_dir: 직렬화된 모델을 저장할 디렉터리
        output_name: 모델 이름의 접두사 (예: `THREEHOT_FIV`). 에폭이 모델 이름에 추가됨 (예: THREEHOT_FIV_EPOCH_30.mod)
        output_every: `N` 에폭마다 모델 저장 및 직렬화
    """
    results = {
        "ACCURACY": [],  # 정확도 기록
        "LOSS_PER_SUBCHAR_TEST": [],  # 테스트 손실 기록
        "LOSS_PER_SUBCHAR_TRAIN": [],  # 학습 손실 기록
        "PERPLEXITY_TEST": [],  # 테스트 퍼플렉시티 기록
        "PERPLEXITY_TRAIN": [],  # 학습 퍼플렉시티 기록
        "PERPLEXITY_TEST_NO_NON_JAMO": [],  # 비자모 제외 테스트 퍼플렉시티 기록
        "PERPLEXITY_TRAIN_NO_NON_JAMO": [],  # 비자모 제외 학습 퍼플렉시티 기록
        "TIME": [],  # 학습 시간 기록
    }

    path = os.path.join(output_dir, output_name)  # 출력 경로 설정

    batched_data = batch_by_target_size(train_data, batch_size)  # 데이터를 배치 크기에 맞게 나눔
    random.shuffle(batched_data)  # 배치를 무작위로 섞음

    tensor_batches = []

    for b in batched_data:  # 각 배치에 대해 반복
        s, t = collate_fn(b, src_dict, tgt_dict)  # 배치 데이터를 텐서로 변환

        tensor_batches.append(
            (torch.tensor(s, device=device), torch.tensor(t, device=device))
        )

    for epoch in range(checkpoint_offset + 1, epochs + 1):  # 각 에폭에 대해 반복
        start = time.time()  # 시작 시간 기록
        train_epoch_target_token_batch_threehot(
            model,
            optimizer,
            tensor_batches,
            src_dict,
            tgt_dict,
            batch_size=batch_size,
            device=device
        )
        results["TIME"].append(time.time() - start)  # 학습 시간 기록

        perp_test = perplexity_threehot_per_class_batched(
            model, test_data, src_dict, tgt_dict, device=device
        )
        
        results["PERPLEXITY_TEST"].append(perp_test)  # 테스트 퍼플렉시티 기록

        logging.info(f"name={output_name}, epoch={epoch}, metrics:{results}")  # 로깅

        if epoch % output_every == 0:  # 지정된 에폭마다 모델 저장 및 직렬화
            serialize_model_and_opt(
                f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
            )

    if epoch % output_every != 0:  # 마지막 에폭에서 모델 저장 및 직렬화
        serialize_model_and_opt(
            f"{path}_EPOCH_{epoch}.mod", model, optimizer, src_dict, tgt_dict, epoch
        )

    return model, optimizer, results  # 학습된 모델, 옵티마이저, 결과 반환