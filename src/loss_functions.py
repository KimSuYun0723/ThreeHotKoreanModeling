import warnings  # 경고 메시지를 처리하기 위해 warnings 모듈 임포트

import torch.nn as nn  # PyTorch의 신경망 모듈 임포트

def onehot_loss(logits, preds, tgt_dict):
    """
    패딩 인덱스를 쉽게 설정할 수 있도록 딕셔너리를 사용하는 크로스 엔트로피 손실 함수의 래퍼입니다.
    """
    crit = nn.CrossEntropyLoss(ignore_index=tgt_dict.pad())  # 패딩 인덱스를 무시하는 크로스 엔트로피 손실 함수 설정

    return crit(logits, preds)  # logits와 preds에 대한 손실 값 반환

def threehot_loss(i, v, f, I, V, F, tgt_dict):
    """
    모델에 구애받지 않는 퍼플렉시티 측정치는 비트-퍼-하위 문자입니다.
    이 손실 함수는 그 관계를 완전히 포착하지는 못하며, 손실은 총 하위 문자 수가 아닌 각 하위 문자 클래스에 대해 평균됩니다.

    `threehot_loss_per_subcharacter`는 퍼플렉시티 측정치와 일치합니다.
    """
    pad_i, pad_v, pad_f = tgt_dict.pad()  # 패딩 인덱스 가져오기
    crit_i = nn.CrossEntropyLoss(ignore_index=pad_i, reduction="mean")  # 초성에 대한 크로스 엔트로피 손실 함수 설정
    crit_v = nn.CrossEntropyLoss(ignore_index=pad_v, reduction="mean")  # 중성에 대한 크로스 엔트로피 손실 함수 설정
    crit_f = nn.CrossEntropyLoss(ignore_index=pad_f, reduction="mean")  # 종성에 대한 크로스 엔트로피 손실 함수 설정

    return crit_i(i, I) + crit_v(v, V) + crit_f(f, F)  # 각 하위 문자에 대한 손실 값을 더하여 반환

def threehot_loss_per_subcharacter(i, v, f, I, V, F, tgt_dict):
    """
    이 손실 함수는 각 토큰에 대한 총 크로스 엔트로피 손실을 계산하고
    총 (패딩이 아닌) 하위 문자 수로 나눕니다. 이를 통해 비트-퍼-하위 문자 메트릭에
    더 정확한 손실 함수를 제공합니다.
    """
    pad_i, pad_v, pad_f = tgt_dict.pad()  # 패딩 인덱스 가져오기
    crit_i = nn.CrossEntropyLoss(ignore_index=pad_i, reduction="sum")  # 초성에 대한 크로스 엔트로피 손실 함수 설정
    crit_v = nn.CrossEntropyLoss(ignore_index=pad_v, reduction="sum")  # 중성에 대한 크로스 엔트로피 손실 함수 설정
    crit_f = nn.CrossEntropyLoss(ignore_index=pad_f, reduction="sum")  # 종성에 대한 크로스 엔트로피 손실 함수 설정

    losses = crit_i(i, I) + crit_v(v, V) + crit_f(f, F)  # 각 하위 문자에 대한 손실 값을 더함
    multiples = (I != pad_i).long() + (V != pad_v).long() + (F != pad_f).long()  # 패딩이 아닌 하위 문자의 수를 계산
    return losses / sum(multiples)  # 총 손실 값을 하위 문자 수로 나누어 반환

def threehot_loss_weighted(i, v, f, I, V, F, tgt_dict):
    warnings.warn(
        "the name `threehot_loss_weighted` is deprecated, use `threehot_loss_per_subcharacter` instead",
        DeprecationWarning,
    )  # 'threehot_loss_weighted' 이름이 더 이상 사용되지 않음을 경고
    return threehot_loss_per_subcharacter(i, v, f, I, V, F, tgt_dict)  # `threehot_loss_per_subcharacter` 함수를 호출하여 반환