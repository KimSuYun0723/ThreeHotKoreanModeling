import random

from .common import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, is_korean_syllable

def batch_by_target_size(data, max_tokens=4000):
    """
    이 함수는 타겟 측 토큰 수에 따라 데이터를 배치합니다. 문장은 (타겟 측) 길이에 따라 정렬되지만,
    동일한 길이의 문장은 배치 전에 셔플되어 매번 동일한 배치가 되지 않도록 합니다.

    배치는 최대 `max_tokens` 토큰까지 그리디하게 채워집니다. 따라서 비슷한 길이의 문장이 한 배치로 그룹화됩니다.
    """
    # 데이터를 타겟 시퀀스 길이에 따라 정렬하지만, 동일한 길이의 문장은 무작위로 셔플
    unstable_sorted = sorted(data, key=lambda x: (len(x[1]), random.random()))

    out = []  # 최종 배치를 저장할 리스트
    running_batch = []  # 현재 배치를 저장할 리스트
    cur_count = 0  # 현재 배치의 토큰 수를 추적할 변수
    for (s, t) in unstable_sorted:
        # 현재 타겟 시퀀스를 추가하면 max_tokens를 초과할 경우
        if cur_count + len(t) > max_tokens:
            out.append(running_batch)  # 현재 배치를 최종 배치 리스트에 추가
            running_batch = [(s, t)]  # 새로운 배치 시작
            cur_count = len(t)  # 현재 배치의 토큰 수 재설정
        else:
            running_batch.append((s, t))  # 현재 시퀀스를 현재 배치에 추가
            cur_count += len(t)  # 현재 배치의 토큰 수 업데이트
    if running_batch != []:
        out.append(running_batch)  # 남아 있는 배치를 최종 배치 리스트에 추가
    return out  # 최종 배치 리스트 반환

def collate_fn(batch, src_dict, tgt_dict):
    """
    클래스 레이블 소스와 타겟 시퀀스를 배치로 빌드합니다.
    """
    src_batch, tgt_batch = [], []  # 소스와 타겟 배치를 저장할 리스트 초기화
    for src_sample, tgt_sample in batch:
        # 각 소스와 타겟 샘플에 SOS와 EOS 토큰 추가
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])

    # 소스와 타겟 배치를 인코딩
    src_batch = src_dict.encode_batch(src_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    return src_batch, tgt_batch  # 인코딩된 소스와 타겟 배치 반환

def _subchar_multiples(x):
    """
    이 함수는 토큰의 하위 문자 수를 결정하는 헬퍼 메서드입니다.
    패드는 0, 비한국어는 1, 한국어 음절은 3을 반환합니다.
    """
    if x == PAD_TOKEN:
        return 0  # 패딩 토큰은 0 반환
    elif is_korean_syllable(x):
        return 3  # 한국어 음절은 3 반환
    else:
        return 1  # 그 외에는 1 반환

def collate_fn_syllable_perplexity(batch, src_dict, tgt_dict):
    """
    클래스 레이블 소스와 타겟 시퀀스를 배치로 빌드합니다.
    이 함수는 하위 문자 인식이 없는 음절 디코더에서 사용됩니다. 우리는 또한 각 토큰의 하위 문자 수
    (한국어인 경우 3, 그렇지 않은 경우 1)를 유지하여 퍼플렉시티 계산에서 비트-퍼-하위 문자를 계산하는 데 사용할 수 있습니다.
    """
    src_batch, tgt_batch = [], []  # 소스와 타겟 배치를 저장할 리스트 초기화
    for src_sample, tgt_sample in batch:
        # 각 소스와 타겟 샘플에 SOS와 EOS 토큰 추가
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])
    # 소스와 타겟 배치를 인코딩
    src_batch = src_dict.encode_batch(src_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    # 타겟 배치의 각 토큰에 대해 하위 문자 수 계산
    subchars = [[_subchar_multiples(x) for x in tgt_dict.decode(t)] for t in tgt_batch]
    return src_batch, tgt_batch, subchars  # 인코딩된 소스, 타겟 배치와 하위 문자 수 반환

def collate_fn_triple(batch, src_dict, tgt_input_dict, tgt_dict):
    """
    클래스 레이블 소스와 타겟 시퀀스를 배치로 빌드합니다.
    이 함수는 타겟 측 인코더와 디코더가 다른 경우(예: 3 핫 인코더 및 음절 디코더 또는 그 반대)에 사용됩니다.

    출력은 (source, target_input, target_output)입니다.
    """
    src_batch, tgt_input_batch, tgt_batch = [], [], []  # 소스, 타겟 입력, 타겟 배치를 저장할 리스트 초기화
    for src_sample, tgt_sample in batch:
        # 각 소스와 타겟 샘플에 SOS와 EOS 토큰 추가
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_input_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])

    # 소스 배치를 인코딩
    src_batch = src_dict.encode_batch(src_batch)
    # 타겟 입력 배치를 타겟 입력 사전으로 인코딩
    tgt_input_batch = tgt_input_dict.encode_batch(tgt_batch)
    # 타겟 배치를 타겟 사전으로 인코딩
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    return src_batch, tgt_input_batch, tgt_batch  # 인코딩된 소스, 타겟 입력, 타겟 배치 반환