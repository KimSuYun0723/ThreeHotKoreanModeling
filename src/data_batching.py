import random

from .common import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, is_korean_syllable

def batch_by_target_size(data, max_tokens=4000): # 시퀀스 --> 배치로 묶기
    """
    1. 데이터를 정렬함
    - 타겟 측 토큰 수에 따라 데이터를 배치
    - 문장은 (타겟 측) 길이에 따라 정렬
    - 동일한 길이의 문장은 배치 전에 셔플 
    --> 매번 동일한 배치가 되지 않도록(특정 배치에 ovefitting 방지, randomness 증대)

    2. 배치 생성
    - 배치는 최대 `max_tokens` 토큰까지 greedy하게 채워지도록
    --> 비슷한 길이의 문장이 한 배치로 그룹화
    """

    # 데이터를 타겟 시퀀스 길이에 따라 정렬 & 동일한 길이의 문장은 무작위로 셔플
    unstable_sorted = sorted(data, key=lambda x: (len(x[1]), random.random()))

    out = []  # 최종 배치를 저장할 리스트
    running_batch = []  # 현재 배치를 저장할 리스트
    cur_count = 0  # 현재 배치의 토큰 수를 추적할 변수
    
    for (s, t) in unstable_sorted:
        # 현재 타겟 시퀀스를 추가 --> max_tokens를 초과할 경우
        if cur_count + len(t) > max_tokens:
            out.append(running_batch)  # 현재 배치를 최종 배치 리스트에 추가
            running_batch = [(s, t)]  # 새로운 배치 시작
            cur_count = len(t)  # 현재 배치의 토큰 수 재설정
        
        # 현재 타겟 시퀀스를 추가 --> max_tokens를 초과하지 않을 경우
        else:
            running_batch.append((s, t))  # 현재 시퀀스를 현재 배치에 추가
            cur_count += len(t)  # 현재 배치의 토큰 수 업데이트
    
    # 배치가 비어있지 않으면 최종 배치 리스트 반환
    if running_batch != []:
        out.append(running_batch)  # 남아 있는 배치를 최종 배치 리스트에 추가
    return out  # 최종 배치 리스트 반환


def collate_fn(batch, src_dict, tgt_dict): # 배치 구성
    """
    : 모델에 입력할 배치를 빌드하는 역할
    1 . 소스, 타켓 데이터 
    """
    src_batch, tgt_batch = [], []  # 소스와 타겟 배치를 저장할 리스트 초기화
    
    for src_sample, tgt_sample in batch: # 소스(입력 문장), 타겟(정답 예측 데이터)
        # 각 소스와 타겟 샘플에 SOS와 EOS 토큰 추가하여 시작과 끝을 명확하게
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])

    # 소스 & 타겟 배치 인코딩
    src_batch = src_dict.encode_batch(src_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    return src_batch, tgt_batch  # 인코딩된 소스와 타겟 배치 반환

# 이 함수는 나중에 perplexity 계산 시에 활용됨(낮을 수록 성능 good)
def _subchar_multiples(x): #음절/비한국어/패딩인지에 따라 특정 값 반환
    """
    : 토큰의 하위 문자 수를 결정하는 헬퍼 메서드입니다.
    패드 = 0
    비한국어 = 1
    한국어 음절 = 3 반환
    """
    if x == PAD_TOKEN:
        return 0  # 패딩 토큰은 0 반환
    elif is_korean_syllable(x):
        return 3  # 한국어 음절은 3 반환
    else:
        return 1  # 그 외에는 1 반환

# 음절 기반 perplexity 계산 시에 활용
def collate_fn_syllable_perplexity(batch, src_dict, tgt_dict):
    """
    : 클래스 레이블 소스와 타겟 시퀀스를 배치로 빌드
    이 함수는 하위 문자 인식이 없는 음절 디코더에서 사용됩니다. 우리는 또한 각 토큰의 하위 문자 수
    (한국어인 경우 3, 그렇지 않은 경우 1)를 유지하여 퍼플렉시티 계산에서 비트-퍼-하위 문자를 계산하는 데 사용할 수 있습니다.
    """
    src_batch, tgt_batch = [], []  # 소스와 타겟 배치를 저장할 리스트 초기화
    for src_sample, tgt_sample in batch:
        # 각 소스와 타겟 샘플에 SOS와 EOS 토큰 추가(collate_fn 함수와 비슷)
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN])
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN])
    # 소스와 타겟 배치를 인코딩
    src_batch = src_dict.encode_batch(src_batch)
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    # 타겟 배치의 각 토큰에 대해 하위 문자 수 계산
    subchars = [[_subchar_multiples(x) for x in tgt_dict.decode(t)] for t in tgt_batch]
    return src_batch, tgt_batch, subchars  # 인코딩된 소스, 타겟 배치와 하위 문자 수 반환
    # 반환된 하위 문자 수로 나중에 .perplexity_functions.py에서 perplexity 계산

# 타겟 측 인코더와 디코더가 다른 경우
def collate_fn_triple(batch, src_dict, tgt_input_dict, tgt_dict):
    """
    : 클래스 레이블 소스와 타겟 시퀀스를 배치로 빌드
    타겟 측 인코더와 디코더가 다른 경우에 사용할 수 있음.
        EX) 인코더는 three-hot encoding, 디코더는 syllable decoding
    소스 시퀀스, 타겟 입력 시퀀스, 타겟 출력 시퀀스를 각각 처리
    """
    src_batch, tgt_input_batch, tgt_batch = [], [], []  # 소스, 타겟 입력, 타겟 배치를 저장할 리스트 초기화
    for src_sample, tgt_sample in batch:
        # 각 소스와 타겟 샘플에 SOS와 EOS 토큰 추가
        src_batch.append([SOS_TOKEN] + src_sample + [EOS_TOKEN]) # 소스 시퀀스
        tgt_input_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN]) # 타겟 입력 시퀀스
        tgt_batch.append([SOS_TOKEN] + tgt_sample + [EOS_TOKEN]) # 타겟 출력 시퀀스

    # 소스 배치를 인코딩
    src_batch = src_dict.encode_batch(src_batch)
    # 타겟 입력 배치를 타겟 입력 사전으로 인코딩
    tgt_input_batch = tgt_input_dict.encode_batch(tgt_batch)
    # 타겟 배치를 타겟 사전으로 인코딩
    tgt_batch = tgt_dict.encode_batch(tgt_batch)
    return src_batch, tgt_input_batch, tgt_batch  # 인코딩된 소스, 타겟 입력, 타겟 배치 반환