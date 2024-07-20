import string

# 특수 토큰 정의
SOS_TOKEN, EOS_TOKEN, PUNC_TOKEN, NUMBER_TOKEN, OTHER_TOKEN, UNK_TOKEN, PAD_TOKEN = (
    "始", # 시작 토큰
    "末", # 종료 토큰
    "点", # 구두점 토큰
    "数", # 숫자 토큰
    "他", # 기타 토큰
    "不", # 알수 없는 토큰
    "無", # 패딩 토큰
)

# 특수 토큰 리스트
SPECIAL_SYMBOLS = [
    SOS_TOKEN,
    EOS_TOKEN,
    PUNC_TOKEN,
    NUMBER_TOKEN,
    OTHER_TOKEN,
    UNK_TOKEN,
    PAD_TOKEN,
]

# 호환되는 자모 리스트
COMPATIBILITY_JAMO = [
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]

# 유니코드 범위로 음절 리스트 생성
SYLLABLES = [chr(c) for c in range(0xAC00, 0xD7A3 + 1)]
_EMPTY_FINAL_JAMO = "☒" # 빈 종성(받침 없는 음절의 종성)
_NO_JAMO_TOKEN = "㋨" # 자모가 없음

# 유니코드 범위로 자모 리스트 생성
_INITIAL_JAMO = [chr(cp) for cp in range(0x1100, 0x1112 + 1)] # 초성
_VOWEL_JAMO = [chr(cp) for cp in range(0x1161, 0x1175 + 1)] # 중성(모음)
_FINAL_JAMO = [chr(cp) for cp in range(0x11A8, 0x11C2 + 1)] + ["☒"] # 종성 & 빈 종성

# 초성 --> 호환되는 자음으로 변환
_INIT2COMPAT = {i: c for (i, c) in zip(_INITIAL_JAMO, "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")}

# 중성 --> 호환되면 모음으로 변환
_VOW2COMPAT = {i: c for (i, c) in zip(_VOWEL_JAMO, "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")}

# 종성 --> 호환되는 자음(받침)으로 변환
_FINAL2COMPAT = {i: c for (i, c) in zip(_FINAL_JAMO, "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ☒")}

# 자모 --> 호환 자모로 변환
_JAMO2COMPAT = {
    i: c
    for (i, c) in zip(
        _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO,
        "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ☒",
    )
}

# 모든 자모 리스트
JAMO = _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO

#영어 알파벳
ENGLISH = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 구두점 문자 리스트
PUNCTUATION = [" "] + list(string.punctuation)


# 한글 음절인지 확인하는 함수
def is_korean_syllable(c): # c = (문자)
    return 44032 <= ord(c) <= 55203


# 한글 음절 --> 초/중/종성으로 변환
def get_jamo(c): # c = (한글 음절)
    if not is_korean_syllable(c):
        return c
    cp = ord(c)
    final = (cp - 44032) % 28
    vowel = 1 + ((cp - 44032 - final) % 588) // 28
    initial = 1 + (cp - 44032) // 588

    if final == 0: # 받침이 없으면
        return _INITIAL_JAMO[initial - 1], _VOWEL_JAMO[vowel - 1], _EMPTY_FINAL_JAMO
    else: # 받침이 있으면
        return (
            _INITIAL_JAMO[initial - 1],
            _VOWEL_JAMO[vowel - 1],
            _FINAL_JAMO[final - 1],
        )
    # (초성, 중성, 종성) 자모 튜플이 return됨


# 문자를 자모 삼중음절로 변환
def char_to_triplet(c): # c = (문자)
    if not is_korean_syllable(c):
        if c == PAD_TOKEN: # 한글 음절이면
            return (PAD_TOKEN, PAD_TOKEN, PAD_TOKEN) # 특수토큰
        else: # 한글 음절이 아니면 (c, x, x)
            return (c, _NO_JAMO_TOKEN, _NO_JAMO_TOKEN)
    return get_jamo(c) # (초성, 중성, 종성) or 특수 문자 튜플


# 자모 삼중음절이 완전한 음절인지 확인하는 함수
def is_full_syllable(i, v, f): # 초성, 중성, 종성
    return i in _INITIAL_JAMO and v in _VOWEL_JAMO and f in _FINAL_JAMO


# 자모인지 확인
def is_jamo(c):
    return c in _INITIAL_JAMO or c in _VOWEL_JAMO or c in _FINAL_JAMO


# 자모 삼중 음절을 표준화하는 함수
def canonicalize_triplets(triplet_seq):
    '''
    Here we canonicalize triplets by:

    1) 패딩 제거
    2) 불완전한 삼중 음절(자모와 비자모 혼합, 패딩과 비패딩 혼합 등)을 제거합니다.
    3) (x, ㋨, ㋨) 형태의 삼중 음절에서 _NO_JAMO_TOKENS를 제거합니다.
    4) 호환 자모로 변환하여 자모 문자열을 얻습니다.
    '''
    out = []
    for (i, v, f) in triplet_seq:

        if is_full_syllable(i, v, f): # 완전한 음절이면
            out.extend((i, v, f))
        else:
            # 패딩
            if i == PAD_TOKEN and v == PAD_TOKEN and f == PAD_TOKEN:
                pass
            # 영어
            elif v == _NO_JAMO_TOKEN and f == _NO_JAMO_TOKEN:
                if i == PAD_TOKEN or is_jamo(i):
                    out.append(UNK_TOKEN)
                else:
                    out.append(i)
            # 불완전한 삼중 음절
            else:
                out.append(UNK_TOKEN)
    return [_JAMO2COMPAT.get(c, c) for c in out] # 호환되는 자모로 변환
