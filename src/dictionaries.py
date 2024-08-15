from .common import *  # common 모듈에서 모든 것을 가져옴
from .common import _NO_JAMO_TOKEN, _INITIAL_JAMO, _VOWEL_JAMO, _FINAL_JAMO  # common 모듈에서 특정 항목들 가져옴

# 한글 음절을 자모(초성, 중성, 종성)로 분해하여 각각의 자모를 인코딩/디코딩
class ThreeHotDict:
    # 예약된 심볼들 설정 : 인코딩에 사용되지 않을 심볼들 미리 설정
    _RESERVED = set(SPECIAL_SYMBOLS + _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO)  

    def __init__(self, symbols):
        assert all(s not in self._RESERVED for s in symbols)  # 모든 심볼이 예약된 심볼에 포함되지 않는지 확인
        self._ini = _INITIAL_JAMO + symbols + SPECIAL_SYMBOLS  # 초성 심볼
        self._vow = _VOWEL_JAMO + [_NO_JAMO_TOKEN, PAD_TOKEN]  # 중성 심볼
        self._fin = _FINAL_JAMO + [_NO_JAMO_TOKEN, PAD_TOKEN]  # 종성 심볼

        self._ini2idx = {e: i for (i, e) in enumerate(self._ini)}  # 초성 심볼을 인덱스로 매핑
        self._idx2ini = {i: e for (i, e) in enumerate(self._ini)}  # 인덱스를 초성 심볼로 매핑
        self._vow2idx = {e: i for (i, e) in enumerate(self._vow)}  # 중성 심볼을 인덱스로 매핑
        self._idx2vow = {i: e for (i, e) in enumerate(self._vow)}  # 인덱스를 중성 심볼로 매핑
        self._fin2idx = {e: i for (i, e) in enumerate(self._fin)}  # 종성 심볼을 인덱스로 매핑
        self._idx2fin = {i: e for (i, e) in enumerate(self._fin)}  # 인덱스를 종성 심볼로 매핑

        self._sos = (
            self._ini2idx[SOS_TOKEN],
            self._vow2idx[_NO_JAMO_TOKEN],
            self._fin2idx[_NO_JAMO_TOKEN],
        )  # 시작 토큰 설정
        self._eos = (
            self._ini2idx[EOS_TOKEN],
            self._vow2idx[_NO_JAMO_TOKEN],
            self._fin2idx[_NO_JAMO_TOKEN],
        )  # 종료 토큰 설정
        self._unk = (
            self._ini2idx[UNK_TOKEN],
            self._vow2idx[_NO_JAMO_TOKEN],
            self._fin2idx[_NO_JAMO_TOKEN],
        )  # 알 수 없는 토큰 설정
        self._pad = (
            self._ini2idx[PAD_TOKEN],
            self._vow2idx[PAD_TOKEN],
            self._fin2idx[PAD_TOKEN],
        )  # 패딩 토큰 설정

    def __getitem__(self, ele): # 초성, 중성, 종성 인덱스로 변환
        if ele == PAD_TOKEN:
            return self._pad  # 패딩 토큰일 경우 패딩 인덱스 반환

        if not is_korean_syllable(ele):
            return (
                self._ini2idx.get(ele, self._ini2idx[UNK_TOKEN]),
                self._vow2idx[_NO_JAMO_TOKEN],
                self._fin2idx[_NO_JAMO_TOKEN],
            )  # 한국어 음절이 아닌 경우 해당 인덱스 반환

        i, v, f = get_jamo(ele)
        return self._ini2idx[i], self._vow2idx[v], self._fin2idx[f]  # 한국어 음절인 경우 자모 분해 후 인덱스 반환

    def sizes(self):
        return len(self._ini), len(self._vow), len(self._fin)  # 초성, 중성, 종성의 크기 반환

    def encode(self, sentence, pad_len=None):
        if pad_len is not None:
            return [self[c] for c in sentence] + [self._pad] * (pad_len - len(sentence))  # 문장을 인코딩하고 패딩 추가
        else:
            return [self[c] for c in sentence]  # 문장을 인코딩

    def encode_batch(self, sentences):
        pad_len = max(len(s) for s in sentences)
        return [self.encode(s, pad_len) for s in sentences]  # 배치 내 최대 길이로 패딩을 맞춘 후 인코딩

    def decode(self, encoded):
        return [
            (self._idx2ini[i], self._idx2vow[v], self._idx2fin[f])
            for (i, v, f) in encoded
        ]  # 인코딩된 값을 자모로 디코딩

    def decode_batch(self, encoded_batch):
        return [self.decode(e) for e in encoded_batch]  # 배치 내 모든 값을 디코딩

    def __len__(self):
        return len(self._ini) + len(self._vow) + len(self._fin)  # 전체 심볼의 길이 반환

    def sos(self):
        return self._sos  # 시작 토큰 반환

    def eos(self):
        return self._eos  # 종료 토큰 반환

    def unk(self):
        return self._unk  # 알 수 없는 토큰 반환

    def pad(self):
        return self._pad  # 패딩 토큰 반환


class ThreeHotDictArbitraryOrdering:
    _RESERVED = set(SPECIAL_SYMBOLS + _INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO)  # 예약된 심볼들 설정
    _JAMO = set(_INITIAL_JAMO + _VOWEL_JAMO + _FINAL_JAMO)  # 자모 설정
    _JAMO_GROUPS = {"i": _INITIAL_JAMO, "v": _VOWEL_JAMO, "f": _FINAL_JAMO}  # 자모 그룹 설정
    _ORDER_PERM = {
        ("i", "v", "f"): (0, 1, 2),
        ("i", "f", "v"): (0, 2, 1),
        ("v", "i", "f"): (1, 0, 2),
        ("v", "f", "i"): (1, 2, 0),
        ("f", "i", "v"): (2, 0, 1),
        ("f", "v", "i"): (2, 1, 0),
    }  # 자모 순서에 대한 순열 설정

    def __init__(self, symbols, ordering=("i", "v", "f")):
        assert all(s not in self._RESERVED for s in symbols)  # 모든 심볼이 예약된 심볼에 포함되지 않는지 확인

        self._ordering = ordering  # 순서 설정
        self._order_perm = self._ORDER_PERM[self._ordering]  # 순서에 따른 순열 설정
        first, second, third = self._ordering

        self._first = self._JAMO_GROUPS[first] + symbols + SPECIAL_SYMBOLS  # 첫 번째 그룹 심볼 설정
        self._second = self._JAMO_GROUPS[second] + [_NO_JAMO_TOKEN, PAD_TOKEN]  # 두 번째 그룹 심볼 설정
        self._third = self._JAMO_GROUPS[third] + [_NO_JAMO_TOKEN, PAD_TOKEN]  # 세 번째 그룹 심볼 설정

        self._first2idx = {e: i for (i, e) in enumerate(self._first)}  # 첫 번째 그룹 심볼을 인덱스로 매핑
        self._idx2first = {i: e for (i, e) in enumerate(self._first)}  # 인덱스를 첫 번째 그룹 심볼로 매핑
        self._second2idx = {e: i for (i, e) in enumerate(self._second)}  # 두 번째 그룹 심볼을 인덱스로 매핑
        self._idx2second = {i: e for (i, e) in enumerate(self._second)}  # 인덱스를 두 번째 그룹 심볼로 매핑
        self._third2idx = {e: i for (i, e) in enumerate(self._third)}  # 세 번째 그룹 심볼을 인덱스로 매핑
        self._idx2third = {i: e for (i, e) in enumerate(self._third)}  # 인덱스를 세 번째 그룹 심볼로 매핑

        self._sos = (
            self._first2idx[SOS_TOKEN],
            self._second2idx[_NO_JAMO_TOKEN],
            self._third2idx[_NO_JAMO_TOKEN],
        )  # 시작 토큰 설정
        self._eos = (
            self._first2idx[EOS_TOKEN],
            self._second2idx[_NO_JAMO_TOKEN],
            self._third2idx[_NO_JAMO_TOKEN],
        )  # 종료 토큰 설정
        self._unk = (
            self._first2idx[UNK_TOKEN],
            self._second2idx[_NO_JAMO_TOKEN],
            self._third2idx[_NO_JAMO_TOKEN],
        )  # 알 수 없는 토큰 설정
        self._pad = (
            self._first2idx[PAD_TOKEN],
            self._second2idx[PAD_TOKEN],
            self._third2idx[PAD_TOKEN],
        )  # 패딩 토큰 설정

    def __getitem__(self, ele):
        if ele is PAD_TOKEN:
            return self._pad  # 패딩 토큰일 경우 패딩 인덱스 반환

        if not is_korean_syllable(ele):
            return (
                self._first2idx.get(ele, self._first2idx[UNK_TOKEN]),
                self._second2idx[_NO_JAMO_TOKEN],
                self._third2idx[_NO_JAMO_TOKEN],
            )  # 한국어 음절이 아닌 경우 해당 인덱스 반환

        i, v, f = self._apply_perm(get_jamo(ele))
        return self._first2idx[i], self._second2idx[v], self._third2idx[f]  # 한국어 음절인 경우 자모 분해 후 순서에 따라 인덱스 반환

    def sizes(self):
        return len(self._first), len(self._second), len(self._third)  # 첫 번째, 두 번째, 세 번째 그룹의 크기 반환

    def encode(self, sentence, pad_len=None):
        if pad_len is not None:
            return [self[c] for c in sentence] + [self._pad] * (pad_len - len(sentence))  # 문장을 인코딩하고 패딩 추가
        else:
            return [self[c] for c in sentence]  # 문장을 인코딩

    def encode_batch(self, sentences):
        pad_len = max(len(s) for s in sentences)
        return [self.encode(s, pad_len) for s in sentences]  # 배치 내 최대 길이로 패딩을 맞춘 후 인코딩

    def _apply_perm(self, l):
        return tuple(l[i] for i in self._order_perm)  # 순서를 적용하여 자모 반환

    def _reverse_perm(self, l):
        out = [None, None, None]  # 역순서를 적용하여 자모 반환
        for i, v in enumerate(self._order_perm):
            out[v] = l[i]

        return tuple(out)

    def decode(self, encoded):
        return [
            (self._idx2first[i], self._idx2second[v], self._idx2third[f])
            if self._idx2first[i] not in self._JAMO
            else self._reverse_perm(
                (self._idx2first[i], self._idx2second[v], self._idx2third[f])
            )
            for (i, v, f) in encoded
        ]  # 인코딩된 값을 자모로 디코딩

    def decode_batch(self, encoded_batch):
        return [self.decode(e) for e in encoded_batch]  # 배치 내 모든 값을 디코딩

    def __len__(self):
        return len(self._first) + len(self._second) + len(self._third)  # 전체 심볼의 길이 반환

    def sos(self):
        return self._sos  # 시작 토큰 반환

    def eos(self):
        return self._eos  # 종료 토큰 반환

    def unk(self):
        return self._unk  # 알 수 없는 토큰 반환

    def pad(self):
        return self._pad  # 패딩 토큰 반환


class AlphabetDict:
    _RESERVED = set(SPECIAL_SYMBOLS)  # 예약된 심볼 설정

    def __init__(self, symbols):
        assert all(s not in self._RESERVED for s in symbols)  # 모든 심볼이 예약된 심볼에 포함되지 않는지 확인
        self._alphabet = SPECIAL_SYMBOLS + symbols  # 알파벳에 특수 기호와 주어진 심볼을 추가

        self._ele2idx = {e: i for (i, e) in enumerate(self._alphabet)}  # 각 심볼에 대해 인덱스를 매핑하는 딕셔너리 생성
        self._idx2ele = {i: e for (i, e) in enumerate(self._alphabet)}  # 인덱스를 심볼로 매핑하는 딕셔너리 생성

        self._sos = self._ele2idx[SOS_TOKEN]  # 시작 토큰 설정
        self._eos = self._ele2idx[EOS_TOKEN]  # 종료 토큰 설정
        self._unk = self._ele2idx[UNK_TOKEN]  # 알 수 없는 토큰 설정
        self._pad = self._ele2idx[PAD_TOKEN]  # 패딩 토큰 설정

    def __getitem__(self, ele):
        return self._ele2idx.get(ele, self._unk)  # 심볼에 해당하는 인덱스를 반환, 없으면 알 수 없는 토큰 반환

    def encode(self, sentence, pad_len=None):
        if pad_len is not None:
            return [self._ele2idx.get(c, self._unk) for c in sentence] + [self._pad] * (
                pad_len - len(sentence)
            )  # 문장을 인코딩하고 필요하면 패딩 추가
        else:
            return [self._ele2idx.get(c, self._unk) for c in sentence]  # 문장을 인코딩

    def encode_batch(self, sentences):
        pad_len = max(len(s) for s in sentences)
        return [self.encode(s, pad_len) for s in sentences]  # 배치 내 최대 길이로 패딩을 맞춘 후 인코딩

    def decode(self, encoded):
        return [self._idx2ele[c] for c in encoded]  # 인코딩된 값을 원래 심볼로 디코딩

    def decode_batch(self, encoded_batch):
        return [self.decode(e) for e in encoded_batch]  # 배치 내 모든 값을 디코딩

    def __len__(self):
        return len(self._alphabet)  # 전체 알파벳의 길이 반환

    def sos(self):
        return self._sos  # 시작 토큰 반환

    def eos(self):
        return self._eos  # 종료 토큰 반환

    def unk(self):
        return self._unk  # 알 수 없는 토큰 반환

    def pad(self):
        return self._pad  # 패딩 토큰 반환