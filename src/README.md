# __init.py__ 
패키지 초기화 파일로, 디렉토리를 패키지로 인식하게 합니다.

# common.py
공통적으로 사용되는 함수 및 클래스를 정의합니다.

# data_batching.py
데이터 배칭 관련 기능을 제공하며, 학습 시 데이터셋을 배치 단위로 나누어 처리합니다.

# decoding_layers.py
`UnrolledRNNDecoder`와 `UnrolledDiagonalRNNDecoder`를 구현하여 자모(jamo) 삼중음절(triplet)의 3단계 RNN 디코딩을 구현하여 한글 자모를 입력으로 받아 음절을 생성합니다.

# dictionaries.py
단어와 인덱스 간의 매핑을 관리하는 사전을 정의합니다.

# embedding_layers.py
임베딩 레이어를 정의하여 입력 데이터를 벡터 형태로 변환합니다.

# loss_function.py
다양한 손실 함수를 정의하여 모델 학습에 사용됩니다. 각 손실 함수는 모델의 성능을 평가하고 개선하는 데 중요한 역할을 합니다.

# perplexity_functions.py
퍼플렉서티(perplexity)를 계산하는 함수들을 정의하여 모델 성능을 평가합니다.

# training.py
모델 학습을 위한 주요 트레이닝 루프와 관련 기능을 제공합니다.

# transformers.py
트랜스포머 모델을 정의하고, 이를 통해 Seq2Seq 모델을 구현합니다.

# utils.py
다양한 유틸리티 함수들을 포함하여 데이터 로드, 모델 저장 및 로깅 등의 작업을 지원합니다.