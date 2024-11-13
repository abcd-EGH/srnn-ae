# SLSTMAutoEncoder와 SGRUAutoEncoder 모듈의 주요 클래스를 임포트
from .slstm_autoencoder import SLSTMAutoEncoder
from .sgru_autoencoder import SGRUAutoEncoder
# from .SLSTMVAE import SLSTMVAE # 새로운 클래스 추가 시 임포트

# 패키지 버전 정보 (선택 사항)
__version__ = "0.1.0" # 버전 업데이트 시 변경

# 외부에서 접근 가능한 클래스 목록 정의
__all__ = ["SLSTMAutoEncoder", "SGRUAutoEncoder"] # 외부에서 접근 가능한 클래스 목록
