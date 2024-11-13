from setuptools import setup, find_packages

setup(
    name="SRNNAutoEncoder",  # 패키지 이름
    version="0.1.0", # 버전 업데이트 시 변경
    packages=find_packages(),
    install_requires=[  # 필요한 라이브러리 목록
        "numpy",
        "torch",
    ],
    author="Jihwan Lee (abcd-EGH)",
    description="Sparcely connections RNN + AutoEncoder Model for Anomaly Detection in Time Series",
    url="https://github.com/abcd-EGH/srnn-ae",
)
