from setuptools import setup, find_packages

setup(
    name="srnn-ae",  # 패키지 이름
    version="0.6.7", # 버전 업데이트 시 변경
    packages=find_packages(exclude=['test']),  # 패키지 디렉터리 목록
    install_requires=[  # 필요한 라이브러리 목록
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "seaborn",
        "arch",
    ],
    author="Jihwan Lee (abcd-EGH)",
    author_email="wlghks7790@gmail.com",
    license="MIT",
    description="Sparsely connections RNN + AutoEncoder Model for Anomaly Detection in Time Series",
    url="https://github.com/abcd-EGH/srnn-ae",
)
