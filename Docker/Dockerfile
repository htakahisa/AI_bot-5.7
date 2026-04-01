# PyTorch公式のCUDA 12.1対応イメージを使用
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

WORKDIR /app

# タイムゾーン等のインタラクティブな質問を避ける設定
ENV DEBIAN_FRONTEND=noninteractive

# システム依存ライブラリとPythonパッケージ
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    onnx

COPY . .

# 実行時にGPUが認識されているか確認するスクリプトをデフォルトに
CMD ["python", "train.py"]