#!/usr/bin/env python3
# check_gpu.py
# 檢查 PyTorch GPU 狀態與做簡單運算測試

import time
import torch


def main():
    print("=" * 60)
    print("PyTorch / CUDA 環境檢查")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.version.cuda: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\n目前 PyTorch 偵測不到 GPU，請檢查：")
        print("1. NVIDIA Driver 是否安裝在 Windows")
        print("2. WSL 是否有安裝 CUDA runtime / lib")
        print("3. 這個虛擬環境是否安裝 GPU 版 PyTorch")
        return

    # 顯示所有 GPU 資訊
    num_gpus = torch.cuda.device_count()
    print(f"\n可用 GPU 數量: {num_gpus}")

    for idx in range(num_gpus):
        props = torch.cuda.get_device_properties(idx)
        print(f"  [{idx}] {props.name}")
        print(f"       compute capability: {props.major}.{props.minor}")
        print(f"       total memory      : {props.total_memory / 1024 ** 3:.2f} GB")

    # 做一個小測試：在 GPU 上執行矩陣乘法
    device = torch.device("cuda:0")
    print("\n開始在 GPU 上做一次 10000 x 10000 的矩陣乘法測試...")

    # 預熱
    x = torch.randn(1000, 1000, device=device)
    y = x @ x
    torch.cuda.synchronize()

    # 實際計時測試
    size = 10000
    x = torch.randn(size, size, device=device)
    torch.cuda.synchronize()
    t0 = time.time()
    y = x @ x
    torch.cuda.synchronize()
    t1 = time.time()

    print(f"矩陣乘法耗時: {t1 - t0:.3f} 秒")
    print(f"結果張量平均值 (y.mean()): {y.mean().item():.6f}")

    print("\n檢查完成。GPU 正常工作。")


if __name__ == "__main__":
    main()
