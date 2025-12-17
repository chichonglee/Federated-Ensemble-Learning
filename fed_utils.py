#!/usr/bin/env python3
# fed_utils.py
#
# Utilities for Federated Learning:
# - create_clients: 將 (X_train, y_train) 切成多個 client 的 DataLoader（記憶體版）
# - create_test_loader: 建立 test DataLoader
# - average_weights: FedAvg 權重加權平均
# - partition_and_save_clients: 切分資料並存到 Datasets/client_x 資料夾中

from __future__ import annotations

from typing import Dict, List, Tuple

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_clients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int = 5,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    將整體的 X_train, y_train 切成 num_clients 份，
    回傳 dict: { "client_1": DataLoader, ..., "client_K": DataLoader }。
    每個 client 的筆數會盡量平均。

    參數
    ----
    X_train : (N, D) numpy array
    y_train : (N,) numpy array
    num_clients : 要幾個 client
    batch_size : DataLoader 的 batch size
    shuffle : DataLoader 是否啟用 shuffle
    seed : 隨機種子，確保每次切分穩定可重現
    """
    assert len(X_train) == len(y_train), "X_train 與 y_train 長度必須相同"

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X_train))
    rng.shuffle(indices)

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    shard_size = len(X_shuffled) // num_clients
    clients: Dict[str, DataLoader] = {}

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else len(X_shuffled)

        X_c = X_shuffled[start:end]
        y_c = y_shuffled[start:end]

        ds = TensorDataset(
            torch.from_numpy(X_c).float(),
            torch.from_numpy(y_c).long()
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        clients[f"client_{i+1}"] = loader

    return clients


def create_test_loader(
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int | None = None,
) -> DataLoader:
    """
    建立 test DataLoader。
    如果 batch_size=None，就用「整個 test set 一個 batch」的方式（方便一次評估整體）。
    """
    if batch_size is None:
        batch_size = len(X_test)

    ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def average_weights(
    state_dicts: List[dict],
    sizes: List[int],
) -> dict:
    """
    FedAvg：依照每個 client 的樣本數加權平均 model weights。

    參數
    ----
    state_dicts : 各 client local 模型的 state_dict 列表
    sizes       : 各 client 的資料筆數，長度需與 state_dicts 相同

    回傳
    ----
    new_state : 加權平均後的 state_dict（可直接 load_state_dict）
    """
    assert len(state_dicts) == len(sizes), "state_dicts 與 sizes 長度必須一致"
    total = float(sum(sizes))
    assert total > 0, "總樣本數必須 > 0"

    # 用第一個 state_dict 的 key 當基準
    new_state = {}
    for key in state_dicts[0].keys():
        new_state[key] = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)

    for state, size in zip(state_dicts, sizes):
        weight = size / total
        for key in new_state.keys():
            new_state[key] += state[key].float() * weight

    return new_state


def partition_and_save_clients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_dir: str = "Datasets",
    num_clients: int = 5,
    seed: int = 42,
    prefix: str = "client_",
    overwrite: bool = True,
) -> None:
    """
    將 X_train, y_train 切成 num_clients 份，並將每份資料存到:
        {base_dir}/{prefix}{i}/data.npz
    例如:
        Datasets/client_1/data.npz
        Datasets/client_2/data.npz
        ...

    data.npz 內含兩個陣列:
        - X : (n_i, D)
        - y : (n_i,)

    參數
    ----
    X_train, y_train : 完整訓練資料
    base_dir         : 根資料夾名稱，例如 "Datasets"
    num_clients      : client 數量
    seed             : 用於隨機打散資料的 seed
    prefix           : client 資料夾前綴，例如 "client_" → client_1, client_2, ...
    overwrite        : 若該資料夾已存在，是否允許覆寫 data.npz
    """
    assert len(X_train) == len(y_train), "X_train 與 y_train 長度必須相同"

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X_train))
    rng.shuffle(indices)

    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    shard_size = len(X_shuffled) // num_clients

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else len(X_shuffled)

        X_c = X_shuffled[start:end]
        y_c = y_shuffled[start:end]

        client_name = f"{prefix}{i+1}"
        client_dir = base_path / client_name
        client_dir.mkdir(parents=True, exist_ok=True)

        out_path = client_dir / "data.npz"
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} already exists and overwrite=False")
            continue

        np.savez(out_path, X=X_c, y=y_c)
        print(f"[SAVE] {client_name}: {X_c.shape[0]} samples -> {out_path}")


def load_client_dataset(
    client_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    從指定 client 資料夾載入 data.npz，回傳 (X, y)。

    用法範例:
        X_c, y_c = load_client_dataset("Datasets/client_1")
    """
    client_path = Path(client_dir)
    npz_path = client_path / "data.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"找不到 {npz_path}")

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    return X, y


if __name__ == "__main__":
    # 簡單本地測試
    X_dummy = np.random.randn(1000, 20).astype(np.float32)
    y_dummy = np.random.randint(0, 5, size=(1000,)).astype(np.int64)

    print("切分並存到 Datasets/client_x...")
    partition_and_save_clients(
        X_dummy,
        y_dummy,
        base_dir="Datasets",
        num_clients=5,
        seed=42,
        prefix="client_",
        overwrite=True,
    )

    # 測試載入其中一個 client
    X_c1, y_c1 = load_client_dataset("Datasets/client_1")
    print("client_1 shape:", X_c1.shape, y_c1.shape)