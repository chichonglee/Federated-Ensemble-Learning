#!/usr/bin/env python3
# models.py
#
# 模型與訓練/預測相關工具：
# - MLP: PyTorch 多層感知器 (20 features → 5 classes)
# - train_local_mlp: 在單一 client 上訓練 MLP
# - train_centralized_rf_xgb: 集中式訓練 RF 與 XGB（FedEL ensemble 用）
# - ensemble_predict: 使用 (MLP + RF + XGB) 做多數決預測

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. PyTorch MLP 模型定義
# ============================================================

class MLP(nn.Module):
    """
    對應論文與原 notebook 的 MLP 結構：
    - Input: 20 維特徵
    - Hidden1: 256, ReLU, Dropout(0.4)
    - Hidden2: 256, ReLU, Dropout(0.4)
    - Output: 5 類別（不經 Softmax，在 loss 裡處理）
    """

    def __init__(
        self,
        input_dim: int = 20,
        num_classes: int = 5,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 2. 單一 client 上的本地訓練函式
# ============================================================

@dataclass
class TrainResult:
    model: MLP
    last_loss: float


def train_local_mlp(
    model: MLP,
    dataloader: DataLoader,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> TrainResult:
    """
    在單一 client 的 DataLoader 上訓練 MLP 一段時間（若干 epochs）。

    參數
    ----
    model      : MLP 模型（已載入 global 權重）
    dataloader : 該 client 的訓練資料
    epochs     : 本地訓練的 epoch 數
    lr         : SGD 學習率
    momentum   : SGD momentum
    weight_decay : L2 regularization，預設 0
    device     : 'cuda' 或 'cpu'，若為 None 則自動選擇

    回傳
    ----
    TrainResult(model, last_loss)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    last_loss = 0.0

    model.train()
    for _ in range(epochs):
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

    return TrainResult(model=model, last_loss=last_loss)


# ============================================================
# 3. 集中式 RF / XGB 訓練（FedEL ensemble 用）
# ============================================================

@dataclass
class EnsembleModels:
    rf: RandomForestClassifier
    xgb: XGBClassifier


def train_centralized_rf_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: Optional[Dict[str, Any]] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> EnsembleModels:
    """
    使用所有集中式訓練資料訓練 RF 與 XGB，作為 FedEL 的 ensemble base learner。

    預設參數可以依照論文設定調整，
    這裡先給一組合理且與你之前設定相近的配置。
    """
    if rf_params is None:
        rf_params = dict(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    if xgb_params is None:
        xgb_params = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=5,
            tree_method="hist",
            random_state=42,
        )

    rf = RandomForestClassifier(**rf_params)
    xgb = XGBClassifier(**xgb_params)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return EnsembleModels(rf=rf, xgb=xgb)


# ============================================================
# 4. FedEL 多數決預測：MLP + RF + XGB
# ============================================================

def ensemble_predict(
    global_model: MLP,
    ensemble_models: EnsembleModels,
    X_test: np.ndarray,
    device: Optional[torch.device] = None,
    num_classes: int = 5,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用 global MLP + RF + XGB 做多數決預測（FedEL）。

    參數
    ----
    global_model    : 已經 FedAvg 後的全域 MLP
    ensemble_models : 包含 rf 與 xgb 的 EnsembleModels
    X_test          : 測試資料 (N, D) numpy array
    device          : 'cuda' 或 'cpu'，None 則自動選擇
    num_classes     : 類別數（預設 5）
    verbose         : 若 True，會印出 accuracy 與 classification_report

    回傳
    ----
    y_pred_final : 最終多數決結果 (N,)
    info         : dict，內含中間結果，例如各模型預測與 accuracy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = global_model.to(device)
    global_model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        logits = global_model(X_t)
        pred_mlp = torch.argmax(logits, dim=1).cpu().numpy()

    # RF / XGB 使用 numpy 直接預測
    pred_rf = ensemble_models.rf.predict(X_test)
    pred_xgb = ensemble_models.xgb.predict(X_test)

    # (N, 3)
    preds_stack = np.stack([pred_mlp, pred_rf, pred_xgb], axis=1)

    def majority_vote(row: np.ndarray) -> int:
        counts = np.bincount(row, minlength=num_classes)
        return int(np.argmax(counts))

    y_pred_final = np.apply_along_axis(majority_vote, 1, preds_stack)

    info: Dict[str, Any] = {
        "pred_mlp": pred_mlp,
        "pred_rf": pred_rf,
        "pred_xgb": pred_xgb,
        "pred_stack": preds_stack,
    }

    if verbose:
        # 如果你在外面有 y_test，就可以另外傳進來計算 accuracy 與 report
        # 這裡只留下介面，真正的 y_test 在 train_fedel.py 裡處理
        print("ensemble_predict 完成（若需 accuracy / report，請在外部計算）")

    return y_pred_final, info


def aggregate_rf_models(
    local_rfs: List[RandomForestClassifier],
    sizes: List[int],
    method: str = "last",
) -> RandomForestClassifier:
    """
    Fed-RF 聚合函式的預留介面。

    目前實作：
    - method="last": 回傳最後一個 client 的 RF（等價於原 Keras 程式「最後一個 client 覆蓋 global_model3」的做法）
    - 之後你要做：
        - "best_acc"（挑 validation 表現最佳的 RF）
        - "bagging"（合併所有 local trees 做一個大 RF）
      都可以在這裡擴充。
    """
    if len(local_rfs) == 0:
        raise ValueError("local_rfs is empty")

    if method == "last":
        return local_rfs[-1]

    # 預留空間：之後可以在這裡實作其他策略
    raise NotImplementedError(f"RF aggregation method '{method}' is not implemented yet.")


def aggregate_xgb_models(
    local_xgbs: List[XGBClassifier],
    sizes: List[int],
    method: str = "last",
) -> XGBClassifier:
    """
    Fed-XGB 聚合函式的預留介面。

    目前實作：
    - method="last": 回傳最後一個 client 的 XGB
      → 完全對齊原 Keras 程式：每輪迭代結束時 global_model2 = local_model2（最後一個 client）
    """
    if len(local_xgbs) == 0:
        raise ValueError("local_xgbs is empty")

    if method == "last":
        return local_xgbs[-1]

    # 預留空間：之後可實作 tree-level 聚合、stacking 等方法
    raise NotImplementedError(f"XGB aggregation method '{method}' is not implemented yet.")

# ============================================================
# 5. 簡單自我測試用（直接執行 models.py 時）
# ============================================================

if __name__ == "__main__":
    # 簡單測試模型 forward，不做訓練
    model = MLP(input_dim=20, num_classes=5, dropout=0.4)
    x_dummy = torch.randn(4, 20)
    logits = model(x_dummy)
    print("MLP logits shape:", logits.shape)
