#!/usr/bin/env python3
# trains.py
#
# Federated-Ensemble Learning (FedEL) training script (PyTorch + sklearn 版)
# - Fed-MLP: MLP 使用 FedAvg 聚合
# - RF / XGB: 目前依原始 Keras 程式邏輯，以「最後一個 client」的模型作為 global RF/XGB
#   （透過 aggregate_rf_models / aggregate_xgb_models 預留 Fed-RF / Fed-XGB 擴充空間）

import argparse
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from data_preprocess import preprocess_5sec_dataset
from fed_utils import create_clients, create_test_loader, average_weights
from models import (
    MLP,
    TrainResult,
    train_local_mlp,
    train_centralized_rf_xgb,
    EnsembleModels,
    ensemble_predict,
    aggregate_rf_models,
    aggregate_xgb_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Federated-Ensemble Learning (FedEL) on 5-second dataset."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="dataset_5secondWindow.csv",
        help="Path to 5-second window CSV dataset.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of federated clients.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Local DataLoader batch size.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of communication rounds (comms_round).",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per round.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for local MLP training.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer in local MLP training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # 固定隨機種子，方便重現結果
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------
    # 1. 資料前處理
    # --------------------------------------------------------
    print(f"[INFO] Loading and preprocessing dataset from: {args.csv_path}")
    data = preprocess_5sec_dataset(
        csv_path=args.csv_path,
        corr_threshold=0.7,
        test_size=0.2,
        random_state=args.seed,
        stratify=True,
    )

    X_train, X_test = data.X_train, data.X_test
    y_train, y_test = data.y_train, data.y_test

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"[INFO] X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[INFO] y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"[INFO] #features={n_features}, #classes={n_classes}")

    # --------------------------------------------------------
    # 2. 切成多個 client 的 DataLoader
    # --------------------------------------------------------
    print(f"[INFO] Creating {args.num_clients} clients (batch_size={args.batch_size}) ...")
    clients = create_clients(
        X_train,
        y_train,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    print("[INFO] Clients:", list(clients.keys()))

    # test loader：一次全 batch，方便每輪直接完整評估
    test_loader = create_test_loader(X_test, y_test, batch_size=None)

    # --------------------------------------------------------
    # 3. 初始化 global 模型 (MLP, RF, XGB)
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    global_mlp = MLP(input_dim=n_features, num_classes=n_classes, dropout=0.4).to(device)
    global_state = global_mlp.state_dict()

    # RF / XGB 的「global 模型」一開始先設為 None
    global_rf = None
    global_xgb = None

    # --------------------------------------------------------
    # 4. Federated training loop
    # --------------------------------------------------------
    print(f"[INFO] Start federated training for {args.rounds} rounds ...")

    for comm_round in range(args.rounds):
        print(f"\n=== Communication Round {comm_round + 1}/{args.rounds} ===")

        # 收集 local MLP weights + client size（用於 FedAvg）
        local_states = []
        local_sizes = []

        # 收集 local RF / XGB（用於 global RF/XGB 聚合）
        local_rf_models = []
        local_xgb_models = []

        # 隨機打亂 client 順序（對齊原 Keras 程式的做法）
        client_names = list(clients.keys())
        random.shuffle(client_names)

        for cname in client_names:
            loader = clients[cname]
            n_samples = len(loader.dataset)

            # ----------------------------
            # 4.1 Local MLP：從 global 初始化，做本地訓練
            # ----------------------------
            local_mlp = MLP(input_dim=n_features, num_classes=n_classes, dropout=0.4)
            local_mlp.load_state_dict(global_state)

            result: TrainResult = train_local_mlp(
                model=local_mlp,
                dataloader=loader,
                epochs=args.local_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=0.0,
                device=device,
            )

            local_states.append(result.model.state_dict())
            local_sizes.append(n_samples)

            # ----------------------------
            # 4.2 從 DataLoader 抽出 numpy，訓練 local RF / XGB
            #     → 對齊原始程式中，將 client 的 batch 展開為 arr/labelpd 再 fit()
            # ----------------------------
            X_local_batches = []
            y_local_batches = []
            for xb, yb in loader:
                X_local_batches.append(xb.numpy())
                y_local_batches.append(yb.numpy())
            X_local = np.concatenate(X_local_batches, axis=0)
            y_local = np.concatenate(y_local_batches, axis=0)

            ensemble_models = train_centralized_rf_xgb(X_local, y_local)
            local_rf_models.append(ensemble_models.rf)
            local_xgb_models.append(ensemble_models.xgb)

            print(
                f"  [Client {cname}] samples={n_samples}, "
                f"last_local_loss={result.last_loss:.4f}"
            )

        # ----------------------------
        # 4.3 FedAvg：聚合所有 local MLP → 更新 global MLP
        # ----------------------------
        global_state = average_weights(local_states, local_sizes)
        global_mlp.load_state_dict(global_state)

        # ----------------------------
        # 4.4 RF / XGB 全域模型：
        #     目前採用「取最後一個 client」，
        #     完全對齊原 Keras 中 global_model2 = local_model2 / global_model3 = local_model3 的邏輯。
        #     未來若要做 Fed-RF / Fed-XGB，只要修改 aggregate_* 即可。
        # ----------------------------
        global_rf = aggregate_rf_models(local_rf_models, local_sizes, method="last")
        global_xgb = aggregate_xgb_models(local_xgb_models, local_sizes, method="last")
        global_ensemble = EnsembleModels(rf=global_rf, xgb=global_xgb)

        # ----------------------------
        # 4.5 評估：FedEL = global MLP + global RF/XGB 多數決
        # ----------------------------
        X_test_batch, y_test_batch = next(iter(test_loader))
        X_test_np = X_test_batch.numpy()
        y_test_np = y_test_batch.numpy()

        y_pred, _ = ensemble_predict(
            global_model=global_mlp,
            ensemble_models=global_ensemble,
            X_test=X_test_np,
            device=device,
            num_classes=n_classes,
        )
        acc = accuracy_score(y_test_np, y_pred)
        print(f"  [FedEL] Test Accuracy after round {comm_round + 1}: {acc:.4f}")

    # --------------------------------------------------------
    # 5. 最終報告
    # --------------------------------------------------------
    print("\n=== Final Evaluation (FedEL) ===")
    print(classification_report(y_test_np, y_pred, digits=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
