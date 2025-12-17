#!/usr/bin/env python3
# data_preprocess.py
#
# Dataset preprocessing utilities for 5-second window transport mode detection
# 將原 Thesis_5SecondWindow_FederatedEnsemble.ipynb 中的前處理段落整理成可重用的函式

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 將文字標籤轉為數字標籤的對照表
TARGET_MAP: Dict[str, int] = {
    "Bus": 0,
    "Car": 1,
    "Still": 2,
    "Train": 3,
    "Walking": 4,
}


@dataclass
class PreprocessResult:
    """方便之後在其他模組使用時，打包所有前處理結果。"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    feature_names: List[str]
    label_name: str = "target"


def analyze_missing(df: pd.DataFrame,
                    mostly_null_pct: float = 60.0
                    ) -> Tuple[List[str], List[str], List[str]]:
    """
    將欄位依照缺失比例分成三類：
    - mostly_null: 缺失率 >= mostly_null_pct
    - partially_null: 0 < 缺失率 < mostly_null_pct
    - no_null: 沒有缺失值
    """
    mostly_null: List[str] = []
    partially_null: List[str] = []
    no_null: List[str] = []

    n_rows = len(df)

    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct = n_missing * 100.0 / n_rows

        if pct >= mostly_null_pct:
            mostly_null.append(col)
        elif n_missing > 0:
            partially_null.append(col)
        else:
            no_null.append(col)

    return mostly_null, partially_null, no_null


def find_high_corr_features(
    df: pd.DataFrame,
    threshold: float = 0.7,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    依照絕對相關係數門檻，找出「需要被丟掉」的高度相關欄位。
    exclude_cols: 例如 ['target']，避免 label 被當成特徵一起丟掉。
    """
    if exclude_cols is None:
        exclude_cols = []

    corr_matrix = df.corr()
    cols = [c for c in corr_matrix.columns if c not in exclude_cols]

    col_corr = set()

    for i in range(len(cols)):
        for j in range(i):
            c_i = cols[i]
            c_j = cols[j]
            if abs(corr_matrix.loc[c_i, c_j]) >= threshold:
                # 跟原 notebook 一樣，只保留其中一個，丟掉 "後面" 的那一個
                col_corr.add(c_i)

    return list(col_corr)


def preprocess_5sec_dataset(
    csv_path: str,
    corr_threshold: float = 0.7,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> PreprocessResult:
    """
    讀取 5-second window CSV，進行：
    1. 缺失值分析與處理
    2. 刪除 mostly_null 欄位 + id/user 欄
    3. target 文字轉數字
    4. 根據相關係數丟掉高度相關特徵
    5. MinMaxScaler 標準化
    6. train/test 切分

    回傳 PreprocessResult，提供後續訓練使用。
    """
    # 1. 讀檔
    df = pd.read_csv(csv_path)

    # 2. 缺失值分析
    mostly_null, partially_null, no_null = analyze_missing(df)

    # 3. 丟掉缺失太多的欄位
    if mostly_null:
        df = df.drop(columns=mostly_null, axis=1)

    # 4. 丟掉 id 欄（若存在）
    if "id" in df.columns:
        df = df.drop(columns=["id"], axis=1)

    # 5. 部分缺失欄位用 0 補
    for col in partially_null:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 6. target 映射到數字
    if "target" not in df.columns:
        raise ValueError("Dataset does not contain 'target' column.")

    df["target"] = df["target"].map(TARGET_MAP)

    # 7. 丟掉 user 欄位（若存在）
    if "user" in df.columns:
        df = df.drop(columns=["user"], axis=1)

    # 8. 確保 target 是最後一欄（方便之後切 X/y）
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    # 9. 依照相關係數丟掉高度相關特徵（排除 target）
    corr_features = find_high_corr_features(
        df=df,
        threshold=corr_threshold,
        exclude_cols=["target"],
    )

    if corr_features:
        df = df.drop(columns=corr_features, axis=1)

    # 再次確保 target 在最後一欄
    cols = [c for c in df.columns if c != "target"] + ["target"]
    df = df[cols]

    # 目前 df 的前 n-1 欄為特徵，最後一欄為 target
    feature_names = cols[:-1]
    label_name = cols[-1]

    # 10. 切 X / y
    X = df[feature_names].values.astype(np.float32)
    y = df[label_name].values.astype(np.int64)

    # 11. MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # 12. train/test split
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        feature_names=feature_names,
        label_name=label_name,
    )


if __name__ == "__main__":
    """
    簡單 CLI 測試：
    在終端機執行：
        python data_preprocess.py /path/to/dataset_5secondWindow.csv
    會印出前處理後的維度與欄位資訊。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess 5-second window transport mode dataset."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to dataset_5secondWindow.csv",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.7,
        help="Correlation threshold for dropping highly correlated features.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size ratio for train_test_split.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified split on target label.",
    )

    args = parser.parse_args()

    result = preprocess_5sec_dataset(
        csv_path=args.csv_path,
        corr_threshold=args.corr_threshold,
        test_size=args.test_size,
        stratify=not args.no_stratify,
    )

    print("=== Preprocess summary ===")
    print(f"X_train: {result.X_train.shape}")
    print(f"X_test : {result.X_test.shape}")
    print(f"y_train: {result.y_train.shape}")
    print(f"y_test : {result.y_test.shape}")
    print(f"#features: {len(result.feature_names)}")
    print("features:", result.feature_names)
    print("label   :", result.label_name)
