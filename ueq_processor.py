"""
UEQデータ処理モジュール
UEQ-Sの8項目データをPragmatic Quality (PQ)とHedonic Quality (HQ)の2尺度に変換します
"""

import pandas as pd
from typing import List


def calculate_ueq_scales(df: pd.DataFrame, method_col: str = 'Method') -> pd.DataFrame:
    """
    UEQ-Sの8項目データをPragmatic Quality (PQ)とHedonic Quality (HQ)の2尺度に変換します
    
    Parameters:
    -----------
    df : pd.DataFrame
        UEQ-Sデータ（ueq_1からueq_8の列を含む）
    method_col : str
        メソッド列名（デフォルト: 'Method'）
    
    Returns:
    --------
    pd.DataFrame
        PQとHQの2列を含むデータフレーム（元のメタデータ列も保持）
    """
    # Method列の値を統一（cma-es -> cma_es）
    if method_col in df.columns:
        df = df.copy()
        df[method_col] = df[method_col].str.replace('cma-es', 'cma_es', regex=False)
    
    # UEQ項目の列名を取得
    ueq_cols = [col for col in df.columns if col.startswith('ueq_')]
    
    if len(ueq_cols) != 8:
        raise ValueError(f"UEQ-Sは8項目である必要があります。現在の項目数: {len(ueq_cols)}")
    
    # 項目をソート（ueq_1, ueq_2, ..., ueq_8の順）
    ueq_cols_sorted = sorted(ueq_cols, key=lambda x: int(x.split('_')[1]))
    
    # Pragmatic Quality (PQ): 項目1-4（ueq_1, ueq_2, ueq_3, ueq_4）
    pq_cols = ueq_cols_sorted[:4]
    
    # Hedonic Quality (HQ): 項目5-8（ueq_5, ueq_6, ueq_7, ueq_8）
    hq_cols = ueq_cols_sorted[4:]
    
    # 結果データフレームを作成（メタデータ列を保持）
    meta_cols = [col for col in df.columns if col not in ueq_cols]
    result_df = df[meta_cols].copy()
    
    # PQとHQのスコアを計算（各尺度の平均値）
    result_df['PQ'] = df[pq_cols].mean(axis=1)
    result_df['HQ'] = df[hq_cols].mean(axis=1)
    
    return result_df


def load_ueq_data(filepath: str) -> pd.DataFrame:
    """
    UEQデータを読み込み、PQとHQに変換します
    
    Parameters:
    -----------
    filepath : str
        UEQデータファイルのパス
    
    Returns:
    --------
    pd.DataFrame
        PQとHQの2列を含むデータフレーム
    """
    df = pd.read_csv(filepath)
    return calculate_ueq_scales(df)

