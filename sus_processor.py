"""
SUSデータ処理モジュール
SUSの10項目データを1つの総合スコア（0-100）に変換します
"""

import pandas as pd
from typing import List


def calculate_sus_score(df: pd.DataFrame, method_col: str = 'Method') -> pd.DataFrame:
    """
    SUSの10項目データを1つの総合スコア（0-100）に変換します
    
    Parameters:
    -----------
    df : pd.DataFrame
        SUSデータ（sus1からsus10の列を含む）
    method_col : str
        メソッド列名（デフォルト: 'Method'）
    
    Returns:
    --------
    pd.DataFrame
        SUS_Score列を含むデータフレーム（元のメタデータ列も保持）
    """
    # Method列の値を統一（cma-es -> cma_es）
    if method_col in df.columns:
        df = df.copy()
        df[method_col] = df[method_col].str.replace('cma-es', 'cma_es', regex=False)
    
    # SUS項目の列名を取得
    sus_cols = [col for col in df.columns if col.startswith('sus')]
    
    if len(sus_cols) != 10:
        raise ValueError(f"SUSは10項目である必要があります。現在の項目数: {len(sus_cols)}")
    
    # 項目をソート（sus1, sus2, ..., sus10の順）
    sus_cols_sorted = sorted(sus_cols, key=lambda x: int(x[3:]) if x[3:].isdigit() else float('inf'))
    
    # 結果データフレームを作成（メタデータ列を保持）
    meta_cols = [col for col in df.columns if col not in sus_cols]
    result_df = df[meta_cols].copy()
    
    # SUSスコアを計算
    # 奇数項目（1, 3, 5, 7, 9）: スコア - 1
    # 偶数項目（2, 4, 6, 8, 10）: 5 - スコア
    # すべての調整されたスコアを合計し、2.5を掛けて0-100のスコアに変換
    
    # 奇数項目のインデックス（0-indexedなので0, 2, 4, 6, 8）
    odd_indices = [0, 2, 4, 6, 8]
    # 偶数項目のインデックス（1-indexedなので1, 3, 5, 7, 9）
    even_indices = [1, 3, 5, 7, 9]
    
    # 奇数項目のスコア調整（スコア - 1）
    odd_scores = df[[sus_cols_sorted[i] for i in odd_indices]].sub(1).sum(axis=1)
    
    # 偶数項目のスコア調整（5 - スコア）
    even_scores = df[[sus_cols_sorted[i] for i in even_indices]].rsub(5).sum(axis=1)
    
    # 総合スコアを計算（合計に2.5を掛けて0-100のスコアに変換）
    result_df['SUS_Score'] = (odd_scores + even_scores) * 2.5
    
    return result_df


def load_sus_data(filepath: str) -> pd.DataFrame:
    """
    SUSデータを読み込み、SUSスコアに変換します
    
    Parameters:
    -----------
    filepath : str
        SUSデータファイルのパス
    
    Returns:
    --------
    pd.DataFrame
        SUS_Score列を含むデータフレーム
    """
    df = pd.read_csv(filepath)
    return calculate_sus_score(df)

