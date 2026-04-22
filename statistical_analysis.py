"""
統計検定モジュール
正規性の検定、Wilcoxon検定、多重比較、Cohen's dの計算を行います
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, wilcoxon, mannwhitneyu, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools


def test_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float, float]:
    """
    正規性の検定（Shapiro-Wilk test）を実行します
    
    Parameters:
    -----------
    data : np.ndarray
        検定するデータ
    alpha : float
        有意水準（デフォルト: 0.05）
    
    Returns:
    --------
    Tuple[bool, float, float]
        (正規性ありかどうか, 統計量, p値)
    """
    # サンプルサイズが3未満の場合は検定不可
    if len(data) < 3:
        return False, np.nan, np.nan
    
    # サンプルサイズが5000を超える場合は最初の5000個のみを使用
    if len(data) > 5000:
        data = data[:5000]
    
    statistic, p_value = shapiro(data)
    is_normal = p_value > alpha
    
    return is_normal, statistic, p_value


def create_qq_plot(data: np.ndarray, ax: Optional[plt.Axes] = None, 
                   title: str = 'Q-Q Plot') -> plt.Axes:
    """
    Q-Q plotを作成します
    
    Parameters:
    -----------
    data : np.ndarray
        データ
    ax : Optional[plt.Axes]
        プロットする軸（Noneの場合は新規作成）
    title : str
        プロットのタイトル
    
    Returns:
    --------
    plt.Axes
        プロットされた軸
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's dを計算します
    
    Parameters:
    -----------
    group1 : np.ndarray
        グループ1のデータ
    group2 : np.ndarray
        グループ2のデータ
    
    Returns:
    --------
    float
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # プールされた標準偏差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return cohens_d


def calculate_wilcoxon_r(group1: np.ndarray, group2: np.ndarray, 
                         statistic: float, p_value: float) -> float:
    """
    Wilcoxon検定の効果量rを計算します
    
    Parameters:
    -----------
    group1 : np.ndarray
        グループ1のデータ（対応データを想定）
    group2 : np.ndarray
        グループ2のデータ（対応データを想定）
    statistic : float
        Wilcoxon検定の統計量
    p_value : float
        Wilcoxon検定のp値
    
    Returns:
    --------
    float
        効果量r
    """
    n = len(group1)
    
    if n < 2:
        return 0.0
    
    # 対応データの場合
    if len(group1) == len(group2):
        # Wilcoxon signed-rank testの場合
        # 統計量Wを正規化してZ値を計算
        # 平均: μ = n(n+1)/4
        # 分散: σ² = n(n+1)(2n+1)/24
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        
        if var_w <= 0:
            return 0.0
        
        std_w = np.sqrt(var_w)
        z = (statistic - mean_w) / std_w
        
        # r = Z / sqrt(N)
        r = z / np.sqrt(n)
    else:
        # Mann-Whitney U検定の場合（対応のないデータ）
        # この場合はCohen's dを使用する方が適切だが、
        # 一貫性のためにrを計算する方法も提供
        n1, n2 = len(group1), len(group2)
        # U統計量からZ値を計算
        mean_u = n1 * n2 / 2.0
        var_u = n1 * n2 * (n1 + n2 + 1) / 12.0
        
        if var_u <= 0:
            return 0.0
        
        std_u = np.sqrt(var_u)
        z = (statistic - mean_u) / std_u
        
        # r = Z / sqrt(N)
        r = z / np.sqrt(n1 + n2)
    
    return r


def perform_wilcoxon_test(group1: np.ndarray, group2: np.ndarray, 
                         alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Wilcoxon検定を実行します（対応のあるデータ用）
    
    注意：この関数は対応データが既に準備されていることを前提としています。
    データの長さが異なる場合はMann-Whitney U検定を使用しますが、
    通常は対応データを準備してから呼び出すべきです。
    
    Parameters:
    -----------
    group1 : np.ndarray
        グループ1のデータ（対応データを想定）
    group2 : np.ndarray
        グループ2のデータ（対応データを想定）
    alternative : str
        検定の種類（'two-sided', 'less', 'greater'）
    
    Returns:
    --------
    Tuple[float, float]
        (統計量, p値)
    """
    if len(group1) != len(group2):
        # 対応のないデータの場合はMann-Whitney U検定を使用
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
    else:
        # 対応のあるデータの場合はWilcoxon検定を使用
        statistic, p_value = wilcoxon(group1, group2, alternative=alternative)
    
    return statistic, p_value


def multiple_comparison_holm(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Holm補正による多重比較を行います
    
    Parameters:
    -----------
    p_values : List[float]
        p値のリスト
    alpha : float
        有意水準（デフォルト: 0.05）
    
    Returns:
    --------
    List[bool]
        各比較が有意かどうかのリスト
    """
    rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='holm')
    return rejected.tolist()


def format_p_value(p_value: float) -> str:
    """
    p値を文字列にフォーマットします
    
    Parameters:
    -----------
    p_value : float
        p値
    
    Returns:
    --------
    str
        フォーマットされたp値
    """
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.01:
        return f"{p_value:.4f}"
    else:
        return f"{p_value:.4f}"


def analyze_group_comparisons(data: pd.DataFrame, value_col: str, 
                              method_col: str = 'Method',
                              subject_col: str = 'UID',
                              methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    グループ間の比較を行い、統計検定の結果を返します（対応のあるデータ用）
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム
    value_col : str
        値の列名
    method_col : str
        メソッド列名（デフォルト: 'Method'）
    subject_col : str
        被験者ID列名（デフォルト: 'UID'）
    methods : Optional[List[str]]
        比較するメソッドのリスト（Noneの場合は全て）
    
    Returns:
    --------
    pd.DataFrame
        検定結果のデータフレーム
    """
    if methods is None:
        methods = data[method_col].unique().tolist()
    
    results = []
    
    # 被験者ID列が存在する場合は対応データを準備
    has_subject_id = subject_col in data.columns
    
    # 全てのペアの組み合わせ
    for method1, method2 in itertools.combinations(methods, 2):
        if has_subject_id:
            # 対応データを準備：両方のメソッドにデータがある被験者のみを使用
            # ピボットテーブルを作成
            pivot_data = data.pivot_table(
                index=subject_col,
                columns=method_col,
                values=value_col,
                aggfunc='first'
            )
            
            # 両方のメソッドにデータがある被験者のみを選択
            # 表記揺れに対応：ピボットテーブルの列に存在するかチェック
            available_methods_in_pivot = pivot_data.columns.tolist()
            
            # method1とmethod2が実際にピボットテーブルに存在するか確認
            # 表記揺れに対応（例：cma_esとcma-es）
            method1_actual = None
            method2_actual = None
            
            for m in available_methods_in_pivot:
                if m == method1 or m.replace('-', '_') == method1 or m.replace('_', '-') == method1:
                    method1_actual = m
                if m == method2 or m.replace('-', '_') == method2 or m.replace('_', '-') == method2:
                    method2_actual = m
            
            if method1_actual is None or method2_actual is None:
                continue
            
            paired_data = pivot_data[[method1_actual, method2_actual]].dropna()
            
            if len(paired_data) == 0:
                continue
            
            group1 = paired_data[method1_actual].values
            group2 = paired_data[method2_actual].values
        else:
            # 被験者IDがない場合は従来の方法
            group1 = data[data[method_col] == method1][value_col].dropna().values
            group2 = data[data[method_col] == method2][value_col].dropna().values
        
        if len(group1) == 0 or len(group2) == 0:
            continue
        
        # 統計量を計算
        mean1, std1 = np.mean(group1), np.std(group1, ddof=1)
        mean2, std2 = np.mean(group2), np.std(group2, ddof=1)
        
        # Wilcoxon検定（対応ありのデータを想定）
        statistic, p_value = perform_wilcoxon_test(group1, group2)
        
        # Wilcoxon検定の効果量rを計算
        effect_r = calculate_wilcoxon_r(group1, group2, statistic, p_value)
        
        # 効果量の解釈（rの場合）
        if abs(effect_r) < 0.1:
            effect_size = "negligible"
        elif abs(effect_r) < 0.3:
            effect_size = "small"
        elif abs(effect_r) < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        # 方向性の判定
        if mean1 > mean2:
            direction = f"{method1} > {method2}"
        else:
            direction = f"{method2} > {method1}"
        
        results.append({
            'method1': method1,
            'method2': method2,
            'direction': direction,
            'mean1': mean1,
            'std1': std1,
            'mean2': mean2,
            'std2': std2,
            'statistic': statistic,
            'p_value': p_value,
            'effect_r': effect_r,
            'effect_size': effect_size
        })
    
    return pd.DataFrame(results)


def perform_friedman_test(data: pd.DataFrame, value_col: str, 
                          method_col: str = 'Method',
                          subject_col: str = 'UID',
                          methods: Optional[List[str]] = None) -> Tuple[float, float, float]:
    """
    Friedman検定を実行します（対応のあるデータ用）
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム（対応のあるデータを想定）
    value_col : str
        値の列名
    method_col : str
        メソッド列名
    subject_col : str
        被験者ID列名（デフォルト: 'UID'）
    methods : Optional[List[str]]
        比較するメソッドのリスト（Noneの場合は全て）
    
    Returns:
    --------
    Tuple[float, float, float]
        (統計量, p値, KendallのW)
    """
    if methods is None:
        methods = data[method_col].unique().tolist()
    
    # 被験者ID列が存在するか確認
    if subject_col not in data.columns:
        # 被験者IDがない場合は、従来の方法にフォールバック
        # （警告：これは正しいFriedman検定ではない）
        method_data_list = []
        for method in methods:
            method_data = data[data[method_col] == method][value_col].dropna().values
            if len(method_data) > 0:
                method_data_list.append(method_data)
        
        if len(method_data_list) < 3:
            return np.nan, 1.0, np.nan
        
        min_len = min(len(d) for d in method_data_list)
        method_data_list = [d[:min_len] for d in method_data_list]
        
        try:
            statistic, p_value = friedmanchisquare(*method_data_list)
            # KendallのWを計算: W = Q / (k * (n - 1))
            k = len(method_data_list)  # 条件数（メソッド数）
            n = min_len  # 被験者数
            kendall_w = statistic / (k * (n - 1)) if n > 1 else np.nan
            return statistic, p_value, kendall_w
        except ValueError:
            return np.nan, 1.0, np.nan
    
    # 被験者ごとに各メソッドのデータを取得
    # ピボットテーブルを作成：被験者を行、メソッドを列とする
    pivot_data = data.pivot_table(
        index=subject_col,
        columns=method_col,
        values=value_col,
        aggfunc='first'  # 重複がある場合は最初の値を使用
    )
    
    # 指定されたメソッドのみを選択（表記揺れに対応）
    available_methods = []
    available_methods_in_pivot = pivot_data.columns.tolist()
    
    for method in methods:
        # 完全一致を優先
        if method in available_methods_in_pivot:
            available_methods.append(method)
        else:
            # 表記揺れに対応（例：cma_esとcma-es）
            for m in available_methods_in_pivot:
                if m.replace('-', '_') == method or m.replace('_', '-') == method:
                    available_methods.append(m)
                    break
    
    if len(available_methods) < 3:
        return np.nan, 1.0, np.nan
    
    pivot_data = pivot_data[available_methods]
    
    # 全てのメソッドにデータがある被験者のみを保持（完全な対応データ）
    pivot_data = pivot_data.dropna()
    
    if len(pivot_data) < 3:  # 被験者数が3未満の場合は検定不可
        return np.nan, 1.0, np.nan
    
    # 各メソッドのデータをリストに変換（被験者の順序を保持）
    method_data_list = [pivot_data[method].values for method in available_methods]
    
    # Friedman検定を実行
    try:
        statistic, p_value = friedmanchisquare(*method_data_list)
        # KendallのWを計算: W = Q / (k * (n - 1))
        k = len(available_methods)  # 条件数（メソッド数）
        n = len(pivot_data)  # 被験者数
        kendall_w = statistic / (k * (n - 1)) if n > 1 else np.nan
        return statistic, p_value, kendall_w
    except ValueError:
        return np.nan, 1.0, np.nan


def apply_holm_correction(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Holm補正を適用します
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        検定結果のデータフレーム
    alpha : float
        有意水準（デフォルト: 0.05）
    
    Returns:
    --------
    pd.DataFrame
        Holm補正後の結果データフレーム
    """
    p_values = results_df['p_value'].values
    rejected = multiple_comparison_holm(p_values, alpha=alpha)
    
    results_df = results_df.copy()
    results_df['significant'] = rejected
    results_df['p_adjusted'] = multipletests(p_values, alpha=alpha, method='holm')[1]
    
    return results_df


def format_result_string(row: pd.Series, data_name: str = '') -> str:
    """
    検定結果を文字列にフォーマットします
    
    Parameters:
    -----------
    row : pd.Series
        検定結果の行
    data_name : str
        データ名（例: 'NASA-TLX total'）
    
    Returns:
    --------
    str
        フォーマットされた結果文字列
    """
    method1 = row['method1']
    method2 = row['method2']
    mean1 = row['mean1']
    std1 = row['std1']
    mean2 = row['mean2']
    std2 = row['std2']
    # 補正後のp値（p_adjusted）を優先的に使用
    p_value = row.get('p_adjusted', row.get('p_value', 1.0))
    # Wilcoxon検定の効果量rを使用（後方互換性のためcohens_dもチェック）
    effect_r = row.get('effect_r', row.get('cohens_d', 0.0))
    
    # 方向性に応じて順序を決定
    if mean1 > mean2:
        direction = f"{method1} > {method2}"
        mean_high = mean1
        std_high = std1
        mean_low = mean2
        std_low = std2
    else:
        direction = f"{method2} > {method1}"
        mean_high = mean2
        std_high = std2
        mean_low = mean1
        std_low = std1
    
    p_str = format_p_value(p_value)
    
    result = f"{data_name} {direction} {mean_high:.2f} ± {std_high:.2f} vs {mean_low:.2f} ± {std_low:.2f} {p_str} r = {effect_r:.2f}"
    
    return result


def save_statistical_results(results_df: pd.DataFrame, output_path: Path, 
                            data_name: str = ''):
    """
    統計検定の結果をログファイルに保存します
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        検定結果のデータフレーム
    output_path : Path
        出力ファイルのパス
    data_name : str
        データ名
    """
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"{data_name}\n")
        f.write(f"{'='*80}\n\n")
        
        # 各比較の有意性レベルを取得し、最大の*数を決定
        max_level = 0
        for _, row in results_df.iterrows():
            p_value = row.get('p_adjusted', row.get('p_value', 1.0))
            level, _ = get_significance_level(p_value)
            max_level = max(max_level, level)
        
        # 最大レベルに応じた*記号を決定
        if max_level >= 99:
            max_symbol = '***'
        elif max_level >= 95:
            max_symbol = '**'
        elif max_level >= 90:
            max_symbol = '*'
        else:
            max_symbol = ''
        
        for _, row in results_df.iterrows():
            result_str = format_result_string(row, data_name)
            # 有意性がある場合のみ*を追加
            p_value = row.get('p_adjusted', row.get('p_value', 1.0))
            level, symbol = get_significance_level(p_value)
            if level > 0 and symbol == max_symbol:
                result_str = f"{max_symbol} {result_str}"
            f.write(f"{result_str}\n")
        
        f.write("\n")


def save_normality_test_result(data_name: str, is_normal: bool, 
                               statistic: float, p_value: float, 
                               output_path: Path):
    """
    正規性検定の結果をログファイルに保存します
    
    Parameters:
    -----------
    data_name : str
        データ名
    is_normal : bool
        正規性ありかどうか
    statistic : float
        統計量
    p_value : float
        p値
    output_path : Path
        出力ファイルのパス
    """
    with open(output_path, 'a', encoding='utf-8') as f:
        # データ名をセクションとして出力
        f.write(f"\n{'='*80}\n")
        f.write(f"{data_name}\n")
        f.write(f"{'='*80}\n\n")
        
        # 正規性検定の結果を出力
        if np.isnan(statistic) or np.isnan(p_value):
            f.write(f"{data_name} サンプルサイズが不足しているため検定不可\n")
        else:
            normal_str = "正規分布に従う" if is_normal else "正規分布に従わない"
            p_str = format_p_value(p_value)
            f.write(f"{data_name} Shapiro-Wilk statistic={statistic:.4f}, p-value={p_str}, {normal_str}\n")
        
        f.write("\n")


def get_significance_level(p_value: float) -> Tuple[int, str]:
    """
    p値から有意水準と記号を返します
    
    Parameters:
    -----------
    p_value : float
        p値
    
    Returns:
    --------
    Tuple[int, str]
        (有意水準のパーセント, 記号)
    """
    if p_value < 0.01:  # 99%
        return 99, '***'
    elif p_value < 0.05:  # 95%
        return 95, '**'
    elif p_value < 0.10:  # 90%
        return 90, '*'
    else:
        return 0, 'ns'

