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


def perform_wilcoxon_test(group1: np.ndarray, group2: np.ndarray, 
                         alternative: str = 'two-sided') -> Tuple[float, float]:
    """
    Wilcoxon検定を実行します（対応のあるデータの場合）
    
    Parameters:
    -----------
    group1 : np.ndarray
        グループ1のデータ
    group2 : np.ndarray
        グループ2のデータ
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
                              methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    グループ間の比較を行い、統計検定の結果を返します
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム
    value_col : str
        値の列名
    method_col : str
        メソッド列名（デフォルト: 'Method'）
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
    
    # 全てのペアの組み合わせ
    for method1, method2 in itertools.combinations(methods, 2):
        group1 = data[data[method_col] == method1][value_col].dropna().values
        group2 = data[data[method_col] == method2][value_col].dropna().values
        
        if len(group1) == 0 or len(group2) == 0:
            continue
        
        # 統計量を計算
        mean1, std1 = np.mean(group1), np.std(group1, ddof=1)
        mean2, std2 = np.mean(group2), np.std(group2, ddof=1)
        
        # Wilcoxon検定
        statistic, p_value = perform_wilcoxon_test(group1, group2)
        
        # Cohen's d
        cohens_d = calculate_cohens_d(group1, group2)
        
        # 効果量の解釈
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
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
            'cohens_d': cohens_d,
            'effect_size': effect_size
        })
    
    return pd.DataFrame(results)


def perform_friedman_test(data: pd.DataFrame, value_col: str, 
                          method_col: str = 'Method',
                          methods: Optional[List[str]] = None) -> Tuple[float, float]:
    """
    Friedman検定を実行します
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム（対応のあるデータを想定）
    value_col : str
        値の列名
    method_col : str
        メソッド列名
    methods : Optional[List[str]]
        比較するメソッドのリスト（Noneの場合は全て）
    
    Returns:
    --------
    Tuple[float, float]
        (統計量, p値)
    """
    if methods is None:
        methods = data[method_col].unique().tolist()
    
    # 各メソッドのデータを取得
    method_data_list = []
    for method in methods:
        method_data = data[data[method_col] == method][value_col].dropna().values
        if len(method_data) > 0:
            method_data_list.append(method_data)
    
    if len(method_data_list) < 3:
        return np.nan, 1.0
    
    # データの長さを揃える（最短の長さに合わせる）
    min_len = min(len(d) for d in method_data_list)
    method_data_list = [d[:min_len] for d in method_data_list]
    
    # Friedman検定を実行
    try:
        statistic, p_value = friedmanchisquare(*method_data_list)
    except ValueError:
        # データが不足している場合
        return np.nan, 1.0
    
    return statistic, p_value


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
    p_value = row['p_value']
    cohens_d = row['cohens_d']
    
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
    
    result = f"{data_name} {direction} {mean_high:.2f} ± {std_high:.2f} vs {mean_low:.2f} ± {std_low:.2f} {p_str} 𝑑𝑧 = {cohens_d:.2f}"
    
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

