"""
テスト用プロットスクリプト
視覚的な調節に用います。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import re
import itertools
from pathlib import Path
from typing import Optional
try:
    from vistats import annotate_brackets
    HAS_VISTATS = True
except ImportError:
    HAS_VISTATS = False
    print("Warning: vistats not found. Using manual bracket drawing.")
import style_config
import ueq_processor
import sus_processor
import statistical_analysis

# スタイル設定を適用
style_config.setup_seaborn_style()

# データディレクトリ
DESIGN_DIR = Path('design')
MARIO_DIR = Path('mario')

# アウトプットディレクトリ
OUTPUT_DIR = Path('plot')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 設定取得用のgetter関数
def get_violin_config_for_grouped():
    """groupedプロット用のバイオリンプロット設定を取得します"""
    config = {
        'width': style_config.GROUPED_PLOT_CONFIG['violin_width'],
        'dodge': style_config.GROUPED_PLOT_CONFIG['violin_dodge'],
        'inner': style_config.GROUPED_PLOT_CONFIG['violin_inner'],
        'density_norm': style_config.GROUPED_PLOT_CONFIG['violin_density_norm'],
        'cut': style_config.GROUPED_PLOT_CONFIG['violin_cut'],
        'bw': style_config.GROUPED_PLOT_CONFIG['violin_bw'],
    }
    # gapパラメータが利用可能な場合（seaborn 0.12以降）
    if 'violin_gap' in style_config.GROUPED_PLOT_CONFIG:
        config['gap'] = style_config.GROUPED_PLOT_CONFIG['violin_gap']
    return config

def get_violin_config_for_single():
    """単一プロット用のバイオリンプロット設定を取得します"""
    config = {k: v for k, v in style_config.VIOLIN_PLOT_CONFIG.items() 
              if k not in ['inner', 'scale']}
    # scaleパラメータをdensity_normに変更（FutureWarning対応）
    if 'scale' in style_config.VIOLIN_PLOT_CONFIG:
        if style_config.VIOLIN_PLOT_CONFIG['scale'] == 'width':
            config['density_norm'] = 'width'
    return config

def get_bar_config_for_grouped():
    """groupedプロット用の棒グラフ設定を取得します"""
    return {
        'width': style_config.GROUPED_PLOT_CONFIG['bar_width'],
        'dodge': style_config.GROUPED_PLOT_CONFIG['bar_dodge'],
        'errorbar': style_config.GROUPED_PLOT_CONFIG['bar_errorbar'],
        'capsize': style_config.GROUPED_PLOT_CONFIG['bar_capsize'],
        'err_kws': {'linewidth': style_config.GROUPED_PLOT_CONFIG['bar_err_linewidth']},
    }

def get_bar_config_for_single():
    """単一プロット用の棒グラフ設定を取得します"""
    return {
        'width': style_config.BAR_PLOT_CONFIG['width'],
        'capsize': style_config.SINGLE_PLOT_CONFIG['bar_capsize'],
        'error_kw': {'elinewidth': style_config.SINGLE_PLOT_CONFIG['bar_err_linewidth']},
    }

def get_bracket_config():
    """ブラケット（統計的有意性表示）の設定を取得します"""
    return style_config.BRACKET_CONFIG.copy()

def load_data(filepath: Path) -> pd.DataFrame:
    """CSVファイルを読み込みます"""
    return pd.read_csv(filepath)

def load_combined_data(filename: str) -> pd.DataFrame:
    """designとmarioの両方のディレクトリからデータを読み込んで結合します"""
    design_file = DESIGN_DIR / filename
    mario_file = MARIO_DIR / filename
    
    # originalの場合はmarioのみを使用
    if filename == 'original.csv':
        if mario_file.exists():
            return load_data(mario_file)
        else:
            return pd.DataFrame()
    
    # その他のファイルは両方を結合
    dfs = []
    if design_file.exists():
        dfs.append(load_data(design_file))
    if mario_file.exists():
        dfs.append(load_data(mario_file))
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def get_data_columns(df: pd.DataFrame) -> list:
    """Timestampより右側の列（実験データ列）を取得します"""
    timestamp_idx = df.columns.get_loc('Timestamp')
    return df.columns[timestamp_idx + 1:].tolist()

def plot_violin_with_box(data: pd.DataFrame, value_col: str, 
                         method_col: str = 'Method', 
                         title: str = '', 
                         ylabel: str = '',
                         data_type: str = ''):
    """
    バイオリンプロット（箱ひげつき）を作成します
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム
    value_col : str
        プロットする値の列名
    method_col : str
        メソッド列名（デフォルト: 'Method'）
    title : str
        グラフのタイトル
    ylabel : str
        Y軸のラベル
    data_type : str
        データタイプ（'ueq', 'sus', 'original'）でy軸範囲を設定
    """
    fig, ax = plt.subplots(figsize=style_config.PLOT_CONFIG['figure.figsize'])
    
    # メソッドの順序を定義
    method_order = style_config.METHODS
    
    # データを準備
    plot_data = data[[method_col, value_col]].copy()
    plot_data = plot_data.dropna(subset=[value_col])
    
    # seabornのviolinplotを使用（箱ひげ図も内部に表示）
    violin_config = get_violin_config_for_single()
    
    violin = sns.violinplot(data=plot_data, x=method_col, y=value_col,
                           order=method_order,
                           palette=style_config.METHOD_COLORS,
                           inner=style_config.VIOLIN_PLOT_CONFIG['inner'],
                           ax=ax,
                           **violin_config)
    
    # violinの外枠だけを削除（boxplotは残す）
    for pc in violin.collections:
        pc.set_edgecolor('none')
    
    # 色盲対策：各メソッドにパターンを追加（目立たない程度に）
    _apply_hatch_patterns(ax, violin, 1, len(style_config.METHODS))
    
    # Y軸の範囲を設定
    if data_type and data_type in style_config.Y_AXIS_LIMITS:
        y_min_limit, y_max_limit = style_config.Y_AXIS_LIMITS[data_type]
        
        # tlx以外はデータの最小値・最大値に基づいて余白を追加
        # SUSの場合は余白を追加しない（5の上の余白をなくすため）
        if data_type != 'tlx' and data_type != 'sus':
            data_min = plot_data[value_col].min()
            data_max = plot_data[value_col].max()
            data_range = data_max - data_min if data_max > data_min else 1
            
            # 上下に5%の余白を追加（ただし設定された範囲内で）
            padding = data_range * 0.05 if data_range > 0 else 0.1
            
            # 余白を追加するが、設定された範囲を超えないようにする
            y_min = max(y_min_limit, data_min - padding)
            y_max = min(y_max_limit, data_max + padding)
            
            # データが範囲の端にある場合は、設定値を使用
            # ただし、最小値が設定値に近い場合は余白を追加しない（Y軸が2から始まるのを防ぐ）
            if abs(data_min - y_min_limit) < 0.1 or data_min <= y_min_limit + 0.1:
                y_min = y_min_limit
            if abs(data_max - y_max_limit) < 0.1:
                y_max = y_max_limit
        else:
            # tlxとsusは余白なしで設定値を使用
            y_min, y_max = y_min_limit, y_max_limit
        
        ax.set_ylim(y_min, y_max)
        
        # Y軸の刻みを設定
        if data_type in style_config.Y_AXIS_TICKS:
            tick_interval = style_config.Y_AXIS_TICKS[data_type]
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    
    # X軸の設定（ラベルとティックラベルを削除）
    ax.set_xlabel('')
    ax.set_xticklabels([''] * len(method_order))
    ax.set_ylabel(ylabel if ylabel else value_col)
    ax.set_title(title if title else f'{value_col} by Method')
    
    plt.tight_layout()
    return fig, ax

def plot_bar_with_error(data: pd.DataFrame, value_col: str,
                       method_col: str = 'Method',
                       title: str = '',
                       ylabel: str = '',
                       data_type: str = ''):
    """
    エラーバー付き棒グラフを作成します（TLX用）
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム
    value_col : str
        プロットする値の列名
    method_col : str
        メソッド列名（デフォルト: 'Method'）
    title : str
        グラフのタイトル
    ylabel : str
        Y軸のラベル
    data_type : str
        データタイプ（'tlx'）でy軸範囲を設定
    """
    fig, ax = plt.subplots(figsize=style_config.PLOT_CONFIG['figure.figsize'])
    
    # メソッドの順序を定義
    method_order = style_config.METHODS
    
    # 各メソッドの平均値と標準誤差を計算
    means = []
    errors = []
    colors = []
    
    for method in method_order:
        method_data = data[data[method_col] == method][value_col].dropna()
        if len(method_data) > 0:
            means.append(method_data.mean())
            errors.append(method_data.std() / np.sqrt(len(method_data)))  # 標準誤差
            colors.append(style_config.METHOD_COLORS[method])
        else:
            means.append(0)
            errors.append(0)
            colors.append(style_config.METHOD_COLORS[method])
    
    # 棒グラフを描画
    bar_config = get_bar_config_for_single()
    x_pos = np.arange(len(method_order))
    bars = ax.bar(x_pos, means, yerr=errors, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.5,
                  **bar_config)
    
    # 色盲対策：各メソッドにパターンを追加（目立たない程度に）
    for i, (bar, method) in enumerate(zip(bars, method_order)):
        hatch_pattern = style_config.METHOD_HATCH_PATTERNS.get(method, '')
        if hatch_pattern:
            bar.set_hatch(hatch_pattern)
            bar.set_linewidth(0.5)
    
    # Y軸の範囲を設定
    if data_type and data_type in style_config.Y_AXIS_LIMITS:
        y_min_limit, y_max_limit = style_config.Y_AXIS_LIMITS[data_type]
        
        # tlxは固定範囲を使用
        y_min, y_max = y_min_limit, y_max_limit
        
        ax.set_ylim(y_min, y_max)
        
        # Y軸の刻みを設定
        if data_type in style_config.Y_AXIS_TICKS:
            tick_interval = style_config.Y_AXIS_TICKS[data_type]
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    
    # X軸の設定（ラベルとティックラベルを削除）
    ax.set_xticks(x_pos)
    ax.set_xticklabels([''] * len(method_order))
    ax.set_xlabel('')
    ax.set_ylabel(ylabel if ylabel else value_col)
    ax.set_title(title if title else f'{value_col} by Method (Mean ± SE)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax

def shorten_question_name(col_name: str, index: int, data_type: str = '') -> str:
    """
    質問項目名を短縮します（アルファベット+数字のみ）
    
    Parameters:
    -----------
    col_name : str
        元の列名
    index : int
        インデックス（1から始まる）
    data_type : str
        データタイプ（'tlx', 'ueq', 'sus', 'original'）
    
    Returns:
    --------
    str
        短縮された質問名（例: Mental, Physical, Q1, Q2, ueq_1, sus1, PQ, HQ）
    """
    # TLXの場合は正式名（頭文字大文字）を使用
    if data_type == 'tlx' and col_name in style_config.TLX_QUESTION_NAMES:
        return style_config.TLX_QUESTION_NAMES[col_name]
    
    # UEQのPQとHQはそのまま表示
    if data_type == 'ueq' and col_name in ['PQ', 'HQ']:
        return col_name
    
    # SUSのSUS_Scoreはそのまま表示
    if data_type == 'sus' and col_name == 'SUS_Score':
        return 'SUS'
    
    # 既に短い形式（ueq_1, sus1など）の場合はそのまま
    if col_name.startswith('ueq_') or col_name.startswith('sus'):
        return col_name
    
    # 数字で始まる場合は、その数字を抽出
    match = re.match(r'^(\d+)\.', col_name)
    if match:
        num = match.group(1)
        return f'Q{num}'
    
    # それ以外はインデックスを使用
    return f'Q{index}'

def _apply_hatch_patterns(ax, plot_obj, n_questions, n_methods):
    """
    色盲対策として、各メソッドにパターン（ハッチング）を追加します
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロットする軸
    plot_obj : seaborn plot object
        seabornのプロットオブジェクト（violinplotまたはbarplot）
    n_questions : int
        質問数
    n_methods : int
        メソッド数
    """
    # violinplotの場合
    if hasattr(plot_obj, 'collections'):
        # collectionsから各violinのパッチを取得
        # 順序: (質問0, メソッド0), (質問0, メソッド1), ..., (質問0, メソッドN-1), (質問1, メソッド0), ...
        for question_idx in range(n_questions):
            for method_idx, method in enumerate(style_config.METHODS):
                collection_idx = question_idx * n_methods + method_idx
                if collection_idx < len(plot_obj.collections):
                    pc = plot_obj.collections[collection_idx]
                    hatch_pattern = style_config.METHOD_HATCH_PATTERNS.get(method, '')
                    if hatch_pattern:
                        pc.set_hatch(hatch_pattern)
                        # パターンを目立たない程度にするため、線の太さを細く
                        pc.set_linewidth(0.5)
    
    # barplotの場合
    elif hasattr(ax, 'patches'):
        patches = ax.patches
        # 順序: (質問0, メソッド0), (質問0, メソッド1), ..., (質問0, メソッドN-1), (質問1, メソッド0), ...
        for question_idx in range(n_questions):
            for method_idx, method in enumerate(style_config.METHODS):
                patch_idx = question_idx * n_methods + method_idx
                if patch_idx < len(patches):
                    patch = patches[patch_idx]
                    hatch_pattern = style_config.METHOD_HATCH_PATTERNS.get(method, '')
                    if hatch_pattern:
                        patch.set_hatch(hatch_pattern)
                        # パターンを目立たない程度にするため、線の太さを細く
                        patch.set_linewidth(0.5)


def _get_patch_positions(ax, question_name, n_methods):
    """
    seabornのプロットから、指定された質問名に対応する各メソッドのパッチ位置を取得します
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロットする軸
    question_name : str
        質問名
    n_methods : int
        メソッド数
    
    Returns:
    --------
    dict : {method_name: x_position} の辞書
    """
    # x軸のティックラベルを取得
    x_tick_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    if question_name not in x_tick_labels:
        return {}
    
    # 質問のインデックスを取得
    question_idx = x_tick_labels.index(question_name)
    
    # パッチを取得
    patches = ax.patches
    
    # パッチの位置を取得
    # seabornのプロットでは、パッチは以下の順序で並んでいる:
    # (質問0, メソッド0), (質問0, メソッド1), ..., (質問0, メソッドN-1), (質問1, メソッド0), ...
    method_positions = {}
    for method_idx, method in enumerate(style_config.METHODS):
        patch_idx = question_idx * n_methods + method_idx
        if patch_idx < len(patches):
            patch = patches[patch_idx]
            # パッチの中心x位置を取得
            method_positions[method] = patch.get_x() + patch.get_width() / 2
        else:
            # パッチが見つからない場合は、x軸の位置から推定
            # seabornのbarplot/violinplotのデフォルトのhueオフセットを計算
            hue_offset = 0.2  # デフォルトのhueオフセット
            method_positions[method] = question_idx + (method_idx - (n_methods - 1) / 2) * hue_offset
    
    return method_positions


def _add_significance_bars(ax, question_name, methods_data, y_max, method_col='Method', data_type=''):
    """
    ダイナマイトプロット（統計的有意性を示す*マーク）を追加します
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロットする軸
    question_name : str
        質問名
    methods_data : pd.DataFrame
        メソッド別のデータ（Question, Value, Method列を含む）
    y_max : float
        Y軸の最大値
    method_col : str
        メソッド列名
    data_type : str
        データタイプ
    """
    # この質問のデータのみをフィルタ（重要：異なる質問間での比較を防ぐ）
    question_data = methods_data[methods_data['Question'] == question_name].copy()
    
    if question_data.empty:
        return
    
    # パッチの位置を取得
    n_methods = len(style_config.METHODS)
    method_positions = _get_patch_positions(ax, question_name, n_methods)
    
    if len(method_positions) < 2:
        return
    
    # 各メソッドの最大値を取得（ブラケットの高さ用）
    method_max_values = {}
    for method in style_config.METHODS:
        if method not in method_positions:
            continue
        method_data = question_data[question_data[method_col] == method]['Value']
        if len(method_data) > 0:
            method_max_values[method] = method_data.max()
    
    if len(method_max_values) < 2:
        return
    
    # Friedman検定を実行（有意な場合のみWilcoxon検定を実行）
    # この質問のデータを準備
    friedman_data = question_data[[method_col, 'Value']].copy()
    friedman_stat, friedman_p = statistical_analysis.perform_friedman_test(
        friedman_data, 'Value', method_col, style_config.METHODS
    )
    
    # Friedman検定で有意な場合のみWilcoxon検定を実行（Holm補正）
    comparisons = []
    if not np.isnan(friedman_stat) and friedman_p < 0.05:
        # 全てのペアでWilcoxon検定を実行（対応ありのwilcoxon）
        raw_p_values = []
        comparison_info = []
        
        for method1, method2 in itertools.combinations(style_config.METHODS, 2):
            if method1 not in method_max_values or method2 not in method_max_values:
                continue
            
            # この質問のデータのみを使用（重要：異なる質問間での比較を防ぐ）
            group1 = question_data[question_data[method_col] == method1]['Value'].dropna().values
            group2 = question_data[question_data[method_col] == method2]['Value'].dropna().values
            
            if len(group1) == 0 or len(group2) == 0:
                continue
            
            # Wilcoxon signed-rank test（対応ありのwilcoxon）
            _, p_value = statistical_analysis.perform_wilcoxon_test(group1, group2)
            raw_p_values.append(p_value)
            comparison_info.append((method1, method2, p_value))
        
        # Holm補正を適用
        if len(raw_p_values) > 0:
            from statsmodels.stats.multitest import multipletests
            rejected, p_adjusted, _, _ = multipletests(raw_p_values, alpha=0.05, method='holm')
            
            # 補正後のp値で有意な比較を抽出
            for i, (method1, method2, _) in enumerate(comparison_info):
                if rejected[i]:
                    adjusted_p = p_adjusted[i]
                    level, symbol = statistical_analysis.get_significance_level(adjusted_p)
                    if level > 0:
                        comparisons.append((level, symbol, method1, method2))
    
    if len(comparisons) == 0:
        return
    
    # 有意な比較をレベル順にソート（高いレベルから）
    comparisons.sort(key=lambda x: x[0], reverse=True)
    
    # ブラケット設定を取得
    bracket_config = get_bracket_config()
    
    # vistatsを使う場合の準備
    # この質問内のメソッドのみを含める（異なる質問間での比較を防ぐ）
    available_methods = [method for method in style_config.METHODS if method in method_positions]
    if len(available_methods) < 2:
        return
    
    # center, height, yerrを準備（この質問内のメソッドのみ）
    center = [method_positions[method] for method in available_methods]
    height = [method_max_values[method] for method in available_methods]
    yerr = [0] * len(center)  # エラーバーは使用しない
    
    # tuplesを準備（available_methods内のインデックスを使用）
    method_to_idx = {method: idx for idx, method in enumerate(available_methods)}
    tuples = []
    for level, symbol, method1, method2 in comparisons:
        if method1 in method_to_idx and method2 in method_to_idx:
            idx1 = method_to_idx[method1]
            idx2 = method_to_idx[method2]
            tuples.append((idx1, idx2, symbol))
    
    if len(tuples) == 0:
        return
    
    # Y軸の上限を拡張（ただし設定値（Y_AXIS_LIMITS）を超えないようにする）
    current_ylim = ax.get_ylim()
    y_range = current_ylim[1] - current_ylim[0]
    
    # 設定された最大値を取得
    y_max_limit = None
    if data_type and data_type in style_config.Y_AXIS_LIMITS:
        _, y_max_limit = style_config.Y_AXIS_LIMITS[data_type]
    
    # ブラケット用の推定空間を計算
    estimated_space = bracket_config['estimated_space_ratio'] * y_range * len(tuples)
    
    # 設定された最大値を超えないようにする
    new_y_max = current_ylim[1] + estimated_space
    if y_max_limit is not None:
        new_y_max = min(new_y_max, y_max_limit)
    
    ax.set_ylim(current_ylim[0], new_y_max)
    
    # vistats.annotate_bracketsを使用（デフォルト設定）
    if HAS_VISTATS:
        try:
            # 参考URLの使い方に従う（デフォルト設定を使用）
            annotate_brackets(
                tuples, center, height, yerr,
                ax=ax,
                fs=bracket_config['fontsize']
            )
        except Exception as e:
            # vistatsが使えない場合は、手動でブラケットを描画
            print(f"Warning: annotate_brackets failed: {e}")
            _draw_manual_brackets(ax, comparisons, method_positions, method_max_values, bracket_config)
    else:
        # vistatsが使えない場合は、手動でブラケットを描画
        _draw_manual_brackets(ax, comparisons, method_positions, method_max_values, bracket_config)


def _draw_manual_brackets(ax, comparisons, method_positions, method_max_values, bracket_config):
    """
    手動でブラケットを描画します（vistatsが使えない場合のフォールバック）
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロットする軸
    comparisons : list
        比較結果のリスト [(level, symbol, method1, method2), ...]
    method_positions : dict
        メソッド名とx位置のマッピング
    method_max_values : dict
        メソッド名と最大値のマッピング
    bracket_config : dict
        ブラケット設定
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    max_y_pos = max(method_max_values.values())
    y_offset_base = max_y_pos + bracket_config['y_offset_base_ratio'] * y_range
    y_spacing = bracket_config['y_spacing_ratio'] * y_range
    bracket_height = bracket_config['bracket_height_ratio'] * y_range
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    min_width = bracket_config['min_bracket_width'] * x_range
    
    for i, (level, symbol, method1, method2) in enumerate(comparisons):
        if method1 not in method_positions or method2 not in method_positions:
            continue
        
        x1 = method_positions[method1]
        x2 = method_positions[method2]
        y_pos = y_offset_base + i * y_spacing
        
        # ブラケットの最小幅を確保（横線が表示されるように）
        if abs(x2 - x1) < min_width:
            center_x = (x1 + x2) / 2
            x1 = center_x - min_width / 2
            x2 = center_x + min_width / 2
        
        # ブラケットを描画（上向き）
        ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bracket_height, y_pos + bracket_height, y_pos],
               color=bracket_config['color'], linewidth=bracket_config['linewidth'], clip_on=False)
        ax.text((x1 + x2) / 2, y_pos + bracket_height, symbol, 
               ha='center', va='bottom', 
               fontsize=bracket_config['fontsize'], 
               fontweight=bracket_config['fontweight'], 
               clip_on=False)


def _plot_single_grouped_bar(ax, plot_data, question_cols_subset, method_col, data_type, ylabel, start_idx=1):
    """
    単一のsubplotにプロットを描画する内部関数
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロットする軸
    plot_data : pd.DataFrame
        プロットデータ（Question, Value, Method列を含む）
    question_cols_subset : list
        このsubplotで表示する質問列名のリスト
    method_col : str
        メソッド列名
    data_type : str
        データタイプ
    ylabel : str
        Y軸のラベル
    start_idx : int
        質問名のインデックスの開始値（デフォルト: 1）
    """
    # このsubplot用のデータをフィルタ
    subplot_data = plot_data[plot_data['Question'].isin(question_cols_subset)].copy()
    
    if subplot_data.empty:
        return
    
    # 全てのデータタイプでバイオリンプロットを使用
    violin_config = get_violin_config_for_grouped()
    violin = sns.violinplot(data=subplot_data, x='Question', y='Value', hue=method_col,
                           hue_order=style_config.METHODS,
                           palette=style_config.METHOD_COLORS,
                           ax=ax,
                           **violin_config)
    
    # violinの外枠だけを削除（boxplotは残す）
    if style_config.GROUPED_PLOT_CONFIG.get('violin_remove_outline', False):
        for pc in violin.collections:
            pc.set_edgecolor('none')
    
    # 色盲対策：各メソッドにパターンを追加（目立たない程度に）
    _apply_hatch_patterns(ax, violin, len(question_cols_subset), len(style_config.METHODS))
    
    # Y軸の範囲を設定
    y_max = None
    if data_type and data_type in style_config.Y_AXIS_LIMITS:
        y_min_limit, y_max_limit = style_config.Y_AXIS_LIMITS[data_type]
        
        # tlx以外はデータの最小値・最大値に基づいて余白を追加
        # SUSの場合は余白を追加しない（5の上の余白をなくすため）
        if data_type != 'tlx' and data_type != 'sus':
            data_min = subplot_data['Value'].min()
            data_max = subplot_data['Value'].max()
            data_range = data_max - data_min if data_max > data_min else 1
            
            # 上下に5%の余白を追加（ただし設定された範囲内で）
            padding = data_range * 0.05 if data_range > 0 else 0.1
            
            # 余白を追加するが、設定された範囲を超えないようにする
            y_min = max(y_min_limit, data_min - padding)
            y_max = min(y_max_limit, data_max + padding)
            
            # データが範囲の端にある場合は、設定値を使用（ただし最小値は設定値より小さくしない）
            if abs(data_min - y_min_limit) < 0.1:
                y_min = y_min_limit
            elif data_min < y_min_limit:
                # データが最小値より小さい場合は、最小値を使用（余白は追加しない）
                y_min = y_min_limit
            if abs(data_max - y_max_limit) < 0.1:
                y_max = y_max_limit
        else:
            # tlxとsusは余白なしで設定値を使用
            y_min, y_max = y_min_limit, y_max_limit
        
        # 設定された範囲を超えないようにする（重要：ブラケット用の余白を追加しても範囲を超えない）
        ax.set_ylim(y_min, y_max)
        
        # Y軸の刻みを設定
        if data_type in style_config.Y_AXIS_TICKS:
            tick_interval = style_config.Y_AXIS_TICKS[data_type]
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    else:
        y_max = subplot_data['Value'].max()
    
    # ダイナマイトプロット（統計的有意性）を追加
    for question_name in question_cols_subset:
        _add_significance_bars(ax, question_name, subplot_data, y_max, method_col, data_type)
    
    # X軸の設定
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    # x軸の目盛りを表示
    ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True)
    ax.grid(True, alpha=style_config.GROUPED_PLOT_CONFIG['grid_alpha'], 
            axis=style_config.GROUPED_PLOT_CONFIG['grid_axis'])
    
    # 左右の縦線（spine）を削除
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 凡例を削除（別ファイルで作成するため）
    if ax.legend_:
        ax.legend_.remove()


def _perform_statistical_tests(data: pd.DataFrame, question_cols: list,
                               method_col: str, title: str, data_type: str,
                               log_file: Optional[Path] = None):
    """
    統計検定を実行してログに保存します（プロットとは分離）
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム
    question_cols : list
        質問の列名のリスト
    method_col : str
        メソッド列名
    title : str
        データタイトル
    data_type : str
        データタイプ
    log_file : Optional[Path]
        ログファイルのパス
    """
    if log_file is None:
        return
    
    qq_output_dir = OUTPUT_DIR / 'qq_plots'
    qq_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Q-Q plotフォルダ内のログファイル
    qq_log_file = qq_output_dir / 'normality_test_results.log'
    # ログファイルを初期化（最初のセクションのみ）
    if not qq_log_file.exists():
        with open(qq_log_file, 'w', encoding='utf-8') as f:
            f.write("正規性検定結果（Shapiro-Wilk test）\n")
            f.write("="*80 + "\n\n")
    
    for idx, col in enumerate(question_cols, start=1):
        question_name = shorten_question_name(col, idx, data_type)
        question_data = data[[method_col, col]].dropna(subset=[col])
        
        if len(question_data) == 0:
            continue
        
        # Q-Q plotを各メソッドごとに作成し、正規性検定を実行
        for method in style_config.METHODS:
            method_data = question_data[question_data[method_col] == method][col].dropna().values
            if len(method_data) > 0:
                # Q-Q plotを作成
                fig, ax = plt.subplots(figsize=(6, 6))
                statistical_analysis.create_qq_plot(
                    method_data, 
                    ax=ax, 
                    title=f'{title} {question_name} - {method}'
                )
                safe_name = f"{title}_{question_name}_{method}".replace(' ', '_').replace('/', '_')
                plt.savefig(qq_output_dir / f'{safe_name}.pdf', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 正規性検定を実行
                is_normal, statistic, p_value = statistical_analysis.test_normality(method_data)
                data_name = f"{title} {question_name} - {method}"
                statistical_analysis.save_normality_test_result(
                    data_name, is_normal, statistic, p_value, qq_log_file
                )
        
        # Friedman検定を実行（全ての結果をログに出力）
        friedman_stat, friedman_p = statistical_analysis.perform_friedman_test(
            question_data, col, method_col, style_config.METHODS
        )
        
        data_name = f"{title} {question_name}"
        
        # Friedman検定の結果を常にログに追加
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{data_name} (Friedman test)\n")
            f.write(f"{'='*80}\n\n")
            if not np.isnan(friedman_stat):
                f.write(f"Friedman test: statistic={friedman_stat:.4f}, p-value={statistical_analysis.format_p_value(friedman_p)}\n")
            else:
                f.write(f"Friedman test: 検定不可（データ不足）\n")
            f.write("\n")
        
        # Friedman検定で有意な場合のみWilcoxon検定を実行（Holm補正）
        if not np.isnan(friedman_stat) and friedman_p < 0.05:
            results_df = statistical_analysis.analyze_group_comparisons(
                question_data, col, method_col, style_config.METHODS
            )
            if not results_df.empty:
                # Holm補正を適用
                results_df = statistical_analysis.apply_holm_correction(results_df)
                statistical_analysis.save_statistical_results(results_df, log_file, data_name)


def plot_grouped_bar(data: pd.DataFrame, question_cols: list,
                     method_col: str = 'Method',
                     title: str = '',
                     ylabel: str = 'Score',
                     data_type: str = '',
                     log_file: Optional[Path] = None):
    """
    複数の質問を1つのプロットにまとめ、質問ごとにhueで表示します
    
    Parameters:
    -----------
    data : pd.DataFrame
        データフレーム
    question_cols : list
        プロットする質問の列名のリスト
    method_col : str
        メソッド列名（デフォルト: 'Method'）
    title : str
        グラフのタイトル
    ylabel : str
        Y軸のラベル
    data_type : str
        データタイプ（'tlx', 'ueq', 'sus', 'original'）
    """
    # データタイプ別のfigure sizeを取得（デフォルトはGROUPED_PLOT_CONFIGから）
    if data_type and data_type in style_config.GROUPED_PLOT_FIGSIZE:
        figsize = style_config.GROUPED_PLOT_FIGSIZE[data_type]
    else:
        figsize = style_config.GROUPED_PLOT_CONFIG['figure.figsize']
    
    # originalの場合は2段のsubplotを作成
    if data_type == 'original' and len(question_cols) > 1:
        # 半分で割って、割り切れない場合は1段目が1つ多い
        n_cols = len(question_cols)
        n_first_row = (n_cols + 1) // 2  # 切り上げ
        n_second_row = n_cols - n_first_row
        
        # 2段のsubplotを作成（縦に2つ）
        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 2))
        
        # 1段目と2段目の質問列を分割
        first_row_cols = question_cols[:n_first_row]
        second_row_cols = question_cols[n_first_row:]
        
        # データをロングフォーマットに変換
        plot_data_list = []
        for idx, col in enumerate(question_cols, start=1):
            if col in data.columns:
                col_data = data[[method_col, col]].copy()
                # 質問名を短縮
                short_name = shorten_question_name(col, idx, data_type)
                col_data['Question'] = short_name
                col_data = col_data.rename(columns={col: 'Value'})
                plot_data_list.append(col_data)
        
        if not plot_data_list:
            return fig, axes
        
        plot_data = pd.concat(plot_data_list, ignore_index=True)
        plot_data = plot_data.dropna(subset=['Value'])
        
        # 統計検定を実行（プロットとは分離）
        _perform_statistical_tests(data, question_cols, method_col, title, data_type, log_file)
        
        # 1段目と2段目の質問名リストを作成
        first_row_question_names = [shorten_question_name(col, idx, data_type) 
                                   for idx, col in enumerate(first_row_cols, start=1)]
        second_row_question_names = [shorten_question_name(col, idx + n_first_row, data_type) 
                                    for idx, col in enumerate(second_row_cols, start=1)]
        
        # 1段目をプロット
        _plot_single_grouped_bar(axes[0], plot_data, first_row_question_names, 
                                method_col, data_type, ylabel)
        axes[0].set_title(title)
        
        # 2段目をプロット
        _plot_single_grouped_bar(axes[1], plot_data, second_row_question_names, 
                                method_col, data_type, ylabel)
        
        plt.tight_layout()
        return fig, axes
    else:
        # 通常の1つのプロット
        fig, ax = plt.subplots(figsize=figsize)
        
        # データをロングフォーマットに変換
        plot_data_list = []
        for idx, col in enumerate(question_cols, start=1):
            if col in data.columns:
                col_data = data[[method_col, col]].copy()
                # 質問名を短縮
                short_name = shorten_question_name(col, idx, data_type)
                col_data['Question'] = short_name
                col_data = col_data.rename(columns={col: 'Value'})
                plot_data_list.append(col_data)
        
        if not plot_data_list:
            return fig, ax
        
        plot_data = pd.concat(plot_data_list, ignore_index=True)
        plot_data = plot_data.dropna(subset=['Value'])
        
        # 統計検定を実行（プロットとは分離）
        _perform_statistical_tests(data, question_cols, method_col, title, data_type, log_file)
        
        # プロットを描画
        question_names = [shorten_question_name(col, idx, data_type) for idx, col in enumerate(question_cols, start=1)]
        _plot_single_grouped_bar(ax, plot_data, question_names, method_col, data_type, ylabel)
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax

def create_legend():
    """凡例を別ファイルとして作成します"""
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # 凡例用のパッチを作成
    from matplotlib.patches import Patch
    legend_elements = []
    for method in style_config.METHODS:
        patch = Patch(facecolor=style_config.METHOD_COLORS[method], 
                     edgecolor='black', linewidth=1.5, 
                     label=style_config.METHOD_DISPLAY_NAMES.get(method, method))
        # 色盲対策：凡例にもパターンを追加
        hatch_pattern = style_config.METHOD_HATCH_PATTERNS.get(method, '')
        if hatch_pattern:
            patch.set_hatch(hatch_pattern)
            patch.set_linewidth(0.5)
        legend_elements.append(patch)
    
    # 凡例を表示
    ax.legend(handles=legend_elements, loc='center', 
             frameon=True, fancybox=True, shadow=False,
             ncol=len(style_config.METHODS))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'legend.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"保存: {OUTPUT_DIR}/legend.pdf")
    plt.close()

def test_plots():
    """テスト用のプロットを生成します"""
    
    # 統計検定結果のログファイル
    log_file = OUTPUT_DIR / 'statistical_results.log'
    # ログファイルを初期化
    if log_file.exists():
        log_file.unlink()
    
    # 凡例を先に作成
    create_legend()
    
    # NASA-TLXデータでテスト（エラーバー付き棒グラフ）
    print("NASA-TLXデータでテスト中...")
    tlx_df = load_combined_data('nasa-tlx.csv')
    if not tlx_df.empty:
        print(f"データ形状: {tlx_df.shape}")
        print(f"メソッド: {tlx_df['Method'].unique()}")
        
        # overallスコアでテスト
        if 'overall' in tlx_df.columns:
            fig, ax = plot_bar_with_error(tlx_df, 'overall', 
                                         title='NASA-TLX Overall Score',
                                         ylabel='Overall Score',
                                         data_type='tlx')
            plt.savefig(OUTPUT_DIR / 'test_tlx_overall.pdf', dpi=300, bbox_inches='tight')
            print(f"保存: {OUTPUT_DIR}/test_tlx_overall.pdf")
            plt.close()
        
        # 各サブスケールを1つのプロットにまとめる（質問ごとのhueで表示）
        tlx_columns = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'overall']
        available_cols = [col for col in tlx_columns if col in tlx_df.columns]
        if available_cols:
            fig, ax = plot_grouped_bar(tlx_df, available_cols,
                                      title='NASA-TLX',
                                      ylabel='Score',
                                      data_type='tlx',
                                      log_file=log_file)
            plt.savefig(OUTPUT_DIR / 'test_tlx_grouped.pdf', dpi=300, bbox_inches='tight')
            print(f"保存: {OUTPUT_DIR}/test_tlx_grouped.pdf")
            plt.close()
    
    # SUSデータでテスト（groupedプロット、SUSスコア1つ）
    print("\nSUSデータでテスト中...")
    # marioのSUSデータのみを使用
    mario_sus_file = MARIO_DIR / 'sus.csv'
    if mario_sus_file.exists():
        sus_df_raw = load_data(mario_sus_file)
        # SUSスコアに変換
        sus_df = sus_processor.calculate_sus_score(sus_df_raw)
        if not sus_df.empty:
            print(f"データ形状: {sus_df.shape}")
            print(f"メソッド: {sus_df['Method'].unique()}")
            
            # SUS_Scoreの1列のみを使用
            sus_columns = ['SUS_Score']
            if 'SUS_Score' in sus_df.columns:
                fig, ax = plot_grouped_bar(sus_df, sus_columns,
                                         title='SUS',
                                         ylabel='Score',
                                         data_type='sus',
                                         log_file=log_file)
                plt.savefig(OUTPUT_DIR / 'test_sus_grouped.pdf', dpi=300, bbox_inches='tight')
                print(f"保存: {OUTPUT_DIR}/test_sus_grouped.pdf")
                plt.close()
    
    # UEQ-Sデータでテスト（groupedプロット、PQとHQの2尺度）
    print("\nUEQ-Sデータでテスト中...")
    # marioのUEQデータのみを使用
    mario_ueq_file = MARIO_DIR / 'ueq-short.csv'
    if mario_ueq_file.exists():
        ueq_df_raw = load_data(mario_ueq_file)
        # PQとHQに変換
        ueq_df = ueq_processor.calculate_ueq_scales(ueq_df_raw)
        if not ueq_df.empty:
            print(f"データ形状: {ueq_df.shape}")
            print(f"メソッド: {ueq_df['Method'].unique()}")
            
            # PQとHQの2列のみを使用
            ueq_columns = ['PQ', 'HQ']
            if all(col in ueq_df.columns for col in ueq_columns):
                fig, ax = plot_grouped_bar(ueq_df, ueq_columns,
                                         title='UEQ-S',
                                         ylabel='Score',
                                         data_type='ueq',
                                         log_file=log_file)
                plt.savefig(OUTPUT_DIR / 'test_ueq_grouped.pdf', dpi=300, bbox_inches='tight')
                print(f"保存: {OUTPUT_DIR}/test_ueq_grouped.pdf")
                plt.close()
    
    # オリジナルデータでテスト（groupedプロット）
    print("\nオリジナルデータでテスト中...")
    original_df = load_combined_data('original.csv')
    if not original_df.empty:
        print(f"データ形状: {original_df.shape}")
        print(f"メソッド: {original_df['Method'].unique()}")
        
        original_columns = get_data_columns(original_df)
        if original_columns:
            fig, ax = plot_grouped_bar(original_df, original_columns,
                                     title='Original',
                                     ylabel='Score',
                                     data_type='original',
                                     log_file=log_file)
            plt.savefig(OUTPUT_DIR / 'test_original_grouped.pdf', dpi=300, bbox_inches='tight')
            print(f"保存: {OUTPUT_DIR}/test_original_grouped.pdf")
            plt.close()

if __name__ == '__main__':
    test_plots()
    print("\nテスト完了！")

