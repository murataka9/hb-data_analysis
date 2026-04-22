"""
共通スタイル定義ファイル
このファイルは読み取り専用として、他の分析スクリプトから参照されます。
"""

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from typing import Dict, List

# メソッド名の定義（順序も定義）
METHODS: List[str] = ['glv_bo_hybrid', 'bo', 'cma_es', 'manual']

# ユニバーサルデザインに配慮した色定義（青・緑系を中心に）
# 色覚多様性に配慮し、ColorBrewerの色覚安全なパレットを参考にしています
# customized_violin_analysis.pyで使用されている明るい色を採用
METHOD_COLORS: Dict[str, str] = {
    'glv_bo_hybrid': '#87CEEB',  # 水色（明るい、customized_violin_analysis.pyから）
    'bo': '#FFA500',              # オレンジ（明るい、customized_violin_analysis.pyから）
    'cma_es': '#98FB98',          # 薄い緑（明るい、customized_violin_analysis.pyから）
    'manual': '#A43AD1'           # 青系（明るい青）
}

# 色のリスト（順序付き）
METHOD_COLOR_LIST: List[str] = [METHOD_COLORS[method] for method in METHODS]

# メソッド名の正式名マッピング
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    'glv_bo_hybrid': 'HB',
    'bo': 'BO',
    'cma_es': 'CMA-ES',
    'manual': 'Manual'
}

# 色盲対策用のパターン（ハッチング）マッピング
# 目立たない程度の細かいパターンを使用
METHOD_HATCH_PATTERNS: Dict[str, str] = {
    'glv_bo_hybrid': '',      # パターンなし（ベース）
    'bo': '...',              # 点々
    'cma_es': '///',          # 斜線（右斜め）
    'manual': 'xxx'           # クロス
}

# Helvetica Neue Lightフォントの設定
HELVETICA_NEUE_LIGHT = FontProperties(family='sans-serif', weight='light')
# 利用可能なフォントを確認してHelvetica Neue Lightを優先
try:
    from matplotlib.font_manager import findfont, FontProperties
    # Helvetica Neue Lightを探す
    helvetica_light = FontProperties(family='Helvetica Neue', weight='light')
    test_font = findfont(helvetica_light)
    if 'Helvetica' in test_font or 'helvetica' in test_font.lower():
        HELVETICA_NEUE_LIGHT = FontProperties(family='Helvetica Neue', weight='light')
except:
    # フォールバック: 通常のHelvetica Neue
    HELVETICA_NEUE_LIGHT = FontProperties(family='Helvetica Neue')

# seabornスタイル設定
def setup_seaborn_style():
    """seabornのスタイルを設定します"""
    sns.set_style("whitegrid")
    sns.set_palette(METHOD_COLOR_LIST)
    # フォント設定（Helvetica Neue Light）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    # フォントサイズの設定
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['figure.titlesize'] = 18
    # 細めのフォントウェイトを設定（可能な場合）
    try:
        plt.rcParams['font.weight'] = 'light'
    except:
        pass

# プロットの共通設定
PLOT_CONFIG = {
    'figure.dpi': 100,
    'figure.figsize': (4, 6),  # プロットをさらに細くする
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none'
}

# Y軸の範囲設定（データタイプ別）
Y_AXIS_LIMITS = {
    'tlx': (0, 100),      # NASA-TLX: 0-100
    'ueq': (-3, 3),       # UEQ-S: -3~3（1-7から変換）
    'sus': (0, 100),      # SUS: 0-100
    'original': (1, 5),   # Original: 1-5
    'ai_original': (1, 5) # AI Original: 1-5
}

# バイオリンプロットの設定
VIOLIN_PLOT_CONFIG = {
    'inner': 'box',  # 箱ひげ図を内部に表示
    'width': 0.7,   # プロットの太さを2/3に調整（0.7 * 2/3 ≈ 0.47）
    'scale': 'width',
    'cut': 0,
    'bw': 0.2,  # バンド幅を小さくしてより敏感に（デフォルトは'scott'、小さい値ほど敏感）
    # linewidthは削除（violinの外枠だけを消すため、プロット後に処理）
}

# バイオリンプロットの色とスタイル設定（customized_violin_analysis.pyの設定を反映）
VIOLIN_COLOR_CONFIG = {
    'alpha': 1.0,  # バイオリンの透明度（1.0で鮮やかに表示）
    'edgecolor': 'none',  # 外枠の色（noneで削除）
    'median_marker_color': 'white',  # 中央値マーカーの色（白）
    'median_marker_size': 30,  # 中央値マーカーのサイズ
    'box_color': 'k',  # 四分位数の箱の色（黒）
    'box_linewidth': 2,  # 四分位数の箱の線の太さ
    'whisker_color': 'k',  # ヒゲの色（黒）
    'whisker_linewidth': 1,  # ヒゲの線の太さ
    'hatch_linewidth': 0.5,  # ハッチングパターンの線の太さ
    'hatch_color': 'white',  # ハッチングパターンの色（白、main.py側で使用）
}

# エラーバー付き棒グラフの設定
BAR_PLOT_CONFIG = {
    'width': 0.7,   # プロットをさらに細くする
    'capsize': 5,
    'error_kw': {'elinewidth': 2, 'capthick': 2}
}

# Y軸の刻み設定（データタイプ別）
Y_AXIS_TICKS = {
    'tlx': 20,         # NASA-TLX: 20刻み
    'ueq': 1,          # UEQ-S: 1刻み（-3~3の範囲）
    'sus': 20,         # SUS: 20刻み（0-100の範囲）
    'original': 1,     # Original: 1刻み
    'ai_original': 1   # AI Original: 1刻み
}

# プロット間の間隔設定（狭くする）
PLOT_SPACING = {
    'bar_spacing': 0.5,  # 棒グラフの間隔（デフォルト1.0より狭く）
    'violin_positions_scale': 0.7  # バイオリンプロットの位置スケール（小さいほど狭い）
}

# Groupedプロットの設定
GROUPED_PLOT_CONFIG = {
    'figure.figsize': (28, 4),  # groupedプロットの図のサイズ（デフォルト、データタイプ別に上書き可能）
    'bar_width': 0.9,  # groupedプロットの棒グラフの幅
    'bar_capsize': 0.5,  # groupedプロットのエラーバーのキャップサイズ
    'bar_err_linewidth': 2,  # groupedプロットのエラーバーの線の太さ
    'bar_dodge': True,  # groupedプロットの棒グラフのdodge設定
    'bar_errorbar': 'se',  # groupedプロットのエラーバーの種類
    'violin_width': 0.85,  # groupedプロットのバイオリンプロットの幅
    'violin_dodge': True,  # groupedプロットのバイオリンプロットのdodge設定
    'violin_gap': 0.1,  # groupedプロットのバイオリンプロットのhueグループ間の間隔（0.0-1.0、大きいほど間隔が広い）
    'violin_inner': 'box',  # groupedプロットのバイオリンプロットのinner設定
    'violin_density_norm': 'width',  # groupedプロットのバイオリンプロットのdensity_norm設定
    'violin_cut': 0.6,  # groupedプロットのバイオリンプロットのcut設定
    'violin_bw': 0.4,  # groupedプロットのバイオリンプロットのバンド幅（敏感度）
    'violin_remove_outline': True,  # groupedプロットのバイオリンプロットの外枠を削除（boxplotは残す）
    'grid_alpha': 0.3,  # groupedプロットのグリッドの透明度
    'grid_axis': 'y',  # groupedプロットのグリッドの軸
}

# Groupedプロットのfigure size設定（データタイプ別）
GROUPED_PLOT_FIGSIZE = {
    'tlx': (15, 4),         # TLXのfigure size
    'ueq': (5, 4),          # UEQのfigure size
    'sus': (3, 4),          # SUSのfigure size
    'original': (22, 4),    # Originalのfigure size
    'ai_original': (15, 4)  # AI Originalのfigure size
}

# customized_violin_analysis.py用のfigure size設定
CUSTOMIZED_VIOLIN_FIGSIZE = (8, 5)  # 1行2列のプロット用（幅×高さ、幅を2/3に調整）

# average_data_count_statistical_analysis.py用のfigure size設定
AVERAGE_DATA_COUNT_FIGSIZE = (4, 5)  # バープロット用（幅×高さ）

# 単一プロットの設定（plot_violin_with_box, plot_bar_with_error用）
SINGLE_PLOT_CONFIG = {
    'bar_capsize': 0.5,  # 単一プロットのエラーバーのキャップサイズ
    'bar_err_linewidth': 2,  # 単一プロットのエラーバーの線の太さ
}

# TLXの項目名の正式名マッピング（正式名称と略称）
TLX_QUESTION_NAMES = {
    'mental': 'Mental D.',
    'physical': 'Physical D.',
    'temporal': 'Temporal D.',
    'performance': 'Performance',
    'effort': 'Effort',
    'frustration': 'Frustration',
    'overall': 'Overall'
}

# ブラケット（統計的有意性表示）の設定
# vistats.annotate_bracketsのデフォルト設定を参考にした値
BRACKET_CONFIG = {
    'y_offset_base_ratio': 0.05,  # プロットの上からの開始位置（Y軸範囲に対する比率）
    'y_spacing_ratio': 0.05,      # 各ブラケット間の間隔（Y軸範囲に対する比率、デフォルト値に近づける）
    'bracket_height_ratio': 0.02, # ブラケットの高さ（Y軸範囲に対する比率）
    'estimated_space_ratio': 0.1, # ブラケット用の推定空間（比較数×この比率）
    'min_bracket_width': 0.01,    # ブラケットの最小幅（X軸範囲に対する比率、デフォルト値に近づける）
    'fontsize': 10,               # ブラケットのテキストフォントサイズ（デフォルト値）
    'fontweight': 'normal',        # ブラケットのテキストフォントウェイト（デフォルト値）
    'linewidth': 1.0,             # ブラケットの線の太さ（デフォルト値）
    'color': 'black'              # ブラケットの色
}

