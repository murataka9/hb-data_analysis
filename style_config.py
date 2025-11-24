"""
共通スタイル定義ファイル
このファイルは読み取り専用として、他の分析スクリプトから参照されます。
"""

import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List

# メソッド名の定義（順序も定義）
METHODS: List[str] = ['glv_bo_hybrid', 'bo', 'cma_es', 'manual']

# ユニバーサルデザインに配慮した色定義（青・緑系を中心に）
# 色覚多様性に配慮し、ColorBrewerの色覚安全なパレットを参考にしています
METHOD_COLORS: Dict[str, str] = {
    'cma_es': '#2E86AB',      # 青系（濃い青）
    'bo': '#06A77D',          # 緑系（エメラルドグリーン）
    'manual': '#118AB2',      # 青系（明るい青）
    'glv_bo_hybrid': '#4ECDC4' # 青緑系（ターコイズ）
}

# 色のリスト（順序付き）
METHOD_COLOR_LIST: List[str] = [METHOD_COLORS[method] for method in METHODS]

# seabornスタイル設定
def setup_seaborn_style():
    """seabornのスタイルを設定します"""
    sns.set_style("whitegrid")
    sns.set_palette(METHOD_COLOR_LIST)
    # フォントサイズの設定
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['figure.titlesize'] = 18

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
    'tlx': (0, 80),      # NASA-TLX: 0-100
    'ueq': (1, 7),        # UEQ-S: 1-7
    'sus': (0, 100),      # SUS: 0-100
    'original': (1, 5)    # Original: 1-5
}

# バイオリンプロットの設定
VIOLIN_PLOT_CONFIG = {
    'inner': 'box',  # 箱ひげ図を内部に表示
    'width': 0.7,    # プロットをさらに細くする
    'scale': 'width',
    'cut': 0,
    'bw': 0.2,  # バンド幅を小さくしてより敏感に（デフォルトは'scott'、小さい値ほど敏感）
    # linewidthは削除（violinの外枠だけを消すため、プロット後に処理）
}

# エラーバー付き棒グラフの設定
BAR_PLOT_CONFIG = {
    'width': 0.7,   # プロットをさらに細くする
    'capsize': 5,
    'error_kw': {'elinewidth': 2, 'capthick': 2}
}

# Y軸の刻み設定（データタイプ別）
Y_AXIS_TICKS = {
    'tlx': 20,      # NASA-TLX: 20刻み
    'ueq': 1,       # UEQ-S: 1刻み
    'sus': 20,      # SUS: 20刻み（0-100の範囲）
    'original': 1   # Original: 1刻み
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
    'violin_width': 0.8,  # groupedプロットのバイオリンプロットの幅
    'violin_dodge': True,  # groupedプロットのバイオリンプロットのdodge設定
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
    'tlx': (12, 4),      # TLXのfigure size
    'ueq': (6, 4),      # UEQのfigure size
    'sus': (3, 4),      # SUSのfigure size
    'original': (20, 4)  # Originalのfigure size
}

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

