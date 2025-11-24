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
from pathlib import Path
import style_config
import ueq_processor
import sus_processor

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
    return {
        'width': style_config.GROUPED_PLOT_CONFIG['violin_width'],
        'dodge': style_config.GROUPED_PLOT_CONFIG['violin_dodge'],
        'inner': style_config.GROUPED_PLOT_CONFIG['violin_inner'],
        'density_norm': style_config.GROUPED_PLOT_CONFIG['violin_density_norm'],
        'cut': style_config.GROUPED_PLOT_CONFIG['violin_cut'],
        'bw': style_config.GROUPED_PLOT_CONFIG['violin_bw'],
    }

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
            if abs(data_min - y_min_limit) < 0.1:
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
    
    # データタイプに応じてプロット方法を選択
    if data_type == 'tlx':
        # TLXはエラーバー付き棒グラフ
        bar_config = get_bar_config_for_grouped()
        sns.barplot(data=subplot_data, x='Question', y='Value', hue=method_col,
                    hue_order=style_config.METHODS,
                    palette=style_config.METHOD_COLORS,
                    ax=ax,
                    **bar_config)
    else:
        # UEQ、SUS、オリジナルはバイオリンプロット
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
    
    # Y軸の範囲を設定
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
            
            # データが範囲の端にある場合は、設定値を使用
            if abs(data_min - y_min_limit) < 0.1:
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
    
    # X軸の設定
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    # x軸の目盛りを表示
    ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True)
    ax.grid(True, alpha=style_config.GROUPED_PLOT_CONFIG['grid_alpha'], 
            axis=style_config.GROUPED_PLOT_CONFIG['grid_axis'])
    
    # 凡例を削除（別ファイルで作成するため）
    if ax.legend_:
        ax.legend_.remove()


def plot_grouped_bar(data: pd.DataFrame, question_cols: list,
                     method_col: str = 'Method',
                     title: str = '',
                     ylabel: str = 'Score',
                     data_type: str = ''):
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
        
        # プロットを描画
        _plot_single_grouped_bar(ax, plot_data, 
                                 [shorten_question_name(col, idx, data_type) 
                                  for idx, col in enumerate(question_cols, start=1)],
                                 method_col, data_type, ylabel)
        ax.set_title(title)
        
        plt.tight_layout()
        return fig, ax

def create_legend():
    """凡例を別ファイルとして作成します"""
    fig, ax = plt.subplots(figsize=(4, 2))
    
    # 凡例用のパッチを作成
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=style_config.METHOD_COLORS[method], 
              edgecolor='black', linewidth=1.5, label=method)
        for method in style_config.METHODS
    ]
    
    # 凡例を表示
    ax.legend(handles=legend_elements, loc='center', 
             frameon=True, fancybox=True, shadow=False,
             ncol=len(style_config.METHODS))
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'legend.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"保存: {OUTPUT_DIR}/legend.png")
    plt.close()

def test_plots():
    """テスト用のプロットを生成します"""
    
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
            plt.savefig(OUTPUT_DIR / 'test_tlx_overall.png', dpi=300, bbox_inches='tight')
            print(f"保存: {OUTPUT_DIR}/test_tlx_overall.png")
            plt.close()
        
        # 各サブスケールを1つのプロットにまとめる（質問ごとのhueで表示）
        tlx_columns = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration', 'overall']
        available_cols = [col for col in tlx_columns if col in tlx_df.columns]
        if available_cols:
            fig, ax = plot_grouped_bar(tlx_df, available_cols,
                                      title='NASA-TLX',
                                      ylabel='Score',
                                      data_type='tlx')
            plt.savefig(OUTPUT_DIR / 'test_tlx_grouped.png', dpi=300, bbox_inches='tight')
            print(f"保存: {OUTPUT_DIR}/test_tlx_grouped.png")
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
                                         data_type='sus')
                plt.savefig(OUTPUT_DIR / 'test_sus_grouped.png', dpi=300, bbox_inches='tight')
                print(f"保存: {OUTPUT_DIR}/test_sus_grouped.png")
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
                                         data_type='ueq')
                plt.savefig(OUTPUT_DIR / 'test_ueq_grouped.png', dpi=300, bbox_inches='tight')
                print(f"保存: {OUTPUT_DIR}/test_ueq_grouped.png")
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
                                     data_type='original')
            plt.savefig(OUTPUT_DIR / 'test_original_grouped.png', dpi=300, bbox_inches='tight')
            print(f"保存: {OUTPUT_DIR}/test_original_grouped.png")
            plt.close()

if __name__ == '__main__':
    test_plots()
    print("\nテスト完了！")

