#!/usr/bin/env python3
"""
Best HBユーザーとその他のユーザーの比較をviolin plotで可視化
style_config.pyのスタイルを使用、英語表記、ダイナマイトプロット対応
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
import sys

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import style_config
from statistical_analysis import calculate_wilcoxon_r

warnings.filterwarnings('ignore')

# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / 'exploration_mode_statistics' / 'alpha_ux_analysis_results'

# vistatsのインポート確認
try:
    from vistats import annotate_brackets
    HAS_VISTATS = True
except ImportError:
    HAS_VISTATS = False
    print("Warning: vistats not available, using manual bracket drawing")

def load_data():
    """データを読み込み"""
    merged_file = OUTPUT_DIR / 'merged_data.csv'
    if not merged_file.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {merged_file}")
    
    df = pd.read_csv(merged_file)
    return df

def calculate_effect_size(group1, group2, statistic):
    """効果量rを計算（Mann-Whitney U検定用）"""
    n1, n2 = len(group1), len(group2)
    mean_u = n1 * n2 / 2.0
    var_u = n1 * n2 * (n1 + n2 + 1) / 12.0
    
    if var_u > 0:
        std_u = np.sqrt(var_u)
        z = (statistic - mean_u) / std_u
        effect_r = z / np.sqrt(n1 + n2)
    else:
        effect_r = 0.0
    
    return effect_r

def interpret_effect_size(effect_r):
    """効果量の解釈"""
    abs_r = abs(effect_r)
    if abs_r < 0.1:
        return "Small"
    elif abs_r < 0.3:
        return "Medium"
    elif abs_r < 0.5:
        return "Large"
    else:
        return "Very Large"

def perform_statistical_tests(df):
    """統計検定を実行して結果を返す"""
    best_hb = df[df['group'] == 'Best HB'].copy()
    other = df[df['group'] == 'Other'].copy()
    
    # 分析する指標
    metrics = {
        'max_playability': 'Max Playability',
        'combined_playability': 'Combined Playability',
        'total_data_count': 'Total Data Count',
        'standard_pct': 'Standard Usage (%)',
        'bold_pct': 'Bold Usage (%)',
        'cautious_pct': 'Cautious Usage (%)',
        'total_changes': 'Alpha Changes Count'
    }
    
    results = {}
    
    for metric_key, metric_label in metrics.items():
        best_values = best_hb[metric_key].dropna()
        other_values = other[metric_key].dropna()
        
        if len(best_values) < 2 or len(other_values) < 2:
            continue
        
        # Mann-Whitney U検定
        statistic, p_value = mannwhitneyu(best_values, other_values, alternative='two-sided')
        
        # 効果量を計算
        effect_r = calculate_effect_size(best_values, other_values, statistic)
        effect_size = interpret_effect_size(effect_r)
        
        # 有意性レベルの決定（* <0.1, ** <0.05, *** <0.01）
        if p_value < 0.01:
            significance = '***'
            level = 99
        elif p_value < 0.05:
            significance = '**'
            level = 95
        elif p_value < 0.1:
            significance = '*'
            level = 90
        else:
            significance = 'ns'
            level = 0
        
        results[metric_key] = {
            'label': metric_label,
            'best_hb_values': best_values,
            'other_values': other_values,
            'best_hb_mean': best_values.mean(),
            'best_hb_median': best_values.median(),
            'best_hb_std': best_values.std(),
            'other_mean': other_values.mean(),
            'other_median': other_values.median(),
            'other_std': other_values.std(),
            'n_best': len(best_values),
            'n_other': len(other_values),
            'statistic': statistic,
            'p_value': p_value,
            'effect_r': effect_r,
            'effect_size': effect_size,
            'significance': significance,
            'level': level
        }
    
    return results

def create_violin_plots(df, results, output_dir):
    """Violin plotsを作成"""
    # スタイル設定
    style_config.setup_seaborn_style()
    
    # フォントサイズを4+3=7ポイント大きく設定（全体）
    plt.rcParams['font.size'] = 15 + 7  # 22
    plt.rcParams['axes.labelsize'] = 16 + 7  # 23
    plt.rcParams['axes.titlesize'] = 16 + 7  # 23
    plt.rcParams['xtick.labelsize'] = 15 + 7  # 22
    plt.rcParams['ytick.labelsize'] = 15 + 7  # 22
    plt.rcParams['legend.fontsize'] = 15 + 7  # 22
    
    # 分析する指標の順序
    metric_order = [
        'max_playability',
        'combined_playability',
        'total_data_count',
        'standard_pct',
        'bold_pct',
        'cautious_pct',
        'total_changes'
    ]
    
    # フィルタリング：結果があるものだけ
    metric_order = [m for m in metric_order if m in results]
    
    n_metrics = len(metric_order)
    
    # 図を作成（横に並べる、縦を1上げる）
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 7))
    
    if n_metrics == 1:
        axes = [axes]
    
    # グループの色（青系2色）
    group_colors = {
        'Best HB': '#4A90E2',  # 青系1
        'Other': '#87CEEB'     # 青系2（薄めの青）
    }
    
    for idx, metric_key in enumerate(metric_order):
        ax = axes[idx]
        result = results[metric_key]
        
        # データを準備
        plot_data = pd.DataFrame({
            'Group': ['Best HB'] * result['n_best'] + ['Other'] * result['n_other'],
            'Value': list(result['best_hb_values']) + list(result['other_values'])
        })
        
        # Violin plotを作成（起伏に富んだ形状にするため、bwを調整）
        violin = sns.violinplot(
            data=plot_data,
            x='Group',
            y='Value',
            order=['Best HB', 'Other'],
            palette=[group_colors['Best HB'], group_colors['Other']],
            inner='box',
            scale='width',
            cut=0.6,  # カットを0.6に設定
            bw=0.5,   # バンド幅を大きくして起伏に富んだ形状に
            ax=ax,
            width=0.7
        )
        
        # violinの外枠を削除（style_configの設定に従う）
        color_config = style_config.VIOLIN_COLOR_CONFIG
        for i, pc in enumerate(violin.collections):
            pc.set_edgecolor(color_config['edgecolor'])
            pc.set_alpha(color_config['alpha'])
            
            # 片方（Other側、インデックス1）に白のハッチングパターンを追加
            if i == 1:  # Other側
                pc.set_hatch('///')  # 斜線パターン
                pc.set_edgecolor('white')
                pc.set_linewidth(color_config['hatch_linewidth'])
        
        # タイトルを追加（グラフの上、黒色、フォントサイズを7ポイント大きく）
        base_title_size = style_config.PLOT_CONFIG.get('axes.titlesize', 16)
        ax.set_title(result['label'], fontsize=base_title_size + 7,
                    fontweight='bold', pad=10, color='black')
        
        # X軸の設定（メモリを表示、フォントサイズを7ポイント大きく）
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True, labelsize=22)  # 7ポイント大きく
        ax.tick_params(axis='y', which='major', labelsize=22)  # Y軸の目盛りも7ポイント大きく
        
        # X軸のラベルを黒色に設定
        for label in ax.get_xticklabels():
            label.set_color('black')
        
        # グリッド設定
        ax.grid(True, alpha=style_config.GROUPED_PLOT_CONFIG['grid_alpha'],
                axis=style_config.GROUPED_PLOT_CONFIG['grid_axis'])
        
        # 左右の縦線を削除
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 統計的有意性をブラケットで表示
        if result['significance'] != 'ns':
            # グループの最大値を取得
            best_max = result['best_hb_values'].max()
            other_max = result['other_values'].max()
            y_max = max(best_max, other_max)
            
            # Y軸の上限を拡張
            current_ylim = ax.get_ylim()
            y_range = current_ylim[1] - current_ylim[0]
            bracket_config = style_config.BRACKET_CONFIG
            estimated_space = bracket_config['estimated_space_ratio'] * y_range
            new_y_max = current_ylim[1] + estimated_space
            ax.set_ylim(current_ylim[0], new_y_max)
            
            # ブラケットを描画
            center = [0, 1]  # Best HB, Other の位置
            height = [best_max, other_max]
            yerr = [0, 0]
            
            tuples = [(0, 1, result['significance'])]
            
            # ブラケットのフォントサイズを7ポイント大きく
            bracket_fontsize = bracket_config['fontsize'] + 7
            
            if HAS_VISTATS:
                try:
                    annotate_brackets(
                        tuples, center, height, yerr,
                        ax=ax,
                        fs=bracket_fontsize
                    )
                except Exception as e:
                    print(f"Warning: annotate_brackets failed: {e}")
                    _draw_manual_brackets(ax, tuples, center, height, bracket_config, bracket_fontsize)
            else:
                _draw_manual_brackets(ax, tuples, center, height, bracket_config, bracket_fontsize)
    
    plt.tight_layout()
    
    output_file = output_dir / 'best_hb_comparison_violin.pdf'
    plt.savefig(output_file, dpi=style_config.PLOT_CONFIG['savefig.dpi'],
                bbox_inches=style_config.PLOT_CONFIG['savefig.bbox'],
                facecolor=style_config.PLOT_CONFIG['savefig.facecolor'],
                edgecolor=style_config.PLOT_CONFIG['savefig.edgecolor'])
    print(f"Violin plotを保存: {output_file}")
    plt.close()

def _draw_manual_brackets(ax, tuples, center, height, bracket_config, fontsize=None):
    """手動でブラケットを描画（vistatsが使えない場合）"""
    if fontsize is None:
        fontsize = bracket_config['fontsize']
    
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_offset = bracket_config['y_offset_base_ratio'] * y_range
    
    for idx1, idx2, symbol in tuples:
        x1, x2 = center[idx1], center[idx2]
        y1, y2 = height[idx1], height[idx2]
        
        y_max = max(y1, y2)
        bracket_y = y_max + y_offset
        
        # 水平線
        ax.plot([x1, x2], [bracket_y, bracket_y], 
                color=bracket_config['color'], linewidth=bracket_config['linewidth'])
        
        # 縦線
        bracket_height = bracket_config['bracket_height_ratio'] * y_range
        ax.plot([x1, x1], [bracket_y - bracket_height, bracket_y],
                color=bracket_config['color'], linewidth=bracket_config['linewidth'])
        ax.plot([x2, x2], [bracket_y - bracket_height, bracket_y],
                color=bracket_config['color'], linewidth=bracket_config['linewidth'])
        
        # テキスト（フォントサイズを大きく）
        text_x = (x1 + x2) / 2
        ax.text(text_x, bracket_y, symbol,
                ha='center', va='bottom',
                fontsize=fontsize,
                fontweight=bracket_config['fontweight'])

def generate_statistical_report(results, output_dir):
    """統計検定の詳細レポートを生成"""
    output_dir = Path(output_dir)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Statistical Test Results: Best HB vs Other Users")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("Statistical Tests Performed:")
    report_lines.append("- Mann-Whitney U Test (two-sided)")
    report_lines.append("- Effect Size (r): r = Z / sqrt(N)")
    report_lines.append("")
    
    report_lines.append("Significance Levels:")
    report_lines.append("- *** p < 0.01")
    report_lines.append("- **  p < 0.05")
    report_lines.append("- *   p < 0.1")
    report_lines.append("- ns  p ≥ 0.1")
    report_lines.append("")
    
    report_lines.append("Effect Size Interpretation (r):")
    report_lines.append("- |r| < 0.1: Small")
    report_lines.append("- 0.1 ≤ |r| < 0.3: Medium")
    report_lines.append("- 0.3 ≤ |r| < 0.5: Large")
    report_lines.append("- |r| ≥ 0.5: Very Large")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("Detailed Results")
    report_lines.append("="*80)
    report_lines.append("")
    
    for metric_key, result in results.items():
        report_lines.append(f"Metric: {result['label']}")
        report_lines.append("-"*80)
        report_lines.append(f"  Best HB:")
        report_lines.append(f"    n = {result['n_best']}")
        report_lines.append(f"    Mean = {result['best_hb_mean']:.3f}")
        report_lines.append(f"    Median = {result['best_hb_median']:.3f}")
        report_lines.append(f"    SD = {result['best_hb_std']:.3f}")
        report_lines.append(f"  Other:")
        report_lines.append(f"    n = {result['n_other']}")
        report_lines.append(f"    Mean = {result['other_mean']:.3f}")
        report_lines.append(f"    Median = {result['other_median']:.3f}")
        report_lines.append(f"    SD = {result['other_std']:.3f}")
        report_lines.append(f"")
        report_lines.append(f"  Mann-Whitney U Test:")
        report_lines.append(f"    U statistic = {result['statistic']:.3f}")
        report_lines.append(f"    p-value = {result['p_value']:.4f}")
        report_lines.append(f"    Significance: {result['significance']}")
        report_lines.append(f"")
        report_lines.append(f"  Effect Size:")
        report_lines.append(f"    r = {result['effect_r']:.4f}")
        report_lines.append(f"    Effect size: {result['effect_size']}")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    report_file = output_dir / 'statistical_test_details.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n統計検定詳細レポートを保存: {report_file}")
    print("\n" + report_text)

def main():
    """メイン処理"""
    print("="*80)
    print("Best HBユーザー比較 Violin Plot作成")
    print("="*80)
    
    # データ読み込み
    df = load_data()
    
    # 統計検定実行
    results = perform_statistical_tests(df)
    
    # Violin plot作成
    create_violin_plots(df, results, OUTPUT_DIR)
    
    # 統計検定詳細レポート生成
    generate_statistical_report(results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("完了！")
    print("="*80)

if __name__ == "__main__":
    main()
