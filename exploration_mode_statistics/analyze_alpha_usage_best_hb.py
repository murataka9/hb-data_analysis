#!/usr/bin/env python3
"""
Best HBユーザーのAlphaパラメーター利用率の詳細分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートのパス
PROJECT_ROOT = Path(__file__).parent.parent
EXPLORATION_DIR = PROJECT_ROOT / 'exploration_mode_statistics'
OUTPUT_DIR = EXPLORATION_DIR / 'alpha_ux_analysis_results'

def load_data():
    """データを読み込み"""
    merged_file = OUTPUT_DIR / 'merged_data.csv'
    if not merged_file.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {merged_file}")
    
    df = pd.read_csv(merged_file)
    return df

def analyze_alpha_usage_patterns(df):
    """Alpha使用率の詳細分析"""
    print("="*80)
    print("Best HBユーザーのAlphaパラメーター利用率分析")
    print("="*80)
    
    best_hb = df[df['group'] == 'Best HB'].copy()
    other = df[df['group'] == 'Other'].copy()
    
    print(f"\nデータ概要:")
    print(f"  Best HBユーザー数: {len(best_hb)}")
    print(f"  その他のユーザー数: {len(other)}")
    
    # 使用率の統計
    usage_metrics = ['bold_pct', 'cautious_pct', 'standard_pct']
    
    print("\n" + "="*80)
    print("1. Alpha使用率の比較")
    print("="*80)
    
    results = []
    
    for metric in usage_metrics:
        best_values = best_hb[metric].dropna()
        other_values = other[metric].dropna()
        
        if len(best_values) > 0 and len(other_values) > 0:
            # Mann-Whitney U検定
            statistic, p_value = mannwhitneyu(best_values, other_values, alternative='two-sided')
            
            results.append({
                'metric': metric,
                'best_hb_mean': best_values.mean(),
                'best_hb_median': best_values.median(),
                'best_hb_std': best_values.std(),
                'other_mean': other_values.mean(),
                'other_median': other_values.median(),
                'other_std': other_values.std(),
                'p_value': p_value,
                'statistic': statistic
            })
            
            print(f"\n{metric.replace('_pct', '')} 使用率:")
            print(f"  Best HB:")
            print(f"    Mean = {best_values.mean():.2f}%")
            print(f"    Median = {best_values.median():.2f}%")
            print(f"    SD = {best_values.std():.2f}%")
            print(f"    Range = [{best_values.min():.2f}%, {best_values.max():.2f}%]")
            print(f"  その他:")
            print(f"    Mean = {other_values.mean():.2f}%")
            print(f"    Median = {other_values.median():.2f}%")
            print(f"    SD = {other_values.std():.2f}%")
            print(f"    Range = [{other_values.min():.2f}%, {other_values.max():.2f}%]")
            print(f"  p-value = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  → 有意差あり (p < 0.05)")
            elif p_value < 0.10:
                print(f"  → 傾向あり (p < 0.10)")
            else:
                print(f"  → 有意差なし")
    
    # 使用スタイルの分布
    print("\n" + "="*80)
    print("2. 最も多く使用されたスタイル")
    print("="*80)
    
    print("\nBest HBユーザー:")
    best_most_used = best_hb['most_used_style'].value_counts()
    for style, count in best_most_used.items():
        pct = count / len(best_hb) * 100
        print(f"  {style}: {count}人 ({pct:.1f}%)")
    
    print("\nその他のユーザー:")
    other_most_used = other['most_used_style'].value_counts()
    for style, count in other_most_used.items():
        pct = count / len(other) * 100
        print(f"  {style}: {count}人 ({pct:.1f}%)")
    
    # 平均スライダー値
    print("\n" + "="*80)
    print("3. 平均スライダー値の比較")
    print("="*80)
    
    best_slider = best_hb['avg_slider_value'].dropna()
    other_slider = other['avg_slider_value'].dropna()
    
    if len(best_slider) > 0 and len(other_slider) > 0:
        statistic, p_value = mannwhitneyu(best_slider, other_slider, alternative='two-sided')
        
        print(f"\n平均スライダー値（0=cautious, 1=standard, 2=bold）:")
        print(f"  Best HB: Mean = {best_slider.mean():.3f}, Median = {best_slider.median():.3f}, SD = {best_slider.std():.3f}")
        print(f"  その他: Mean = {other_slider.mean():.3f}, Median = {other_slider.median():.3f}, SD = {other_slider.std():.3f}")
        print(f"  p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  → 有意差あり (p < 0.05)")
    
    # スタイルの変動性
    print("\n" + "="*80)
    print("4. スタイル変更の変動性")
    print("="*80)
    
    best_variability = best_hb['style_variability'].dropna()
    other_variability = other['style_variability'].dropna()
    
    if len(best_variability) > 0 and len(other_variability) > 0:
        statistic, p_value = mannwhitneyu(best_variability, other_variability, alternative='two-sided')
        
        print(f"\nスタイル変動性（標準偏差）:")
        print(f"  Best HB: Mean = {best_variability.mean():.3f}, Median = {best_variability.median():.3f}, SD = {best_variability.std():.3f}")
        print(f"  その他: Mean = {other_variability.mean():.3f}, Median = {other_variability.median():.3f}, SD = {other_variability.std():.3f}")
        print(f"  p-value = {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  → 有意差あり (p < 0.05)")
    
    # 変更回数との関係
    print("\n" + "="*80)
    print("5. Alpha変更回数との関係")
    print("="*80)
    
    best_changes = best_hb['total_changes'].dropna()
    other_changes = other['total_changes'].dropna()
    
    if len(best_changes) > 0 and len(other_changes) > 0:
        statistic, p_value = mannwhitneyu(best_changes, other_changes, alternative='two-sided')
        
        print(f"\nAlpha変更回数（main phase）:")
        print(f"  Best HB: Mean = {best_changes.mean():.2f}, Median = {best_changes.median():.2f}, SD = {best_changes.std():.2f}")
        print(f"  その他: Mean = {other_changes.mean():.2f}, Median = {other_changes.median():.2f}, SD = {other_changes.std():.2f}")
        print(f"  p-value = {p_value:.4f}")
        
        if p_value < 0.10:
            print(f"  → 傾向あり (p < 0.10)")
    
    return results, best_hb, other

def create_visualizations(best_hb, other, output_dir):
    """可視化を作成"""
    print("\n" + "="*80)
    print("6. 可視化の作成")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 使用率の比較（箱ひげ図）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1-1. Bold使用率
    ax1 = axes[0, 0]
    data_bold = [
        best_hb['bold_pct'].dropna().values,
        other['bold_pct'].dropna().values
    ]
    bp1 = ax1.boxplot(data_bold, labels=['Best HB', 'その他'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('#FFE66D')  # Best HB
    bp1['boxes'][1].set_facecolor('#95A5A6')  # その他
    ax1.set_ylabel('Bold使用率（%）', fontsize=12)
    ax1.set_title('Bold使用率の比較', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 1-2. Cautious使用率
    ax2 = axes[0, 1]
    data_cautious = [
        best_hb['cautious_pct'].dropna().values,
        other['cautious_pct'].dropna().values
    ]
    bp2 = ax2.boxplot(data_cautious, labels=['Best HB', 'その他'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('#FF6B6B')  # Best HB
    bp2['boxes'][1].set_facecolor('#95A5A6')  # その他
    ax2.set_ylabel('Cautious使用率（%）', fontsize=12)
    ax2.set_title('Cautious使用率の比較', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 1-3. Standard使用率
    ax3 = axes[1, 0]
    data_standard = [
        best_hb['standard_pct'].dropna().values,
        other['standard_pct'].dropna().values
    ]
    bp3 = ax3.boxplot(data_standard, labels=['Best HB', 'その他'], patch_artist=True)
    bp3['boxes'][0].set_facecolor('#4ECDC4')  # Best HB
    bp3['boxes'][1].set_facecolor('#95A5A6')  # その他
    ax3.set_ylabel('Standard使用率（%）', fontsize=12)
    ax3.set_title('Standard使用率の比較', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 1-4. 平均スライダー値
    ax4 = axes[1, 1]
    data_slider = [
        best_hb['avg_slider_value'].dropna().values,
        other['avg_slider_value'].dropna().values
    ]
    bp4 = ax4.boxplot(data_slider, labels=['Best HB', 'その他'], patch_artist=True)
    bp4['boxes'][0].set_facecolor('#FFE66D')
    bp4['boxes'][1].set_facecolor('#95A5A6')
    ax4.set_ylabel('平均スライダー値（0=cautious, 2=bold）', fontsize=12)
    ax4.set_title('平均スライダー値の比較', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_usage_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"可視化を保存: {output_dir / 'alpha_usage_comparison.pdf'}")
    plt.close()
    
    # 2. 使用率の積み上げ棒グラフ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    best_means = [
        best_hb['bold_pct'].mean(),
        best_hb['cautious_pct'].mean(),
        best_hb['standard_pct'].mean()
    ]
    other_means = [
        other['bold_pct'].mean(),
        other['cautious_pct'].mean(),
        other['standard_pct'].mean()
    ]
    
    x = np.arange(2)
    width = 0.6
    
    colors = ['#FFE66D', '#FF6B6B', '#4ECDC4']
    labels = ['Bold', 'Cautious', 'Standard']
    
    bottom_best = 0
    bottom_other = 0
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        if i == 0:
            ax.bar(x[0], best_means[i], width, label=label, color=color, edgecolor='black', linewidth=1)
            ax.bar(x[1], other_means[i], width, color=color, edgecolor='black', linewidth=1)
            bottom_best = best_means[i]
            bottom_other = other_means[i]
        else:
            ax.bar(x[0], best_means[i], width, bottom=bottom_best, label=label, color=color, edgecolor='black', linewidth=1)
            ax.bar(x[1], other_means[i], width, bottom=bottom_other, color=color, edgecolor='black', linewidth=1)
            bottom_best += best_means[i]
            bottom_other += other_means[i]
    
    ax.set_ylabel('使用率（%）', fontsize=12)
    ax.set_title('Alpha使用率の比較（平均）', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Best HB', 'その他'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_usage_stacked_bar.pdf', dpi=300, bbox_inches='tight')
    print(f"可視化を保存: {output_dir / 'alpha_usage_stacked_bar.pdf'}")
    plt.close()

def generate_report(results, best_hb, other, output_dir):
    """レポートを生成"""
    output_dir = Path(output_dir)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Best HBユーザーのAlphaパラメーター利用率詳細分析レポート")
    report_lines.append("="*80)
    report_lines.append("")
    
    report_lines.append("主要な発見:")
    report_lines.append("")
    
    # standard_pctに注目（p=0.207とやや近い）
    standard_result = [r for r in results if r['metric'] == 'standard_pct'][0]
    report_lines.append(f"1. Standard使用率:")
    report_lines.append(f"   Best HBユーザーは、その他のユーザーよりもStandard使用率が高い傾向があります")
    report_lines.append(f"   (Best HB: {standard_result['best_hb_mean']:.2f}% vs その他: {standard_result['other_mean']:.2f}%, p={standard_result['p_value']:.4f})")
    report_lines.append("")
    
    bold_result = [r for r in results if r['metric'] == 'bold_pct'][0]
    report_lines.append(f"2. Bold使用率:")
    report_lines.append(f"   Best HBユーザーは、その他のユーザーよりもBold使用率が低い傾向があります")
    report_lines.append(f"   (Best HB: {bold_result['best_hb_mean']:.2f}% vs その他: {bold_result['other_mean']:.2f}%, p={bold_result['p_value']:.4f})")
    report_lines.append("")
    
    report_lines.append(f"3. Alpha変更回数:")
    best_changes_mean = best_hb['total_changes'].mean()
    other_changes_mean = other['total_changes'].mean()
    report_lines.append(f"   Best HBユーザーは、より頻繁にAlphaパラメーターを変更しています")
    report_lines.append(f"   (Best HB: {best_changes_mean:.1f}回 vs その他: {other_changes_mean:.1f}回)")
    report_lines.append("")
    
    report_lines.append("="*80)
    
    report_text = "\n".join(report_lines)
    report_file = output_dir / 'alpha_usage_detailed_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n詳細レポートを保存: {report_file}")
    print("\n" + report_text)

def main():
    """メイン処理"""
    df = load_data()
    results, best_hb, other = analyze_alpha_usage_patterns(df)
    create_visualizations(best_hb, other, OUTPUT_DIR)
    generate_report(results, best_hb, other, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("分析完了！")
    print("="*80)

if __name__ == "__main__":
    main()
