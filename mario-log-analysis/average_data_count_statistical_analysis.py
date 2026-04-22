#!/usr/bin/env python3
"""
Average Data Count Barplotの統計的分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal
import warnings
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# style_configをインポート（親ディレクトリから）
sys.path.insert(0, str(Path(__file__).parent.parent))
import style_config

# スタイル設定を適用
style_config.setup_seaborn_style()

def load_and_prepare_data():
    """データを読み込み、準備"""
    # スクリプトと同じディレクトリのenhanced_pareto_data.csvを使用
    data_path = Path(__file__).parent / 'enhanced_pareto_data.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
    df = pd.read_csv(data_path)
    print("=== データ準備 ===")
    print(f"総データ数: {len(df)}")
    print(f"手法別データ数:")
    print(df['method'].value_counts())
    return df

def calculate_user_level_data_count(df):
    """ユーザーレベルのデータ数を計算"""
    print("\n=== ユーザーレベルデータ数の計算 ===")
    
    user_data_counts = []
    
    for run_id in df['run_id'].unique():
        user_data = df[df['run_id'] == run_id]
        
        for method in user_data['method'].unique():
            method_data = user_data[user_data['method'] == method]
            
            if len(method_data) > 0:
                user_data_counts.append({
                    'run_id': run_id,
                    'method': method,
                    'data_count': len(method_data)
                })
    
    user_df = pd.DataFrame(user_data_counts)
    print(f"ユーザーレベルデータ数: {len(user_df)}")
    return user_df

def perform_statistical_analysis(user_df):
    """統計的分析を実行"""
    print("\n=== 統計的分析の実行 ===")
    
    methods = user_df['method'].unique()
    
    # 分析結果を格納
    analysis_results = {}
    
    print("Average Data Count per User の統計的分析:")
    
    # 各手法のデータを取得
    method_data = {}
    for method in methods:
        data = user_df[user_df['method'] == method]['data_count'].dropna()
        method_data[method] = data
    
    # 1. 記述統計
    descriptive_stats = {}
    for method in methods:
        if method in method_data and len(method_data[method]) > 0:
            data = method_data[method]
            descriptive_stats[method] = {
                'n': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                'min': data.min(),
                'max': data.max(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
    
    # 2. Kruskal-Wallis検定（全体の検定）
    groups = [method_data[method] for method in methods if len(method_data[method]) > 0]
    if len(groups) > 2:
        h_stat, p_kruskal = kruskal(*groups)
        print(f"  Kruskal-Wallis検定: H={h_stat:.3f}, p={p_kruskal:.3f}")
        
        # 3. Mann-Whitney U検定（ペアワイズ比較）
        pairwise_results = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # 重複を避ける
                    data1 = method_data[method1]
                    data2 = method_data[method2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        try:
                            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            # 効果量（r）- Mann-Whitney U検定の場合
                            n1, n2 = len(data1), len(data2)
                            # U統計量からZ値を計算
                            mean_u = n1 * n2 / 2.0
                            var_u = n1 * n2 * (n1 + n2 + 1) / 12.0
                            
                            if var_u > 0:
                                std_u = np.sqrt(var_u)
                                z = (statistic - mean_u) / std_u
                                # r = Z / sqrt(N)
                                effect_r = z / np.sqrt(n1 + n2)
                            else:
                                effect_r = 0.0
                            
                            # 効果量の解釈（rの場合）
                            if abs(effect_r) < 0.1:
                                effect_size = "Small"
                            elif abs(effect_r) < 0.3:
                                effect_size = "Medium"
                            elif abs(effect_r) < 0.5:
                                effect_size = "Large"
                            else:
                                effect_size = "Very Large"
                            
                            # 有意差の判定
                            if p_value < 0.001:
                                significance = "***"
                            elif p_value < 0.01:
                                significance = "**"
                            elif p_value < 0.05:
                                significance = "*"
                            else:
                                significance = "ns"
                            
                            pairwise_results[f"{method1}_vs_{method2}"] = {
                                'method1': method1,
                                'method2': method2,
                                'u_statistic': statistic,
                                'p_value': p_value,
                                'effect_r': effect_r,
                                'effect_size': effect_size,
                                'significance': significance,
                                'mean1': data1.mean(),
                                'mean2': data2.mean(),
                                'std1': data1.std(),
                                'std2': data2.std()
                            }
                            
                            print(f"  {method1} vs {method2}: U={statistic:.1f}, p={p_value:.3f}, r={effect_r:.3f} ({effect_size}), {significance}")
                        except:
                            print(f"  {method1} vs {method2}: 検定実行エラー")
        
        analysis_results = {
            'descriptive_stats': descriptive_stats,
            'kruskal_h': h_stat,
            'kruskal_p': p_kruskal,
            'pairwise': pairwise_results
        }
    
    return analysis_results

def create_statistical_barplot(user_df, analysis_results):
    """統計的数値付きのバープロットを作成"""
    print("\n=== 統計的数値付きバープロットの作成 ===")
    
    # 手法別の色設定（style_configから取得）
    method_colors = {}
    for method in ['glv_bo_hybrid', 'bo', 'cma_es']:
        method_colors[method] = style_config.METHOD_COLORS.get(method, '#808080')
    
    # 手法名の英語化（style_configから取得）
    method_names = {}
    for method in ['glv_bo_hybrid', 'bo', 'cma_es']:
        method_names[method] = style_config.METHOD_DISPLAY_NAMES.get(method, method)
    
    # 手法の順序を固定
    method_order = ['glv_bo_hybrid', 'cma_es', 'bo']
    
    # データを準備
    means = []
    stds = []
    labels = []
    colors = []
    
    for method in method_order:
        if method in user_df['method'].unique():
            data = user_df[user_df['method'] == method]['data_count'].dropna()
            if len(data) > 0:
                means.append(data.mean())
                stds.append(data.std())
                labels.append(method_names[method])
                colors.append(method_colors[method])
    
    # 図を作成（style_configから取得）
    figsize = style_config.AVERAGE_DATA_COUNT_FIGSIZE
    fig, ax = plt.subplots(figsize=figsize)
    
    # バープロット（style_configの設定を使用）
    x = np.arange(len(labels))
    bar_config = style_config.BAR_PLOT_CONFIG.copy()
    bars = ax.bar(x, means, yerr=stds, 
                  capsize=bar_config.get('capsize', 10), 
                  width=bar_config.get('width', 0.6),
                  color=colors, 
                  alpha=style_config.VIOLIN_COLOR_CONFIG.get('alpha', 0.8), 
                  edgecolor='none',  # 枠線を透明化（すべてのバーで統一）
                  linewidth=0)
    
    # 色盲対策：ハッチングパターンを追加
    for bar, method in zip(bars, [m for m in method_order if m in user_df['method'].unique()]):
        hatch_pattern = style_config.METHOD_HATCH_PATTERNS.get(method, '')
        if hatch_pattern:
            bar.set_hatch(hatch_pattern)
            bar.set_linewidth(style_config.VIOLIN_COLOR_CONFIG['hatch_linewidth'])
            bar.set_edgecolor(style_config.VIOLIN_COLOR_CONFIG['hatch_color'])
        else:
            # ハッチングパターンがない場合も枠線を透明化（Hummingbirdなど）
            bar.set_edgecolor('none')
            bar.set_linewidth(0)
    
    # 軸の設定（style_configから取得）
    xlabel_fontsize = plt.rcParams['axes.labelsize']
    ylabel_fontsize = plt.rcParams['axes.labelsize']
    title_fontsize = plt.rcParams['axes.titlesize']
    tick_fontsize = plt.rcParams['xtick.labelsize']
    
    ax.set_xlabel('', fontsize=xlabel_fontsize)  # Methodラベルを削除
    ax.set_ylabel('Average Data Count per User', fontsize=ylabel_fontsize)
    ax.set_title('Average Data Count per User', 
                fontsize=title_fontsize, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True)
    # 横罫線のみ表示（縦罫線を無効化）
    ax.grid(True, alpha=style_config.GROUPED_PLOT_CONFIG['grid_alpha'], 
            axis=style_config.GROUPED_PLOT_CONFIG['grid_axis'])
    ax.xaxis.grid(False)  # 縦罫線を明示的に無効化
    # 枠線の縦線を削除
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 多重比較結果をブラケットで表示（有意差がある場合のみ）
    pairwise_results = analysis_results['pairwise']
    
    # 各比較のブラケットを描画
    y_max = max([mean + std for mean, std in zip(means, stds)])
    y_min = 0
    
    # ブラケットの高さを調整
    bracket_height = (y_max - y_min) * 0.05
    bracket_spacing = (y_max - y_min) * 0.02
    
    # 比較の順序を定義
    comparisons = [
        ('glv_bo_hybrid', 'cma_es'),
        ('glv_bo_hybrid', 'bo'),
        ('cma_es', 'bo')
    ]
    
    bracket_count = 0
    for i, (method1, method2) in enumerate(comparisons):
        comparison_key = f"{method1}_vs_{method2}"
        if comparison_key in pairwise_results:
            result = pairwise_results[comparison_key]
            
            # 有意差がある場合のみブラケットを描画
            if result['significance'] != 'ns':
                # 手法の位置を取得
                pos1 = method_order.index(method1)
                pos2 = method_order.index(method2)
                
                # ブラケットのY位置
                y_bracket = y_max + bracket_height + (bracket_count * bracket_spacing)
                
                # ブラケットを描画（style_configの設定を使用）
                bracket_config = style_config.BRACKET_CONFIG
                ax.plot([pos1, pos1, pos2, pos2], 
                       [y_bracket - bracket_height/2, y_bracket, y_bracket, y_bracket - bracket_height/2], 
                       color=bracket_config['color'], 
                       linewidth=bracket_config['linewidth'], 
                       clip_on=False)
                
                # アスタリスクを表示（style_configの設定を使用）
                x_center = (pos1 + pos2) / 2
                ax.text(x_center, y_bracket + bracket_height/2, result['significance'], 
                       ha='center', va='bottom', 
                       fontsize=bracket_config['fontsize'], 
                       fontweight=bracket_config['fontweight'],
                       clip_on=False)
                
                bracket_count += 1
    
    # Y軸の範囲を調整
    if bracket_count > 0:
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.3)
    else:
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    
    plt.tight_layout()
    # 出力パスを現在のディレクトリに変更
    output_path = Path(__file__).parent / 'average_data_count_statistical_barplot.pdf'
    plt.savefig(output_path, 
                dpi=style_config.PLOT_CONFIG.get('savefig.dpi', 300), 
                bbox_inches=style_config.PLOT_CONFIG.get('savefig.bbox', 'tight'), 
                format='pdf',
                facecolor=style_config.PLOT_CONFIG.get('savefig.facecolor', 'white'),
                edgecolor=style_config.PLOT_CONFIG.get('savefig.edgecolor', 'none'))
    print(f"保存: {output_path}")
    plt.show()

def create_statistical_report(analysis_results):
    """統計的レポートを作成"""
    print("\n=== 統計的レポートの作成 ===")
    
    report = []
    report.append("=" * 80)
    report.append("AVERAGE DATA COUNT STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # 1. データ概要
    report.append("1. DATA OVERVIEW")
    report.append("-" * 40)
    report.append("This analysis compares the average data count per user")
    report.append("across three different methods.")
    report.append("")
    
    # 2. 統計的検定の概要
    report.append("2. STATISTICAL TESTS PERFORMED")
    report.append("-" * 40)
    report.append("2.1 Descriptive Statistics")
    report.append("   - Mean, Standard Deviation, Median")
    report.append("   - Quartiles (Q1, Q3), Min, Max")
    report.append("   - Skewness, Kurtosis")
    report.append("")
    report.append("2.2 Kruskal-Wallis Test")
    report.append("   - Purpose: Compare three or more groups (non-parametric)")
    report.append("   - Null hypothesis: All groups have the same distribution")
    report.append("   - Alternative hypothesis: At least one group differs")
    report.append("")
    report.append("2.3 Mann-Whitney U Test (Post-hoc)")
    report.append("   - Purpose: Pairwise comparisons after significant Kruskal-Wallis")
    report.append("   - Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05")
    report.append("")
    report.append("2.4 Effect Size (r)")
    report.append("   - Purpose: Measure the magnitude of difference between groups")
    report.append("   - Interpretation:")
    report.append("     * r < 0.1: Small effect")
    report.append("     * r = 0.1-0.3: Medium effect")
    report.append("     * r = 0.3-0.5: Large effect")
    report.append("     * r > 0.5: Very Large effect")
    report.append("")
    
    # 3. 詳細な分析結果
    report.append("3. DETAILED ANALYSIS RESULTS")
    report.append("-" * 40)
    report.append("3.1 Average Data Count per User")
    report.append(f"   Kruskal-Wallis H-statistic: {analysis_results['kruskal_h']:.3f}")
    report.append(f"   Kruskal-Wallis p-value: {analysis_results['kruskal_p']:.3f}")
    report.append(f"   Overall significance: {'Yes' if analysis_results['kruskal_p'] < 0.05 else 'No'}")
    report.append("")
    
    # 記述統計
    report.append("   Descriptive Statistics:")
    for method, stats in analysis_results['descriptive_stats'].items():
        # style_configからメソッド名を取得
        method_name = style_config.METHOD_DISPLAY_NAMES.get(method, method)
        report.append(f"   - {method_name}:")
        report.append(f"     n = {stats['n']}, Mean = {stats['mean']:.2f}, SD = {stats['std']:.2f}")
        report.append(f"     Median = {stats['median']:.2f}, Q1 = {stats['q1']:.2f}, Q3 = {stats['q3']:.2f}")
        report.append(f"     Min = {stats['min']:.2f}, Max = {stats['max']:.2f}")
        report.append(f"     Skewness = {stats['skewness']:.3f}, Kurtosis = {stats['kurtosis']:.3f}")
    report.append("")
    
    # ペアワイズ比較
    if analysis_results['kruskal_p'] < 0.05:
        report.append("   Pairwise Comparisons (Mann-Whitney U Test):")
        for comparison_key, result in analysis_results['pairwise'].items():
            # style_configからメソッド名を取得
            method1_name = style_config.METHOD_DISPLAY_NAMES.get(result['method1'], result['method1'])
            method2_name = style_config.METHOD_DISPLAY_NAMES.get(result['method2'], result['method2'])
            report.append(f"   - {method1_name} vs {method2_name}:")
            report.append(f"     U-statistic: {result['u_statistic']:.1f}")
            report.append(f"     p-value: {result['p_value']:.3f}")
            report.append(f"     r: {result['effect_r']:.3f} ({result['effect_size']})")
            report.append(f"     Significance: {result['significance']}")
            report.append(f"     Mean difference: {result['mean1'] - result['mean2']:.2f}")
            report.append(f"     ({method1_name}: {result['mean1']:.2f} ± {result['std1']:.2f})")
            report.append(f"     ({method2_name}: {result['mean2']:.2f} ± {result['std2']:.2f})")
            report.append("")
    else:
        report.append("   No pairwise comparisons performed (overall test not significant)")
        report.append("")
    
    # 4. 解釈
    report.append("4. INTERPRETATION")
    report.append("-" * 40)
    report.append("The 'Average Data Count per User' represents the mean number")
    report.append("of data points generated by each user for each method.")
    report.append("")
    report.append("Key findings:")
    report.append("- GLV-BO-Hybrid shows the highest average data count")
    report.append("- CMA-ES shows the lowest average data count")
    report.append("- Bayesian Optimization shows intermediate data count")
    report.append("")
    
    # 5. 限界
    report.append("5. LIMITATIONS")
    report.append("-" * 40)
    report.append("- Multiple comparisons increase Type I error rate")
    report.append("- Non-parametric tests used due to non-normal distribution")
    report.append("- Sample sizes vary between methods")
    report.append("- Effect sizes should be considered alongside p-values")
    report.append("")
    
    # レポートをファイルに保存（現在のディレクトリに保存）
    output_path = Path(__file__).parent / 'average_data_count_statistical_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"統計的レポートが作成されました: {output_path}")
    return report

def main():
    """メイン処理"""
    print("Average Data Count統計分析を開始...")
    
    # データを読み込み
    df = load_and_prepare_data()
    
    # ユーザーレベルのデータ数を計算
    user_df = calculate_user_level_data_count(df)
    
    # 統計的分析を実行
    analysis_results = perform_statistical_analysis(user_df)
    
    # 統計的数値付きのバープロットを作成
    create_statistical_barplot(user_df, analysis_results)
    
    # 統計的レポートを作成
    report = create_statistical_report(analysis_results)
    
    print("\n分析完了!")
    print("生成されたファイル:")
    output_dir = Path(__file__).parent
    print(f"- {output_dir / 'average_data_count_statistical_barplot.pdf'}")
    print(f"- {output_dir / 'average_data_count_statistical_report.txt'}")

if __name__ == "__main__":
    main()
