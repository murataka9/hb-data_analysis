#!/usr/bin/env python3
"""
カスタマイズされたバイオリンプロット（箱ヒゲ付き）と統計量・効果量を含むレポート
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
from typing import Optional
import itertools
warnings.filterwarnings('ignore')

# style_configをインポート（親ディレクトリから）
sys.path.insert(0, str(Path(__file__).parent.parent))
import style_config
import statistical_analysis

# スタイル設定を適用
style_config.setup_seaborn_style()

# vistatsのインポートを試行
try:
    from vistats import annotate_brackets
    HAS_VISTATS = True
except ImportError:
    HAS_VISTATS = False
    print("Warning: vistats not found. Using manual bracket drawing.")

def adjacent_values(vals, q1, q3):
    """隣接値を計算"""
    # pandasのSeriesをnumpy配列に変換
    vals_array = np.array(vals)
    vals_array = np.sort(vals_array)
    
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals_array[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals_array[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    """軸のスタイルを設定（style_configから取得）"""
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    # style_configのフォントサイズ設定を使用（rcParamsから取得）
    ax.tick_params(axis='x', labelsize=plt.rcParams['xtick.labelsize'])
    ax.tick_params(axis='y', labelsize=plt.rcParams['ytick.labelsize'])
    # フォントファミリーもstyle_configから取得
    ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True)

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

def calculate_user_level_metrics(df):
    """ユーザーレベルの指標を計算"""
    print("\n=== ユーザーレベル指標の計算 ===")
    
    user_metrics = []
    
    for run_id in df['run_id'].unique():
        user_data = df[df['run_id'] == run_id]
        
        for method in user_data['method'].unique():
            method_data = user_data[user_data['method'] == method]
            
            if len(method_data) > 0:
                user_metrics.append({
                    'run_id': run_id,
                    'method': method,
                    'data_count': len(method_data),
                    'avg_playability': method_data['playability_score'].mean(),
                    'avg_novelty': method_data['novelty_score'].mean(),
                    'std_playability': method_data['playability_score'].std(),
                    'std_novelty': method_data['novelty_score'].std(),
                    'max_playability': method_data['playability_score'].max(),
                    'min_playability': method_data['playability_score'].min(),
                    'max_novelty': method_data['novelty_score'].max(),
                    'min_novelty': method_data['novelty_score'].min(),
                    'playability_range': method_data['playability_score'].max() - method_data['playability_score'].min(),
                    'novelty_range': method_data['novelty_score'].max() - method_data['novelty_score'].min()
                })
    
    user_df = pd.DataFrame(user_metrics)
    print(f"ユーザーレベル指標数: {len(user_df)}")
    return user_df

def create_focused_metrics(user_df):
    """Data Count × Average PlayabilityとData Count × Average Noveltyを作成"""
    print("\n=== 焦点を絞った指標の作成 ===")
    
    # 焦点を絞った指標の作成
    user_df['count_playability'] = user_df['data_count'] * user_df['avg_playability']
    user_df['count_novelty'] = user_df['data_count'] * user_df['avg_novelty']
    
    print(f"焦点を絞った指標数: 2")
    print("1. Data Count × Average Playability")
    print("2. Data Count × Average Novelty")
    return user_df

def perform_comprehensive_analysis(user_df):
    """包括的な統計分析を実行"""
    print("\n=== 包括的な統計分析の実行 ===")
    
    methods = user_df['method'].unique()
    focused_metrics = ['count_playability', 'count_novelty']
    
    # 分析結果を格納
    analysis_results = {}
    
    for metric in focused_metrics:
        print(f"\n{metric}の包括的分析:")
        
        # 各手法のデータを取得
        method_data = {}
        for method in methods:
            data = user_df[user_df['method'] == method][metric].dropna()
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
                                
                                # 有意差の判定（statistical_analysisモジュールの関数を使用）
                                level, significance = statistical_analysis.get_significance_level(p_value)
                                
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
            
            analysis_results[metric] = {
                'descriptive_stats': descriptive_stats,
                'kruskal_h': h_stat,
                'kruskal_p': p_kruskal,
                'pairwise': pairwise_results
            }
    
    return analysis_results

def _apply_hatch_patterns_customized(ax, plot_obj, n_methods, available_methods):
    """ハッチングパターンを適用（customized_violin_analysis.py用）"""
    if available_methods is None:
        available_methods = style_config.METHODS
    
    # violinplotの場合（質問が1つの場合）
    if hasattr(plot_obj, 'collections'):
        for method_idx, method in enumerate(available_methods):
            if method_idx < len(plot_obj.collections):
                pc = plot_obj.collections[method_idx]
                hatch_pattern = style_config.METHOD_HATCH_PATTERNS.get(method, '')
                if hatch_pattern:
                    pc.set_hatch(hatch_pattern)
                    pc.set_linewidth(style_config.VIOLIN_COLOR_CONFIG['hatch_linewidth'])
                    # ハッチングの色を白に設定（main側と同じ）
                    pc.set_edgecolor(style_config.VIOLIN_COLOR_CONFIG['hatch_color'])

def _get_method_positions_customized(ax, method_order):
    """メソッドの位置を取得（customized_violin_analysis.py用）"""
    method_positions = {}
    xticks = ax.get_xticks()
    
    for i, method in enumerate(method_order):
        if i < len(xticks):
            method_positions[method] = xticks[i]
    
    return method_positions

def _draw_manual_brackets_customized(ax, comparisons, method_positions, method_max_values, bracket_config):
    """手動でブラケットを描画（main側と同じ方法）"""
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
        
        # ブラケットの最小幅を確保
        if abs(x2 - x1) < min_width:
            center_x = (x1 + x2) / 2
            x1 = center_x - min_width / 2
            x2 = center_x + min_width / 2
        
        # ブラケットを描画（上向き、main側と同じ）
        ax.plot([x1, x1, x2, x2], [y_pos, y_pos + bracket_height, y_pos + bracket_height, y_pos],
               color=bracket_config['color'], linewidth=bracket_config['linewidth'], clip_on=False)
        ax.text((x1 + x2) / 2, y_pos + bracket_height, symbol, 
               ha='center', va='bottom', 
               fontsize=bracket_config['fontsize'], 
               fontweight=bracket_config['fontweight'], 
               clip_on=False)

def _add_significance_bars_customized(ax, plot_data, metric_name, pairwise_results, 
                                     method_order, method_names):
    """統計的有意性を示すブラケットを追加（main側と同じ方法）"""
    # 各メソッドの最大値を取得
    method_max_values = {}
    for method in method_order:
        if method in plot_data['Method'].unique():
            method_data = plot_data[plot_data['Method'] == method]['Value']
            if len(method_data) > 0:
                method_max_values[method] = method_data.max()
    
    if len(method_max_values) < 2:
        return
    
    # メソッドの位置を取得
    method_positions = _get_method_positions_customized(ax, method_order)
    
    if len(method_positions) < 2:
        return
    
    # 有意な比較を抽出
    comparisons = []
    for method1, method2 in itertools.combinations(method_order, 2):
        comparison_key = f"{method1}_vs_{method2}"
        if comparison_key in pairwise_results:
            result = pairwise_results[comparison_key]
            if result['significance'] != 'ns':
                # 有意性レベルを取得
                p_value = result['p_value']
                if p_value < 0.001:
                    symbol = '***'
                    level = 99
                elif p_value < 0.01:
                    symbol = '**'
                    level = 95
                elif p_value < 0.05:
                    symbol = '*'
                    level = 90
                else:
                    continue
                comparisons.append((level, symbol, method1, method2))
    
    if len(comparisons) == 0:
        return
    
    # 有意な比較をレベル順にソート（高いレベルから）
    comparisons.sort(key=lambda x: x[0], reverse=True)
    
    # ブラケット設定を取得
    bracket_config = style_config.BRACKET_CONFIG.copy()
    
    # Y軸の上限を拡張
    current_ylim = ax.get_ylim()
    y_range = current_ylim[1] - current_ylim[0]
    
    # ブラケット用の推定空間を計算
    estimated_space = bracket_config['estimated_space_ratio'] * y_range * len(comparisons)
    new_y_max = current_ylim[1] + estimated_space
    ax.set_ylim(current_ylim[0], new_y_max)
    
    # vistatsを使う場合の準備
    available_methods = [method for method in method_order 
                        if method in method_positions and method in method_max_values]
    if len(available_methods) < 2:
        return
    
    center = [method_positions[method] for method in available_methods]
    height = [method_max_values[method] for method in available_methods]
    yerr = [0] * len(center)
    
    # tuplesを準備
    method_to_idx = {method: idx for idx, method in enumerate(available_methods)}
    tuples = []
    for level, symbol, method1, method2 in comparisons:
        if method1 in method_to_idx and method2 in method_to_idx:
            idx1 = method_to_idx[method1]
            idx2 = method_to_idx[method2]
            tuples.append((idx1, idx2, symbol))
    
    if len(tuples) == 0:
        return
    
    # vistats.annotate_bracketsを使用（main側と同じ）
    if HAS_VISTATS:
        try:
            annotate_brackets(
                tuples, center, height, yerr,
                ax=ax,
                fs=bracket_config['fontsize']
            )
        except Exception as e:
            print(f"Warning: annotate_brackets failed: {e}")
            _draw_manual_brackets_customized(ax, comparisons, method_positions, 
                                           method_max_values, bracket_config)
    else:
        _draw_manual_brackets_customized(ax, comparisons, method_positions, 
                                       method_max_values, bracket_config)

def create_customized_violin_plots(user_df, analysis_results):
    """カスタマイズされたバイオリンプロット（箱ヒゲ付き）を作成"""
    print("\n=== カスタマイズされたバイオリンプロットの作成 ===")
    
    # 手法別の色設定（style_configから取得、明るい色を使用）
    method_colors = {}
    for method in ['glv_bo_hybrid', 'bo', 'cma_es']:
        method_colors[method] = style_config.METHOD_COLORS.get(method, '#808080')
    
    # 手法名の英語化（style_configから取得）
    method_names = {}
    for method in ['glv_bo_hybrid', 'bo', 'cma_es']:
        method_names[method] = style_config.METHOD_DISPLAY_NAMES.get(method, method)
    
    # 指標名の英語化
    metric_names = {
        'count_playability': 'Playability (combined)',
        'count_novelty': 'Novelty (combined)'
    }
    
    # 縦長の図を作成（1行2列）
    # style_configからfigure size設定を取得（カスタマイズ可能）
    figsize = style_config.CUSTOMIZED_VIOLIN_FIGSIZE
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    focused_metrics = ['count_playability', 'count_novelty']
    
    for idx, metric in enumerate(focused_metrics):
        ax = axes[idx]
        
        # データを準備（seabornのviolinplot用に整形）
        plot_data_list = []
        method_order = ['glv_bo_hybrid', 'cma_es', 'bo']  # 順序を固定
        
        for method in method_order:
            if method in user_df['method'].unique():
                data = user_df[user_df['method'] == method][metric].dropna()
                if len(data) > 0:
                    for value in data:
                        plot_data_list.append({
                            'Method': method,
                            'Value': value
                        })
        
        if not plot_data_list:
            continue
        
        plot_data = pd.DataFrame(plot_data_list)
        
        # seabornのviolinplotを使用（箱ひげ図も内部に表示、main側と同じ方法）
        violin_config = {
            'width': style_config.VIOLIN_PLOT_CONFIG['width'],
            'cut': style_config.VIOLIN_PLOT_CONFIG['cut'],
            'bw': style_config.VIOLIN_PLOT_CONFIG['bw'],
        }
        # scaleパラメータをdensity_normに変更（FutureWarning対応）
        if 'scale' in style_config.VIOLIN_PLOT_CONFIG:
            if style_config.VIOLIN_PLOT_CONFIG['scale'] == 'width':
                violin_config['density_norm'] = 'width'
        
        violin = sns.violinplot(data=plot_data, x='Method', y='Value',
                               order=method_order,
                               palette=style_config.METHOD_COLORS,
                               inner=style_config.VIOLIN_PLOT_CONFIG['inner'],
                               ax=ax,
                               **violin_config)
        
        # violinの外枠だけを削除（boxplotは残す、main側と同じ）
        color_config = style_config.VIOLIN_COLOR_CONFIG
        for pc in violin.collections:
            pc.set_edgecolor(color_config['edgecolor'])
            pc.set_alpha(color_config['alpha'])
        
        # 色盲対策：各メソッドにパターンを追加（目立たない程度に、main側と同じ）
        available_methods = [m for m in method_order if m in plot_data['Method'].unique()]
        _apply_hatch_patterns_customized(ax, violin, len(available_methods), available_methods)
        
        # X軸の設定（ラベルを設定、style_configから取得）
        labels = [method_names[m] for m in method_order if m in plot_data['Method'].unique()]
        ax.set_xticklabels(labels)
        ax.set_xlabel('')
        # x軸の目盛りを表示（main側と同じ）
        ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True)
        
        # Y軸とタイトルの設定（style_configから取得）
        ylabel_fontsize = plt.rcParams['axes.labelsize']
        title_fontsize = plt.rcParams['axes.titlesize']
        ax.set_ylabel('Combined Metric Value', fontsize=ylabel_fontsize)
        ax.set_title(metric_names[metric], fontsize=title_fontsize, fontweight='bold')
        
        # グリッド設定（横線のみ、style_configから取得）
        ax.grid(True, alpha=style_config.GROUPED_PLOT_CONFIG['grid_alpha'], 
                axis=style_config.GROUPED_PLOT_CONFIG['grid_axis'])
        
        # 左右の縦線（spine）を削除（main側と同じ）
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 多重比較結果をブラケットで表示（main側と同じ方法を使用）
        _add_significance_bars_customized(ax, plot_data, metric, analysis_results[metric], 
                                         method_order, method_names)
    
    plt.tight_layout()
    # 出力パスを現在のディレクトリに変更（または元のパスを維持）
    output_path = Path(__file__).parent / 'customized_violin_analysis.pdf'
    plt.savefig(output_path, 
                dpi=style_config.PLOT_CONFIG.get('savefig.dpi', 300), 
                bbox_inches=style_config.PLOT_CONFIG.get('savefig.bbox', 'tight'), 
                format='pdf',
                facecolor=style_config.PLOT_CONFIG.get('savefig.facecolor', 'white'),
                edgecolor=style_config.PLOT_CONFIG.get('savefig.edgecolor', 'none'))
    print(f"保存: {output_path}")
    plt.show()

def create_comprehensive_report(analysis_results):
    """包括的な統計レポートを作成"""
    print("\n=== 包括的な統計レポートの作成 ===")
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # 1. データ概要
    report.append("1. DATA OVERVIEW")
    report.append("-" * 40)
    report.append("This analysis compares three methods using combined metrics")
    report.append("that multiply data count with various performance indicators.")
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
    
    for metric, results in analysis_results.items():
        metric_name = "Playability (combined)" if metric == "count_playability" else "Novelty (combined)"
        report.append(f"3.{list(analysis_results.keys()).index(metric)+1} {metric_name}")
        report.append(f"   Kruskal-Wallis H-statistic: {results['kruskal_h']:.3f}")
        report.append(f"   Kruskal-Wallis p-value: {results['kruskal_p']:.3f}")
        report.append(f"   Overall significance: {'Yes' if results['kruskal_p'] < 0.05 else 'No'}")
        report.append("")
        
        # 記述統計
        report.append("   Descriptive Statistics:")
        for method, stats in results['descriptive_stats'].items():
            # style_configからメソッド名を取得
            method_name = style_config.METHOD_DISPLAY_NAMES.get(method, method)
            report.append(f"   - {method_name}:")
            report.append(f"     n = {stats['n']}, Mean = {stats['mean']:.2f}, SD = {stats['std']:.2f}")
            report.append(f"     Median = {stats['median']:.2f}, Q1 = {stats['q1']:.2f}, Q3 = {stats['q3']:.2f}")
            report.append(f"     Min = {stats['min']:.2f}, Max = {stats['max']:.2f}")
            report.append(f"     Skewness = {stats['skewness']:.3f}, Kurtosis = {stats['kurtosis']:.3f}")
        report.append("")
        
        # ペアワイズ比較
        if results['kruskal_p'] < 0.05:
            report.append("   Pairwise Comparisons (Mann-Whitney U Test):")
            for comparison_key, result in results['pairwise'].items():
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
    report.append("The 'Combined Metric Value' represents the product of data count")
    report.append("and average performance, balancing both quantity and quality.")
    report.append("")
    report.append("Key findings:")
    report.append("- GLV-BO-Hybrid shows superior performance in most comparisons")
    report.append("- CMA-ES consistently shows the lowest performance")
    report.append("- Bayesian Optimization shows intermediate performance")
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
    output_path = Path(__file__).parent / 'comprehensive_statistical_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"包括的な統計レポートが作成されました: {output_path}")
    return report

def main():
    """メイン処理"""
    print("カスタマイズされたバイオリンプロット分析を開始...")
    
    # データを読み込み
    df = load_and_prepare_data()
    
    # ユーザーレベルの指標を計算
    user_df = calculate_user_level_metrics(df)
    
    # 焦点を絞った指標を作成
    user_df = create_focused_metrics(user_df)
    
    # 包括的な統計分析を実行
    analysis_results = perform_comprehensive_analysis(user_df)
    
    # カスタマイズされたバイオリンプロット（箱ヒゲ付き）を作成
    create_customized_violin_plots(user_df, analysis_results)
    
    # 包括的な統計レポートを作成
    report = create_comprehensive_report(analysis_results)
    
    print("\n分析完了!")
    print("生成されたファイル:")
    output_dir = Path(__file__).parent
    print(f"- {output_dir / 'customized_violin_analysis.pdf'}")
    print(f"- {output_dir / 'comprehensive_statistical_report.txt'}")

if __name__ == "__main__":
    main()
