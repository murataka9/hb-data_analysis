#!/usr/bin/env python3
"""
Alphaパラメーター変更（exploration_mode）とUX指標の相関分析
本番（main phase）のデータのみを対象に、marioタスクのデータを分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートのパス
PROJECT_ROOT = Path(__file__).parent.parent
EXPLORATION_DIR = PROJECT_ROOT / 'exploration_mode_statistics'
MARIO_ORIGINAL_DIR = PROJECT_ROOT / 'mario' / 'original'
MARIO_LOG_DIR = PROJECT_ROOT / 'mario-log-analysis'

def load_exploration_mode_data():
    """exploration_modeのタイムラインデータを読み込み、main phaseのみを抽出"""
    timeline_file = EXPLORATION_DIR / 'exploration_mode_timeline.csv'
    if not timeline_file.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {timeline_file}")
    
    df = pd.read_csv(timeline_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # main phaseのみを抽出
    df_main = df[df['phase'] == 'main'].copy()
    
    print(f"=== Exploration Modeデータ読み込み ===")
    print(f"総データ数: {len(df)}")
    print(f"Main phaseデータ数: {len(df_main)}")
    print(f"ユーザー数: {df_main['run_id'].nunique()}")
    
    return df_main

def calculate_alpha_usage_patterns(df_main):
    """ユーザーごとのalpha（exploration_mode）使用パターンを計算"""
    print("\n=== Alpha使用パターンの計算 ===")
    
    user_patterns = []
    
    for run_id in df_main['run_id'].unique():
        user_data = df_main[df_main['run_id'] == run_id].copy()
        
        # スタイル変更回数
        total_changes = len(user_data)
        
        # 各スタイルの使用回数と割合
        style_counts = user_data['style'].value_counts()
        bold_count = style_counts.get('bold', 0)
        cautious_count = style_counts.get('cautious', 0)
        standard_count = style_counts.get('standard', 0)
        
        bold_pct = (bold_count / total_changes * 100) if total_changes > 0 else 0
        cautious_pct = (cautious_count / total_changes * 100) if total_changes > 0 else 0
        standard_pct = (standard_count / total_changes * 100) if total_changes > 0 else 0
        
        # スライダー値の平均（0=cautious, 1=standard, 2=bold）
        valid_slider = user_data[user_data['slider_value'] >= 0]['slider_value']
        avg_slider_value = valid_slider.mean() if len(valid_slider) > 0 else np.nan
        
        # 最も多く使われたスタイル
        most_used_style = style_counts.index[0] if len(style_counts) > 0 else 'unknown'
        
        # スタイルの変動性（標準偏差）
        style_variability = valid_slider.std() if len(valid_slider) > 0 else np.nan
        
        user_patterns.append({
            'run_id': run_id,
            'total_changes': total_changes,
            'bold_count': bold_count,
            'cautious_count': cautious_count,
            'standard_count': standard_count,
            'bold_pct': bold_pct,
            'cautious_pct': cautious_pct,
            'standard_pct': standard_pct,
            'avg_slider_value': avg_slider_value,
            'most_used_style': most_used_style,
            'style_variability': style_variability,
            'method': user_data['method'].iloc[0] if len(user_data) > 0 else 'unknown'
        })
    
    patterns_df = pd.DataFrame(user_patterns)
    print(f"ユーザーパターン数: {len(patterns_df)}")
    
    return patterns_df

def load_playability_data():
    """Playabilityデータを読み込み、ユーザーレベルの指標を計算"""
    playability_file = MARIO_LOG_DIR / 'enhanced_pareto_data.csv'
    if not playability_file.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {playability_file}")
    
    df = pd.read_csv(playability_file)
    
    # glv_bo_hybrid（HB）のデータのみを抽出
    df_hb = df[df['method'] == 'glv_bo_hybrid'].copy()
    
    # ユーザーレベルの指標を計算
    user_metrics = []
    
    for run_id in df_hb['run_id'].unique():
        user_data = df_hb[df_hb['run_id'] == run_id]
        
        user_metrics.append({
            'run_id': run_id,
            'avg_playability': user_data['playability_score'].mean(),
            'max_playability': user_data['playability_score'].max(),
            'min_playability': user_data['playability_score'].min(),
            'total_data_count': len(user_data),
            'avg_novelty': user_data['novelty_score'].mean() if 'novelty_score' in user_data.columns else np.nan,
            'combined_playability': len(user_data) * user_data['playability_score'].mean()  # count × avg
        })
    
    playability_df = pd.DataFrame(user_metrics)
    
    print(f"\n=== Playabilityデータ読み込み ===")
    print(f"総データ数: {len(df)}")
    print(f"HBデータ数: {len(df_hb)}")
    print(f"ユーザー数: {len(playability_df)}")
    
    return playability_df

def load_ux_data():
    """UXアンケートデータ（original.csvとai.csv）を読み込み"""
    original_file = MARIO_ORIGINAL_DIR / 'original.csv'
    ai_file = MARIO_ORIGINAL_DIR / 'ai.csv'
    
    ux_data = []
    
    # original.csvを読み込み
    if original_file.exists():
        df_orig = pd.read_csv(original_file)
        # UIDとMethodの組み合わせでユーザーを識別
        # ただし、run_idとの対応が必要なため、まずは列を確認
        print(f"\n=== Original UXデータ読み込み ===")
        print(f"データ数: {len(df_orig)}")
        print(f"列名: {list(df_orig.columns)}")
        ux_data.append(('original', df_orig))
    
    # ai.csvを読み込み
    if ai_file.exists():
        df_ai = pd.read_csv(ai_file)
        print(f"\n=== AI UXデータ読み込み ===")
        print(f"データ数: {len(df_ai)}")
        print(f"列名: {list(df_ai.columns)}")
        ux_data.append(('ai', df_ai))
    
    return ux_data

def calculate_best_hb_users(playability_df, alpha_patterns_df):
    """HBで最も良い成果を得たユーザーを特定"""
    print("\n=== Best HBユーザーの特定 ===")
    
    # 結合
    merged = playability_df.merge(alpha_patterns_df, on='run_id', how='inner')
    
    if len(merged) == 0:
        print("警告: 結合できるデータがありません")
        return None, None, None
    
    # 上位25%をBest HBグループとする
    playability_threshold = merged['combined_playability'].quantile(0.75)
    best_users = merged[merged['combined_playability'] >= playability_threshold]['run_id'].tolist()
    other_users = merged[merged['combined_playability'] < playability_threshold]['run_id'].tolist()
    
    print(f"Best HBユーザー数: {len(best_users)}")
    print(f"その他のユーザー数: {len(other_users)}")
    print(f"Playability閾値: {playability_threshold:.2f}")
    
    return merged, best_users, other_users

def analyze_correlations(merged_df, alpha_cols, ux_cols):
    """AlphaパターンとUX指標の相関分析"""
    print("\n=== 相関分析 ===")
    
    correlations = []
    
    for alpha_col in alpha_cols:
        if alpha_col not in merged_df.columns:
            continue
        
        for ux_col in ux_cols:
            if ux_col not in merged_df.columns:
                continue
            
            # 欠損値を除外
            valid_data = merged_df[[alpha_col, ux_col]].dropna()
            
            if len(valid_data) < 3:
                continue
            
            # Pearson相関
            pearson_r, pearson_p = pearsonr(valid_data[alpha_col], valid_data[ux_col])
            
            # Spearman相関
            spearman_r, spearman_p = spearmanr(valid_data[alpha_col], valid_data[ux_col])
            
            correlations.append({
                'alpha_metric': alpha_col,
                'ux_metric': ux_col,
                'n': len(valid_data),
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # すべての相関を表示（有意性に関わらず）
    if len(corr_df) > 0:
        print("\nすべての相関係数:")
        for _, row in corr_df.iterrows():
            sig_marker = ""
            if (row['pearson_p'] < 0.05) or (row['spearman_p'] < 0.05):
                sig_marker = " *"
            print(f"  {row['alpha_metric']} ↔ {row['ux_metric']}:")
            print(f"    Pearson r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}{sig_marker}")
            print(f"    Spearman ρ={row['spearman_r']:.3f}, p={row['spearman_p']:.4f}{sig_marker}")
        
        # 有意な相関のみを表示
        significant = corr_df[(corr_df['pearson_p'] < 0.05) | (corr_df['spearman_p'] < 0.05)]
        if len(significant) > 0:
            print(f"\n有意な相関（p < 0.05）: {len(significant)}件")
        else:
            print("\n有意な相関（p < 0.05）は見つかりませんでした")
    
    return corr_df

def compare_groups(merged_df, best_users, other_users, alpha_cols, ux_cols):
    """Best HBユーザーとその他のユーザーを比較"""
    print("\n=== Best HBユーザー vs その他の比較 ===")
    
    comparisons = []
    
    best_data = merged_df[merged_df['run_id'].isin(best_users)]
    other_data = merged_df[merged_df['run_id'].isin(other_users)]
    
    all_cols = alpha_cols + ux_cols
    
    for col in all_cols:
        if col not in merged_df.columns:
            continue
        
        best_values = best_data[col].dropna()
        other_values = other_data[col].dropna()
        
        if len(best_values) < 2 or len(other_values) < 2:
            continue
        
        # Mann-Whitney U検定
        statistic, p_value = mannwhitneyu(best_values, other_values, alternative='two-sided')
        
        comparisons.append({
            'metric': col,
            'group': 'best_hb',
            'n': len(best_values),
            'mean': best_values.mean(),
            'median': best_values.median(),
            'std': best_values.std()
        })
        
        comparisons.append({
            'metric': col,
            'group': 'other',
            'n': len(other_values),
            'mean': other_values.mean(),
            'median': other_values.median(),
            'std': other_values.std(),
            'p_value': p_value,
            'statistic': statistic
        })
    
    comp_df = pd.DataFrame(comparisons)
    
    # 有意な差がある指標を表示
    if len(comp_df) > 0:
        # 各指標についてまとめる
        metrics = comp_df['metric'].unique()
        print("\n有意差がある指標（p < 0.05）:")
        
        for metric in metrics:
            metric_data = comp_df[comp_df['metric'] == metric]
            if len(metric_data) >= 2:
                p_val = metric_data['p_value'].dropna().iloc[0] if not metric_data['p_value'].dropna().empty else None
                
                if p_val and p_val < 0.05:
                    best_row = metric_data[metric_data['group'] == 'best_hb'].iloc[0]
                    other_row = metric_data[metric_data['group'] == 'other'].iloc[0]
                    
                    print(f"\n  {metric}:")
                    print(f"    Best HB: n={best_row['n']}, mean={best_row['mean']:.3f}, median={best_row['median']:.3f}")
                    print(f"    その他: n={other_row['n']}, mean={other_row['mean']:.3f}, median={other_row['median']:.3f}")
                    print(f"    p={p_val:.4f}")
    
    return comp_df

def create_visualizations(merged_df, best_users, other_users, output_dir):
    """可視化を作成"""
    print("\n=== 可視化の作成 ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # グループ列を追加
    merged_df['group'] = merged_df['run_id'].apply(
        lambda x: 'Best HB' if x in best_users else 'Other'
    )
    
    # 1. Alpha変更回数 vs Playabilityの散布図
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1-1. 変更回数 vs Combined Playability
    ax1 = axes[0, 0]
    for group in ['Best HB', 'Other']:
        group_data = merged_df[merged_df['group'] == group]
        ax1.scatter(group_data['total_changes'], group_data['combined_playability'],
                   label=group, alpha=0.6, s=50)
    
    ax1.set_xlabel('Alpha変更回数（main phase）', fontsize=12)
    ax1.set_ylabel('Combined Playability', fontsize=12)
    ax1.set_title('Alpha変更回数とPlayabilityの関係', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 相関係数を追加
    valid_data = merged_df[['total_changes', 'combined_playability']].dropna()
    if len(valid_data) >= 3:
        r, p = pearsonr(valid_data['total_changes'], valid_data['combined_playability'])
        ax1.text(0.05, 0.95, f'r={r:.3f}, p={p:.4f}', transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 1-2. Bold使用率 vs Playability
    ax2 = axes[0, 1]
    for group in ['Best HB', 'Other']:
        group_data = merged_df[merged_df['group'] == group]
        ax2.scatter(group_data['bold_pct'], group_data['combined_playability'],
                   label=group, alpha=0.6, s=50)
    
    ax2.set_xlabel('Bold使用率（%）', fontsize=12)
    ax2.set_ylabel('Combined Playability', fontsize=12)
    ax2.set_title('Bold使用率とPlayabilityの関係', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 1-3. 平均スライダー値 vs Playability
    ax3 = axes[1, 0]
    for group in ['Best HB', 'Other']:
        group_data = merged_df[merged_df['group'] == group].dropna(subset=['avg_slider_value'])
        ax3.scatter(group_data['avg_slider_value'], group_data['combined_playability'],
                   label=group, alpha=0.6, s=50)
    
    ax3.set_xlabel('平均スライダー値（0=cautious, 2=bold）', fontsize=12)
    ax3.set_ylabel('Combined Playability', fontsize=12)
    ax3.set_title('平均Alpha値とPlayabilityの関係', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1-4. 変更回数の分布比較
    ax4 = axes[1, 1]
    best_changes = merged_df[merged_df['group'] == 'Best HB']['total_changes'].dropna()
    other_changes = merged_df[merged_df['group'] == 'Other']['total_changes'].dropna()
    
    if len(best_changes) > 0 and len(other_changes) > 0:
        ax4.hist(best_changes, alpha=0.6, label='Best HB', bins=10, edgecolor='black')
        ax4.hist(other_changes, alpha=0.6, label='Other', bins=10, edgecolor='black')
        ax4.set_xlabel('Alpha変更回数', fontsize=12)
        ax4.set_ylabel('ユーザー数', fontsize=12)
        ax4.set_title('Alpha変更回数の分布比較', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_playability_correlations.pdf', dpi=300, bbox_inches='tight')
    print(f"可視化を保存: {output_dir / 'alpha_playability_correlations.pdf'}")
    plt.close()

def generate_detailed_report(merged_df, best_users, other_users, corr_df, comp_df, output_dir):
    """詳細な分析レポートを生成"""
    print("\n=== 詳細レポート生成 ===")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Alphaパラメーター変更とUX指標の相関分析レポート")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 1. データ概要
    report_lines.append("1. データ概要")
    report_lines.append("-"*80)
    report_lines.append(f"総ユーザー数: {len(merged_df)}")
    report_lines.append(f"Best HBユーザー数: {len(best_users)}")
    report_lines.append(f"その他のユーザー数: {len(other_users)}")
    report_lines.append("")
    
    # 2. Best HBユーザーの特徴
    report_lines.append("2. Best HBユーザーの特徴（上位25%のPlayability）")
    report_lines.append("-"*80)
    best_data = merged_df[merged_df['run_id'].isin(best_users)]
    
    report_lines.append("\n2.1 Alphaパラメーター使用パターン:")
    alpha_metrics = ['total_changes', 'bold_pct', 'cautious_pct', 'standard_pct', 
                     'avg_slider_value', 'style_variability']
    
    for metric in alpha_metrics:
        if metric in best_data.columns:
            values = best_data[metric].dropna()
            if len(values) > 0:
                report_lines.append(f"  {metric}:")
                report_lines.append(f"    Mean = {values.mean():.3f}, Median = {values.median():.3f}")
                report_lines.append(f"    SD = {values.std():.3f}, Range = [{values.min():.3f}, {values.max():.3f}]")
    
    report_lines.append("\n2.2 Playability指標:")
    playability_metrics = ['avg_playability', 'max_playability', 'combined_playability', 
                          'total_data_count', 'avg_novelty']
    
    for metric in playability_metrics:
        if metric in best_data.columns:
            values = best_data[metric].dropna()
            if len(values) > 0:
                report_lines.append(f"  {metric}:")
                report_lines.append(f"    Mean = {values.mean():.3f}, Median = {values.median():.3f}")
                report_lines.append(f"    SD = {values.std():.3f}, Range = [{values.min():.3f}, {values.max():.3f}]")
    
    # 3. 相関分析結果
    report_lines.append("")
    report_lines.append("3. AlphaパラメーターとPlayability指標の相関分析")
    report_lines.append("-"*80)
    
    if len(corr_df) > 0:
        # 有意な相関を強調
        significant = corr_df[(corr_df['pearson_p'] < 0.05) | (corr_df['spearman_p'] < 0.05)]
        
        if len(significant) > 0:
            report_lines.append(f"\n有意な相関（p < 0.05）: {len(significant)}件")
            for _, row in significant.iterrows():
                report_lines.append(f"\n  {row['alpha_metric']} ↔ {row['ux_metric']}:")
                report_lines.append(f"    Pearson r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}")
                report_lines.append(f"    Spearman ρ = {row['spearman_r']:.3f}, p = {row['spearman_p']:.4f}")
                report_lines.append(f"    n = {row['n']}")
        else:
            report_lines.append("\n有意な相関（p < 0.05）は見つかりませんでした")
        
        report_lines.append("\n全ての相関:")
        for _, row in corr_df.iterrows():
            sig_marker = " *" if (row['pearson_p'] < 0.05) or (row['spearman_p'] < 0.05) else ""
            report_lines.append(f"  {row['alpha_metric']} ↔ {row['ux_metric']}:")
            report_lines.append(f"    Pearson r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}{sig_marker}")
            report_lines.append(f"    Spearman ρ = {row['spearman_r']:.3f}, p = {row['spearman_p']:.4f}{sig_marker}")
    
    # 4. グループ比較結果
    report_lines.append("")
    report_lines.append("4. Best HBユーザー vs その他のユーザーの比較")
    report_lines.append("-"*80)
    
    if len(comp_df) > 0:
        # 有意な差がある指標を表示
        significant_comps = []
        metrics_seen = set()
        
        for _, row in comp_df.iterrows():
            if pd.notna(row.get('p_value')) and row['p_value'] < 0.05:
                metric = row['metric']
                if metric not in metrics_seen:
                    metrics_seen.add(metric)
                    significant_comps.append(metric)
        
        if len(significant_comps) > 0:
            report_lines.append(f"\n有意差がある指標（p < 0.05）: {len(significant_comps)}件")
            
            for metric in significant_comps:
                metric_data = comp_df[comp_df['metric'] == metric]
                if len(metric_data) >= 2:
                    best_row = metric_data[metric_data['group'] == 'best_hb'].iloc[0]
                    other_row = metric_data[metric_data['group'] == 'other'].iloc[0]
                    p_val = other_row['p_value']
                    
                    report_lines.append(f"\n  {metric}:")
                    report_lines.append(f"    Best HB: n={best_row['n']}, mean={best_row['mean']:.3f}, median={best_row['median']:.3f}, SD={best_row['std']:.3f}")
                    report_lines.append(f"    その他: n={other_row['n']}, mean={other_row['mean']:.3f}, median={other_row['median']:.3f}, SD={other_row['std']:.3f}")
                    report_lines.append(f"    p-value = {p_val:.4f}")
        else:
            report_lines.append("\n有意差がある指標（p < 0.05）は見つかりませんでした")
    
    # 5. 結論
    report_lines.append("")
    report_lines.append("5. 結論と考察")
    report_lines.append("-"*80)
    report_lines.append("\n主要な知見:")
    
    if len(comp_df) > 0:
        significant_comps = []
        for _, row in comp_df.iterrows():
            if pd.notna(row.get('p_value')) and row['p_value'] < 0.05:
                metric = row['metric']
                if metric not in significant_comps:
                    significant_comps.append(metric)
        
        if len(significant_comps) > 0:
            report_lines.append("\n- Best HBユーザー（上位25%のPlayability）は、以下の指標で有意に高い値を示しました:")
            for metric in significant_comps:
                report_lines.append(f"  * {metric}")
    
    report_lines.append("\n" + "="*80)
    
    # レポートをファイルに保存
    report_text = "\n".join(report_lines)
    report_file = output_dir / 'detailed_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"詳細レポートを保存: {report_file}")
    
    # レポートの内容をコンソールにも表示
    print("\n" + report_text)

def save_results(patterns_df, playability_df, merged_df, corr_df, comp_df, output_dir):
    """結果をCSVに保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patterns_df.to_csv(output_dir / 'alpha_patterns_by_user.csv', index=False, encoding='utf-8')
    playability_df.to_csv(output_dir / 'playability_by_user.csv', index=False, encoding='utf-8')
    merged_df.to_csv(output_dir / 'merged_data.csv', index=False, encoding='utf-8')
    
    if len(corr_df) > 0:
        corr_df.to_csv(output_dir / 'correlations.csv', index=False, encoding='utf-8')
    
    if len(comp_df) > 0:
        comp_df.to_csv(output_dir / 'group_comparisons.csv', index=False, encoding='utf-8')
    
    print(f"\n=== 結果を保存 ===")
    print(f"出力ディレクトリ: {output_dir}")

def main():
    """メイン処理"""
    print("="*80)
    print("Alphaパラメーター変更とUX指標の相関分析")
    print("="*80)
    
    # データ読み込み
    df_main = load_exploration_mode_data()
    patterns_df = calculate_alpha_usage_patterns(df_main)
    playability_df = load_playability_data()
    ux_data = load_ux_data()
    
    # データ結合
    merged_df = playability_df.merge(patterns_df, on='run_id', how='inner')
    
    print(f"\n=== データ結合結果 ===")
    print(f"結合後のユーザー数: {len(merged_df)}")
    
    if len(merged_df) == 0:
        print("警告: 結合できるデータがありません。分析を終了します。")
        return
    
    # Best HBユーザーの特定
    merged_df, best_users, other_users = calculate_best_hb_users(playability_df, patterns_df)
    
    if merged_df is None:
        print("警告: 分析可能なデータがありません。")
        return
    
    # Alpha指標の列名
    alpha_cols = ['total_changes', 'bold_pct', 'cautious_pct', 'standard_pct',
                  'avg_slider_value', 'style_variability']
    
    # UX指標の列名（Playability関連）
    ux_cols = ['avg_playability', 'max_playability', 'combined_playability',
               'avg_novelty', 'total_data_count']
    
    # 相関分析
    corr_df = analyze_correlations(merged_df, alpha_cols, ux_cols)
    
    # グループ比較
    comp_df = compare_groups(merged_df, best_users, other_users, alpha_cols, ux_cols)
    
    # 詳細レポート生成
    output_dir = EXPLORATION_DIR / 'alpha_ux_analysis_results'
    generate_detailed_report(merged_df, best_users, other_users, corr_df, comp_df, output_dir)
    
    # 可視化
    create_visualizations(merged_df, best_users, other_users, output_dir)
    
    # 結果保存
    save_results(patterns_df, playability_df, merged_df, corr_df, comp_df, output_dir)
    
    print("\n" + "="*80)
    print("分析完了！")
    print("="*80)

if __name__ == "__main__":
    main()
