#!/usr/bin/env python3
"""
last surveyのq1_best_methodでglv-boを選んだ人とそれ以外のユーザーの比較をviolin plotで可視化
style_config.pyのスタイルを使用、英語表記、ダイナマイトプロット対応
フォントサイズをさらに5ポイント大きく（合計12ポイント）
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
MARIO_ORIGINAL_DIR = PROJECT_ROOT / 'mario' / 'original'
QDA_SURVEY_FILE = PROJECT_ROOT / 'datas' / 'QDA-last-survey.csv'

# vistatsのインポート確認
try:
    from vistats import annotate_brackets
    HAS_VISTATS = True
except ImportError:
    HAS_VISTATS = False
    print("Warning: vistats not available, using manual bracket drawing")

def create_uid_runid_mapping():
    """UIDとrun_idの対応関係を作成"""
    # original.csvとexploration_mode_timeline.csvを使って対応関係を作成
    original_file = MARIO_ORIGINAL_DIR / 'original.csv'
    timeline_file = PROJECT_ROOT / 'exploration_mode_statistics' / 'exploration_mode_timeline.csv'
    
    if not original_file.exists() or not timeline_file.exists():
        print("Warning: UID-run_id mapping作成に必要なファイルが見つかりません")
        return None
    
    original_df = pd.read_csv(original_file)
    timeline_df = pd.read_csv(timeline_file)
    
    # glv_bo_hybridのデータのみを使用
    original_glv = original_df[original_df['Method'] == 'glv_bo_hybrid'].copy()
    timeline_glv = timeline_df[timeline_df['method'] == 'glv_bo_hybrid'].copy()
    
    # Timestampで対応付けを試みる（簡易版：最初のrun_idを使用）
    # より正確には、original.csvの各UIDのglv_bo_hybridの最初のTimestampと
    # timelineの各run_idの最初のtimestampを比較
    mapping = {}
    
    # 各UIDの最初のglv_bo_hybridのTimestampを取得
    for uid in original_glv['UID'].unique():
        uid_data = original_glv[original_glv['UID'] == uid].sort_values('Timestamp')
        if len(uid_data) > 0:
            first_timestamp = pd.to_datetime(uid_data['Timestamp'].iloc[0])
            
            # timelineから最も近いtimestampのrun_idを探す
            timeline_glv['timestamp_dt'] = pd.to_datetime(timeline_glv['timestamp'])
            timeline_glv['time_diff'] = abs(timeline_glv['timestamp_dt'] - first_timestamp)
            
            # 最も近いrun_idを取得（1時間以内）
            closest = timeline_glv[timeline_glv['time_diff'] < pd.Timedelta(hours=1)]
            if len(closest) > 0:
                closest_run_id = closest.sort_values('time_diff').iloc[0]['run_id']
                mapping[uid] = closest_run_id
    
    print(f"\n=== UID-run_id Mapping作成 ===")
    print(f"マッピング数: {len(mapping)}")
    print(f"マッピング例: {dict(list(mapping.items())[:3])}")
    
    return mapping

def load_data_with_survey():
    """データを読み込み、q1_best_method情報を追加"""
    merged_file = OUTPUT_DIR / 'merged_data.csv'
    if not merged_file.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {merged_file}")
    
    df = pd.read_csv(merged_file)
    
    # last surveyのq1_best_methodデータを読み込み
    if not QDA_SURVEY_FILE.exists():
        print(f"Warning: {QDA_SURVEY_FILE} が見つかりません。既存のgroup列を使用します。")
        return df, None
    
    survey_df = pd.read_csv(QDA_SURVEY_FILE)
    print(f"\n=== QDA Last Surveyデータ読み込み ===")
    print(f"ファイル: {QDA_SURVEY_FILE.name}")
    print(f"データ数: {len(survey_df)}")
    print(f"列名: {list(survey_df.columns)}")
    
    # q1_best_method列を確認
    if 'q1_best_method' not in survey_df.columns:
        print("Warning: q1_best_method列が見つかりません。既存のgroup列を使用します。")
        return df, None
    
    # UIDとq1_best_methodを抽出
    survey_data = survey_df[['UID', 'q1_best_method']].copy()
    print(f"\nq1_best_methodの値:")
    print(survey_data['q1_best_method'].value_counts())
    
    # UIDとrun_idの対応関係を作成
    uid_runid_mapping = create_uid_runid_mapping()
    
    if uid_runid_mapping:
        # mappingを使ってrun_idを追加
        survey_data['run_id'] = survey_data['UID'].map(uid_runid_mapping)
        survey_data = survey_data[survey_data['run_id'].notna()].copy()
        
        # run_idでマージ
        df_with_survey = df.merge(survey_data[['run_id', 'q1_best_method']], on='run_id', how='left', suffixes=('', '_survey'))
        print(f"\nマージ後のデータ数: {len(df_with_survey)}")
        print(f"q1_best_methodがマージできた数: {df_with_survey['q1_best_method'].notna().sum()}")
    else:
        print("Warning: UID-run_id mappingが作成できませんでした。既存のgroup列を使用します。")
        df_with_survey = df.copy()
    
    return df_with_survey, 'q1_best_method'

def identify_glv_bo_users(df, q1_col=None):
    """q1_best_methodでglv-boを選んだユーザーを識別"""
    if q1_col is None:
        print("Warning: q1_best_method列が見つかりません。group列を使用します。")
        if 'group' in df.columns:
            glv_bo_users = df[df['group'] == 'Best HB']['run_id'].unique().tolist()
            other_users = df[df['group'] == 'Other']['run_id'].unique().tolist()
            return glv_bo_users, other_users
        else:
            raise ValueError("q1_best_method列もgroup列も見つかりません")
    
    # q1_best_methodでglv-boを選んだユーザーを識別
    # glv-boの表記揺れに対応: glv_bo_hybrid, glv-bo, GLV-BO, etc.
    glv_bo_patterns = ['glv_bo_hybrid', 'glv-bo', 'glv_bo', 'GLV-BO', 'GLV_BO', 'HB', 'glvbo']
    
    if q1_col in df.columns:
        # glv-boを選んだユーザーのrun_idを取得
        glv_bo_mask = df[q1_col].str.lower().isin([p.lower() for p in glv_bo_patterns])
        glv_bo_users = df[glv_bo_mask]['run_id'].dropna().unique().tolist()
        other_users = df[~glv_bo_mask & df['run_id'].notna()]['run_id'].dropna().unique().tolist()
    else:
        raise ValueError(f"列 {q1_col} が見つかりません")
    
    print(f"\n=== GLV-BO選択ユーザーの識別 ===")
    print(f"GLV-BOを選んだユーザー数: {len(glv_bo_users)}")
    print(f"その他のユーザー数: {len(other_users)}")
    print(f"GLV-BOを選んだrun_ids: {glv_bo_users[:5]}...")  # 最初の5つを表示
    
    return glv_bo_users, other_users

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

def perform_statistical_tests(df, glv_bo_users, other_users):
    """統計検定を実行して結果を返す"""
    # グループを識別
    df['survey_group'] = df['run_id'].apply(
        lambda x: 'HB Selected' if x in glv_bo_users else ('Other' if x in other_users else None)
    )
    df = df[df['survey_group'].notna()].copy()
    
    glv_bo_group = df[df['survey_group'] == 'HB Selected'].copy()
    other_group = df[df['survey_group'] == 'Other'].copy()
    
    # 分析する指標（Usage, Playability, Noveltyに分類）
    metrics = {
        # Usage
        'standard_pct': 'Standard Usage (%)',
        'bold_pct': 'Bold Usage (%)',
        'cautious_pct': 'Cautious Usage (%)',
        'total_changes': 'Alpha Changes Count',
        # Playability
        'max_playability': 'Max Playability',
        'combined_playability': 'Combined Playability',
        'total_data_count': 'Total Data Count',
        # Novelty
        'avg_novelty': 'Average Novelty'
    }
    
    results = {}
    
    for metric_key, metric_label in metrics.items():
        glv_bo_values = glv_bo_group[metric_key].dropna()
        other_values = other_group[metric_key].dropna()
        
        if len(glv_bo_values) < 2 or len(other_values) < 2:
            continue
        
        # Mann-Whitney U検定
        statistic, p_value = mannwhitneyu(glv_bo_values, other_values, alternative='two-sided')
        
        # 効果量を計算
        effect_r = calculate_effect_size(glv_bo_values, other_values, statistic)
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
            'glv_bo_values': glv_bo_values,
            'other_values': other_values,
            'glv_bo_mean': glv_bo_values.mean(),
            'glv_bo_median': glv_bo_values.median(),
            'glv_bo_std': glv_bo_values.std(),
            'other_mean': other_values.mean(),
            'other_median': other_values.median(),
            'other_std': other_values.std(),
            'n_glv_bo': len(glv_bo_values),
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
    """Violin plotsを作成（Usage, Playability, Noveltyの3グループを別々の図に分けて）"""
    # スタイル設定
    style_config.setup_seaborn_style()
    
    # フォントサイズをさらに5ポイント大きく設定（合計12ポイント = 7+5）
    plt.rcParams['font.size'] = 15 + 12  # 27
    plt.rcParams['axes.labelsize'] = 16 + 12  # 28
    plt.rcParams['axes.titlesize'] = 16 + 12  # 28
    plt.rcParams['xtick.labelsize'] = 15 + 12  # 27
    plt.rcParams['ytick.labelsize'] = 15 + 12  # 27
    plt.rcParams['legend.fontsize'] = 15 + 12  # 27
    
    # 指標を3つのグループに分類
    usage_metrics = ['standard_pct', 'bold_pct', 'cautious_pct', 'total_changes']
    playability_metrics = ['max_playability', 'combined_playability', 'total_data_count']
    novelty_metrics = ['avg_novelty']
    
    # 各グループで結果があるものだけをフィルタリング
    usage_order = [m for m in usage_metrics if m in results]
    playability_order = [m for m in playability_metrics if m in results]
    novelty_order = [m for m in novelty_metrics if m in results]
    
    # 各グループの指標リスト
    groups = [
        ('Usage', usage_order),
        ('Playability', playability_order),
        ('Novelty', novelty_order)
    ]
    
    # グループの色（青系2色）
    group_colors = {
        'HB Selected': '#4A90E2',  # 青系1
        'Other': '#87CEEB'     # 青系2（薄めの青）
    }
    
    # 各グループごとに別々の図を作成
    for group_name, metric_order in groups:
        if len(metric_order) == 0:
            continue
        
        # Playability系の場合は、各指標を別々の図に分ける
        if group_name == 'Playability':
            # 各指標ごとに別々の図を作成
            for metric_key in metric_order:
                # 1つの指標だけのリストを作成
                single_metric_order = [metric_key]
                _create_single_group_plot(group_name, single_metric_order, results, group_colors, output_dir)
        else:
            # その他のグループは1つの図にまとめる
            _create_single_group_plot(group_name, metric_order, results, group_colors, output_dir)

def _create_single_group_plot(group_name, metric_order, results, group_colors, output_dir):
    """1つのグループ（または1つの指標）のviolin plotを作成"""
    # 各指標ごとに新しい図を作成
    fig, ax = plt.subplots(1, 1, figsize=(len(metric_order) * 3, 7))
    
    n_metrics_in_group = len(metric_order)
    
    # 全データを準備（各指標×各グループ）
    all_plot_data = []
    x_positions = []
    x_labels = []
    bracket_info = []  # 各指標のブラケット情報を保存
    
    for metric_idx, metric_key in enumerate(metric_order):
        result = results[metric_key]
        
        # 各指標の位置を計算（グループ内での位置）
        # 各指標ごとに2つのviolin（HB SelectedとOther）を配置
        base_pos = metric_idx * 3  # 各指標の間隔（3単位）
        pos_glv_bo = base_pos
        pos_other = base_pos + 1
        
        # データを追加
        for val in result['glv_bo_values']:
            all_plot_data.append({
                'Metric': result['label'],
                'Group': 'HB Selected',
                'Value': val,
                'x_pos': pos_glv_bo
            })
        for val in result['other_values']:
            all_plot_data.append({
                'Metric': result['label'],
                'Group': 'Other',
                'Value': val,
                'x_pos': pos_other
            })
        
        x_positions.extend([pos_glv_bo, pos_other])
        x_labels.extend([result['label'] + '\nHB Selected', result['label'] + '\nOther'])
        
        # ブラケット情報を保存
        if result['significance'] != 'ns':
            bracket_info.append({
                'metric_idx': metric_idx,
                'pos_glv_bo': pos_glv_bo,
                'pos_other': pos_other,
                'glv_bo_max': result['glv_bo_values'].max(),
                'other_max': result['other_values'].max(),
                'significance': result['significance']
            })
    
    plot_df = pd.DataFrame(all_plot_data)
    
    # seabornのviolinplotを使うために、x軸に指標名とグループ名の組み合わせを作成
    # 各指標ごとに2つのviolin（HB SelectedとOther）を横に並べる
    plot_data_for_seaborn = []
    x_order = []
    
    for metric_idx, metric_key in enumerate(metric_order):
        result = results[metric_key]
        metric_label = result['label']
        
        # HB Selected
        glv_bo_data = plot_df[(plot_df['Metric'] == metric_label) & 
                              (plot_df['Group'] == 'HB Selected')]['Value'].values
        for val in glv_bo_data:
            plot_data_for_seaborn.append({
                'x_label': metric_label,
                'Group': 'HB Selected',
                'Value': val
            })
        
        # Other
        other_data = plot_df[(plot_df['Metric'] == metric_label) & 
                            (plot_df['Group'] == 'Other')]['Value'].values
        for val in other_data:
            plot_data_for_seaborn.append({
                'x_label': metric_label,
                'Group': 'Other',
                'Value': val
            })
        
        x_order.append(metric_label)
    
    plot_seaborn_df = pd.DataFrame(plot_data_for_seaborn)
    
    # seabornのviolinplotを使用
    color_config = style_config.VIOLIN_COLOR_CONFIG
    violin_config = {
        'width': 0.7,
        'cut': 0.6,
        'bw': 0.5,
    }
    
    # hueパラメータを使ってグループを分ける
    violin = sns.violinplot(
        data=plot_seaborn_df,
        x='x_label',
        y='Value',
        hue='Group',
        hue_order=['HB Selected', 'Other'],
        order=x_order,
        palette=[group_colors['HB Selected'], group_colors['Other']],
        inner='box',
        scale='width',
        ax=ax,
        **violin_config
    )
    
    # violinの外枠を削除（style_configの設定に従う）
    for i, pc in enumerate(violin.collections):
        pc.set_edgecolor(color_config['edgecolor'])
        pc.set_alpha(color_config['alpha'])
        
        # 片方（Other側）に白のハッチングパターンを追加
        # seabornのviolinplotでは、hueごとに順番にcollectionsが作成される
        # 各指標ごとに2つのviolin（HB SelectedとOther）が作成される
        # インデックスが奇数の場合がOther側
        if i % 2 == 1:  # Other側
            pc.set_hatch('///')
            pc.set_edgecolor('white')
            pc.set_linewidth(color_config['hatch_linewidth'])
    
    # グループ名をタイトルとして追加
    ax.set_title(group_name, fontsize=28, fontweight='bold', pad=10, color='black')
    
    # X軸の設定
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # X軸の目盛りを設定（seabornが自動設定したものを使用）
    ax.tick_params(axis='x', which='major', length=5, width=1, bottom=True, labelsize=27)
    ax.tick_params(axis='y', which='major', labelsize=27)
    
    # X軸のラベルを黒色に設定
    for label in ax.get_xticklabels():
        label.set_color('black')
    
    # グリッド設定
    ax.grid(True, alpha=style_config.GROUPED_PLOT_CONFIG['grid_alpha'],
            axis=style_config.GROUPED_PLOT_CONFIG['grid_axis'])
    
    # 左右の縦線を削除
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 凡例を削除（タイトルでグループが分かるため）
    ax.legend_.remove()
    
    # 統計的有意性をブラケットで表示
    bracket_config = style_config.BRACKET_CONFIG
    bracket_fontsize = bracket_config['fontsize'] + 12
    
    # seabornのviolinplotでは、x軸の位置が自動的に決まる
    # 各指標の位置を取得
    xticks = ax.get_xticks()
    
    for bracket in bracket_info:
        if bracket['significance'] != 'ns':
            # Y軸の上限を拡張（最初のブラケットのみ）
            if bracket == bracket_info[0]:
                current_ylim = ax.get_ylim()
                y_range = current_ylim[1] - current_ylim[0]
                estimated_space = bracket_config['estimated_space_ratio'] * y_range
                new_y_max = current_ylim[1] + estimated_space
                ax.set_ylim(current_ylim[0], new_y_max)
            
            # ブラケットを描画
            metric_idx = bracket['metric_idx']
            if metric_idx < len(xticks):
                base_pos = xticks[metric_idx]
                # hueを使う場合、各指標内で2つのviolinが横に並ぶ
                # デフォルトでは、各指標の中央がxticks[i]で、hueで-0.2と+0.2の位置に配置される
                center = [base_pos - 0.2, base_pos + 0.2]  # HB Selected, Other
                height = [bracket['glv_bo_max'], bracket['other_max']]
                yerr = [0, 0]
                tuples = [(0, 1, bracket['significance'])]
                
                if HAS_VISTATS:
                    try:
                        annotate_brackets(
                            tuples, center, height, yerr,
                            ax=ax,
                            fs=bracket_fontsize
                        )
                    except Exception as e:
                        _draw_manual_brackets(ax, tuples, center, height, bracket_config, bracket_fontsize)
                else:
                    _draw_manual_brackets(ax, tuples, center, height, bracket_config, bracket_fontsize)
    
    plt.tight_layout()
    
    # ファイル名を決定（Playability系の場合は各指標ごとに別々のファイル名）
    if group_name == 'Playability' and n_metrics_in_group == 1:
        # Playability系の1つの指標の場合、指標名を含める
        metric_key = metric_order[0]
        metric_label = results[metric_key]['label'].lower().replace(' ', '_')
        output_file = output_dir / f'glv_bo_survey_comparison_violin_{group_name.lower()}_{metric_label}.pdf'
    else:
        # その他の場合はグループ名のみ
        output_file = output_dir / f'glv_bo_survey_comparison_violin_{group_name.lower()}.pdf'
    
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
    report_lines.append("Statistical Test Results: HB Selected vs Other Users")
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
        report_lines.append(f"  HB Selected:")
        report_lines.append(f"    n = {result['n_glv_bo']}")
        report_lines.append(f"    Mean = {result['glv_bo_mean']:.3f}")
        report_lines.append(f"    Median = {result['glv_bo_median']:.3f}")
        report_lines.append(f"    SD = {result['glv_bo_std']:.3f}")
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
    report_file = output_dir / 'glv_bo_survey_statistical_test_details.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n統計検定詳細レポートを保存: {report_file}")
    print("\n" + report_text)

def main():
    """メイン処理"""
    print("="*80)
    print("GLV-BO Survey選択ユーザー比較 Violin Plot作成")
    print("="*80)
    
    # データ読み込み（survey情報を含む）
    try:
        df, q1_col = load_data_with_survey()
    except Exception as e:
        print(f"Error loading survey data: {e}")
        # フォールバック：既存のmerged_dataを使用
        merged_file = OUTPUT_DIR / 'merged_data.csv'
        df = pd.read_csv(merged_file)
        q1_col = None
    
    # GLV-BOを選んだユーザーを識別
    glv_bo_users, other_users = identify_glv_bo_users(df, q1_col)
    
    # 統計検定実行
    results = perform_statistical_tests(df, glv_bo_users, other_users)
    
    # Violin plot作成
    create_violin_plots(df, results, OUTPUT_DIR)
    
    # 統計検定詳細レポート生成
    generate_statistical_report(results, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("完了！")
    print("="*80)

if __name__ == "__main__":
    main()
