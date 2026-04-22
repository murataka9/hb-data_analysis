#!/usr/bin/env python3
"""
Exploration Mode (exploration-mode-slider) の使用統計を分析
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from datetime import datetime
import glob

def load_all_logs(log_dir='/home/takahito/Develop/Hummingbird-Kernel_a/app/results'):
    """すべてのログファイルを読み込む"""
    log_files = glob.glob(os.path.join(log_dir, 'log_*.jsonl'))
    all_events = []
    
    print(f"=== ログファイル読み込み ===")
    print(f"見つかったログファイル数: {len(log_files)}")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        all_events.append(event)
        except Exception as e:
            print(f"警告: {log_file} の読み込みに失敗: {e}")
    
    print(f"総イベント数: {len(all_events)}")
    return all_events

def extract_exploration_events(events):
    """exploration_style_changedイベントと関連するイベントを抽出"""
    exploration_changes = []
    user_actions = []
    
    for event in events:
        if event.get('event') == 'exploration_style_changed':
            exploration_changes.append(event)
        elif event.get('event') == 'user_action':
            details = event.get('details', {})
            if 'exploration_style' in details:
                user_actions.append(event)
    
    print(f"\n=== Exploration Style変更イベント ===")
    print(f"exploration_style_changedイベント数: {len(exploration_changes)}")
    print(f"exploration_styleを含むuser_actionイベント数: {len(user_actions)}")
    
    return exploration_changes, user_actions

def analyze_exploration_statistics(exploration_changes, user_actions):
    """統計分析を実行"""
    print("\n=== Exploration Mode統計分析 ===")
    
    # 1. スタイル変更の基本統計
    style_counts = defaultdict(int)
    method_style_counts = defaultdict(lambda: defaultdict(int))
    phase_style_counts = defaultdict(lambda: defaultdict(int))
    slider_value_counts = defaultdict(int)
    
    # 2. タイムライン分析用
    style_timeline = []
    user_style_usage = defaultdict(lambda: defaultdict(int))
    
    for event in exploration_changes:
        details = event.get('details', {})
        new_style = details.get('new_style', 'unknown')
        previous_style = details.get('previous_style', 'unknown')
        slider_value = details.get('slider_value', -1)
        method = event.get('method', 'unknown')
        phase = details.get('phase', 'unknown')
        run_id = event.get('run_id', 'unknown')
        timestamp = event.get('timestamp', '')
        
        style_counts[new_style] += 1
        method_style_counts[method][new_style] += 1
        phase_style_counts[phase][new_style] += 1
        if slider_value >= 0:
            slider_value_counts[slider_value] += 1
        
        user_style_usage[run_id][new_style] += 1
        
        # タイムライン用
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            style_timeline.append({
                'timestamp': dt,
                'style': new_style,
                'previous_style': previous_style,
                'method': method,
                'phase': phase,
                'slider_value': slider_value,
                'run_id': run_id
            })
        except:
            pass
    
    # 3. 各イベント時のexploration_styleを集計
    style_at_actions = defaultdict(int)
    method_style_at_actions = defaultdict(lambda: defaultdict(int))
    
    for event in user_actions:
        details = event.get('details', {})
        exploration_style = details.get('exploration_style', 'unknown')
        method = event.get('method', 'unknown')
        
        style_at_actions[exploration_style] += 1
        method_style_at_actions[method][exploration_style] += 1
    
    return {
        'style_counts': dict(style_counts),
        'method_style_counts': {k: dict(v) for k, v in method_style_counts.items()},
        'phase_style_counts': {k: dict(v) for k, v in phase_style_counts.items()},
        'slider_value_counts': dict(slider_value_counts),
        'user_style_usage': {k: dict(v) for k, v in user_style_usage.items()},
        'style_timeline': style_timeline,
        'style_at_actions': dict(style_at_actions),
        'method_style_at_actions': {k: dict(v) for k, v in method_style_at_actions.items()}
    }

def print_statistics(stats):
    """統計結果を表示"""
    print("\n" + "="*80)
    print("EXPLORATION MODE 統計結果")
    print("="*80)
    
    # 1. スタイル変更回数
    print("\n1. スタイル変更回数（exploration_style_changedイベント）:")
    print("-" * 40)
    total_changes = sum(stats['style_counts'].values())
    for style, count in sorted(stats['style_counts'].items()):
        percentage = (count / total_changes * 100) if total_changes > 0 else 0
        print(f"  {style:12s}: {count:4d}回 ({percentage:5.1f}%)")
    
    # 2. 手法別スタイル変更
    print("\n2. 手法別スタイル変更回数:")
    print("-" * 40)
    for method in sorted(stats['method_style_counts'].keys()):
        print(f"\n  {method}:")
        method_total = sum(stats['method_style_counts'][method].values())
        for style, count in sorted(stats['method_style_counts'][method].items()):
            percentage = (count / method_total * 100) if method_total > 0 else 0
            print(f"    {style:12s}: {count:4d}回 ({percentage:5.1f}%)")
    
    # 3. フェーズ別スタイル変更
    print("\n3. フェーズ別スタイル変更回数:")
    print("-" * 40)
    for phase in sorted(stats['phase_style_counts'].keys()):
        print(f"\n  {phase}:")
        phase_total = sum(stats['phase_style_counts'][phase].values())
        for style, count in sorted(stats['phase_style_counts'][phase].items()):
            percentage = (count / phase_total * 100) if phase_total > 0 else 0
            print(f"    {style:12s}: {count:4d}回 ({percentage:5.1f}%)")
    
    # 4. スライダー値の分布
    print("\n4. スライダー値の分布:")
    print("-" * 40)
    slider_labels = {0: 'cautious', 1: 'standard', 2: 'bold'}
    total_slider = sum(stats['slider_value_counts'].values())
    for value in sorted(stats['slider_value_counts'].keys()):
        count = stats['slider_value_counts'][value]
        label = slider_labels.get(value, f'unknown({value})')
        percentage = (count / total_slider * 100) if total_slider > 0 else 0
        print(f"  {value} ({label:12s}): {count:4d}回 ({percentage:5.1f}%)")
    
    # 5. ユーザーアクション時のスタイル分布
    print("\n5. ユーザーアクション時のExploration Style分布:")
    print("-" * 40)
    total_actions = sum(stats['style_at_actions'].values())
    for style, count in sorted(stats['style_at_actions'].items()):
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  {style:12s}: {count:4d}回 ({percentage:5.1f}%)")
    
    # 6. 手法別アクション時のスタイル分布
    print("\n6. 手法別アクション時のExploration Style分布:")
    print("-" * 40)
    for method in sorted(stats['method_style_at_actions'].keys()):
        print(f"\n  {method}:")
        method_total = sum(stats['method_style_at_actions'][method].values())
        for style, count in sorted(stats['method_style_at_actions'][method].items()):
            percentage = (count / method_total * 100) if method_total > 0 else 0
            print(f"    {style:12s}: {count:4d}回 ({percentage:5.1f}%)")
    
    # 7. ユーザーごとの使用パターン
    print("\n7. ユーザーごとの使用パターン（上位10ユーザー）:")
    print("-" * 40)
    user_totals = {uid: sum(styles.values()) for uid, styles in stats['user_style_usage'].items()}
    sorted_users = sorted(user_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for uid, total in sorted_users:
        print(f"\n  User {uid} (合計{total}回):")
        for style, count in sorted(stats['user_style_usage'][uid].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {style:12s}: {count:4d}回 ({percentage:5.1f}%)")

def create_visualizations(stats):
    """可視化を作成"""
    print("\n=== 可視化の作成 ===")
    
    # 1. スタイル変更回数の棒グラフ
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1-1. 全体のスタイル変更回数
    ax1 = axes[0, 0]
    styles = list(stats['style_counts'].keys())
    counts = [stats['style_counts'][s] for s in styles]
    colors = {'cautious': '#FF6B6B', 'standard': '#4ECDC4', 'bold': '#FFE66D'}
    bar_colors = [colors.get(s, '#95A5A6') for s in styles]
    bars = ax1.bar(styles, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Exploration Style変更回数（全体）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('変更回数', fontsize=12)
    ax1.set_xlabel('スタイル', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 1-2. 手法別スタイル変更
    ax2 = axes[0, 1]
    methods = list(stats['method_style_counts'].keys())
    if methods:
        x = np.arange(len(methods))
        width = 0.25
        style_list = ['cautious', 'standard', 'bold']
        
        for i, style in enumerate(style_list):
            counts = [stats['method_style_counts'].get(m, {}).get(style, 0) for m in methods]
            ax2.bar(x + i*width, counts, width, label=style, 
                   color=colors.get(style, '#95A5A6'), alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_title('手法別Exploration Style変更回数', fontsize=14, fontweight='bold')
        ax2.set_ylabel('変更回数', fontsize=12)
        ax2.set_xlabel('手法', fontsize=12)
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 1-3. フェーズ別スタイル変更
    ax3 = axes[1, 0]
    phases = list(stats['phase_style_counts'].keys())
    if phases:
        x = np.arange(len(phases))
        width = 0.25
        
        for i, style in enumerate(style_list):
            counts = [stats['phase_style_counts'].get(p, {}).get(style, 0) for p in phases]
            ax3.bar(x + i*width, counts, width, label=style,
                   color=colors.get(style, '#95A5A6'), alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_title('フェーズ別Exploration Style変更回数', fontsize=14, fontweight='bold')
        ax3.set_ylabel('変更回数', fontsize=12)
        ax3.set_xlabel('フェーズ', fontsize=12)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(phases)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 1-4. ユーザーアクション時のスタイル分布
    ax4 = axes[1, 1]
    styles = list(stats['style_at_actions'].keys())
    counts = [stats['style_at_actions'][s] for s in styles]
    bar_colors = [colors.get(s, '#95A5A6') for s in styles]
    bars = ax4.bar(styles, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_title('ユーザーアクション時のExploration Style分布', fontsize=14, fontweight='bold')
    ax4.set_ylabel('アクション回数', fontsize=12)
    ax4.set_xlabel('スタイル', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/takahito/Develop/Hummingbird-Kernel_a/exploration_mode_statistics.pdf',
                dpi=300, bbox_inches='tight', format='pdf')
    print("可視化を保存しました: exploration_mode_statistics.pdf")
    plt.show()

def export_to_csv(stats):
    """統計データをCSVにエクスポート"""
    print("\n=== CSVエクスポート ===")
    
    # 1. スタイル変更の詳細データ
    timeline_data = []
    for item in stats['style_timeline']:
        timeline_data.append({
            'timestamp': item['timestamp'].isoformat(),
            'style': item['style'],
            'previous_style': item['previous_style'],
            'method': item['method'],
            'phase': item['phase'],
            'slider_value': item['slider_value'],
            'run_id': item['run_id']
        })
    
    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        df_timeline.to_csv('/home/takahito/Develop/Hummingbird-Kernel_a/exploration_mode_timeline.csv',
                          index=False, encoding='utf-8')
        print("タイムラインデータを保存しました: exploration_mode_timeline.csv")
    
    # 2. 集計統計データ
    summary_data = []
    
    # 全体統計
    for style, count in stats['style_counts'].items():
        summary_data.append({
            'category': '全体',
            'subcategory': 'style_change',
            'item': style,
            'count': count
        })
    
    # 手法別統計
    for method, styles in stats['method_style_counts'].items():
        for style, count in styles.items():
            summary_data.append({
                'category': '手法別',
                'subcategory': method,
                'item': style,
                'count': count
            })
    
    # フェーズ別統計
    for phase, styles in stats['phase_style_counts'].items():
        for style, count in styles.items():
            summary_data.append({
                'category': 'フェーズ別',
                'subcategory': phase,
                'item': style,
                'count': count
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv('/home/takahito/Develop/Hummingbird-Kernel_a/exploration_mode_summary.csv',
                         index=False, encoding='utf-8')
        print("集計統計データを保存しました: exploration_mode_summary.csv")
    
    # 3. ユーザー別統計
    user_data = []
    for run_id, styles in stats['user_style_usage'].items():
        total = sum(styles.values())
        for style, count in styles.items():
            user_data.append({
                'run_id': run_id,
                'style': style,
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            })
    
    if user_data:
        df_users = pd.DataFrame(user_data)
        df_users.to_csv('/home/takahito/Develop/Hummingbird-Kernel_a/exploration_mode_by_user.csv',
                       index=False, encoding='utf-8')
        print("ユーザー別統計データを保存しました: exploration_mode_by_user.csv")

def main():
    """メイン処理"""
    print("Exploration Mode統計分析を開始...")
    
    # ログファイルを読み込み
    events = load_all_logs()
    
    # Exploration関連イベントを抽出
    exploration_changes, user_actions = extract_exploration_events(events)
    
    # 統計分析
    stats = analyze_exploration_statistics(exploration_changes, user_actions)
    
    # 結果を表示
    print_statistics(stats)
    
    # 可視化
    if exploration_changes or user_actions:
        create_visualizations(stats)
        export_to_csv(stats)
    
    print("\n分析完了!")
    print("生成されたファイル:")
    print("- exploration_mode_statistics.pdf")
    print("- exploration_mode_timeline.csv")
    print("- exploration_mode_summary.csv")
    print("- exploration_mode_by_user.csv")

if __name__ == "__main__":
    main()

