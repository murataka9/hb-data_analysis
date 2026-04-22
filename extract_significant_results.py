#!/usr/bin/env python3
"""
統計結果ログファイルから有意差がある結果を抽出するスクリプト
"""

import re
from pathlib import Path
from typing import List, Tuple


def extract_significant_results(log_file_path: Path) -> List[Tuple[str, str]]:
    """
    logファイルから有意差がある結果を抽出します
    
    Parameters:
    -----------
    log_file_path : Path
        logファイルのパス
    
    Returns:
    --------
    List[Tuple[str, str]]
        (質問タイトル, 有意差がある行)のリスト
    """
    results = []
    current_question = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 質問タイトルを抽出（"===="で囲まれた行の次の行、かつ"(Friedman test)"や"(Wilcoxon signed-rank test)"が含まれていない行）
        if line.startswith('=') and len(line) > 10:
            # 次の行を確認
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # "(Friedman test)"や"(Wilcoxon signed-rank test)"が含まれていない場合、質問タイトルとして扱う
                if (next_line and 
                    not next_line.startswith('=') and 
                    '(Friedman test)' not in next_line and 
                    '(Wilcoxon signed-rank test)' not in next_line and
                    'Friedman test:' not in next_line and
                    '条件数が' not in next_line):
                    current_question = next_line
        
        # 有意差がある行を抽出（*で始まる行）
        if line.startswith('*') and current_question:
            # *記号を除去して行を取得
            clean_line = line.lstrip('*').strip()
            results.append((current_question, clean_line))
    
    return results


def format_output(results: List[Tuple[str, str]], output_file: Path = None):
    """
    抽出した結果を整形して出力します
    
    Parameters:
    -----------
    results : List[Tuple[str, str]]
        抽出した結果のリスト
    output_file : Path, optional
        出力ファイルのパス（Noneの場合は標準出力）
    """
    if output_file:
        f = open(output_file, 'w', encoding='utf-8')
    else:
        f = None
    
    try:
        if not results:
            message = "有意差がある結果は見つかりませんでした。"
            print(message)
            if f:
                f.write(message + '\n')
            return
        
        # 質問タイトルごとにグループ化
        question_groups = {}
        for question, line in results:
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(line)
        
        # 出力
        header = f"有意差がある結果（全{len(results)}件）\n"
        header += "=" * 80 + "\n\n"
        print(header, end='')
        if f:
            f.write(header)
        
        for question, lines in question_groups.items():
            section = f"【{question}】\n"
            section += "-" * 80 + "\n"
            for line in lines:
                section += f"  {line}\n"
            section += "\n"
            print(section, end='')
            if f:
                f.write(section)
        
    finally:
        if f:
            f.close()


def main():
    """メイン処理"""
    import sys
    
    # コマンドライン引数からlogファイルのパスを取得
    if len(sys.argv) > 1:
        log_file_path = Path(sys.argv[1])
    else:
        # デフォルトのパス
        script_dir = Path(__file__).parent
        log_file_path = script_dir / 'plot' / 'statistical_results_design.log'
    
    if not log_file_path.exists():
        print(f"エラー: ファイルが見つかりません: {log_file_path}")
        sys.exit(1)
    
    print(f"ログファイルを読み込み中: {log_file_path}")
    
    # 有意差がある結果を抽出
    results = extract_significant_results(log_file_path)
    
    # 出力ファイルのパスを決定
    output_file = log_file_path.parent / f"{log_file_path.stem}_significant.txt"
    
    # 結果を出力
    format_output(results, output_file)
    
    print(f"\n結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()



