"""
メインスクリプト
designとmarioの両方のデータからプロットを生成します
"""

from pathlib import Path
import test_plot

# データディレクトリ
DESIGN_DIR = Path('design')
MARIO_DIR = Path('mario')

# アウトプットディレクトリ
OUTPUT_DIR = Path('plot')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """メイン処理：designとmarioの両方からプロットを生成"""
    
    print("="*80)
    print("データ分析プロット生成")
    print("="*80)
    
    # designデータからプロットを生成
    if DESIGN_DIR.exists():
        print("\n" + "="*80)
        print("designデータの処理を開始")
        print("="*80)
        test_plot.generate_plots(DESIGN_DIR, OUTPUT_DIR, 'design')
        print("\ndesignデータの処理が完了しました")
    else:
        print(f"\n警告: {DESIGN_DIR} ディレクトリが見つかりません")
    
    # marioデータからプロットを生成
    if MARIO_DIR.exists():
        print("\n" + "="*80)
        print("marioデータの処理を開始")
        print("="*80)
        test_plot.generate_plots(MARIO_DIR, OUTPUT_DIR, 'mario')
        print("\nmarioデータの処理が完了しました")
    else:
        print(f"\n警告: {MARIO_DIR} ディレクトリが見つかりません")
    
    print("\n" + "="*80)
    print("全ての処理が完了しました！")
    print("="*80)

if __name__ == '__main__':
    main()


