import os
import subprocess
import sys
from pathlib import Path

def run_command(command, cwd=None):
    """コマンドを実行し、結果を出力する"""
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def main():
    root_dir = Path(__file__).parent.parent.absolute()
    os.chdir(root_dir)
    print(f"Setting up project at: {root_dir}")

    # 1. 必要なディレクトリの作成
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/loader",
        "src/preprocessor",
        "src/analysis",
        "src/visualization",
        "tests"
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
        # .gitkeep の作成
        gitkeep = Path(d) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print("[OK] Directory structure verified.")

    # 2. 仮想環境の作成
    venv_dir = root_dir / "venv"
    if not venv_dir.exists():
        print("Creating virtual environment...")
        run_command(f"{sys.executable} -m venv venv")
    else:
        print("[OK] Virtual environment already exists.")

    # 3. 依存ライブラリのインストール
    print("Installing dependencies...")
    pip_path = venv_dir / "Scripts" / "pip" if os.name == "nt" else venv_dir / "bin" / "pip"
    if run_command(f"{pip_path} install -r requirements.txt"):
        print("[OK] Dependencies installed.")
    else:
        print("[ERR] Failed to install dependencies.")

    # 4. .env ファイルの作成 (テンプレート)
    env_file = root_dir / ".env"
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("OPENAI_API_KEY=your_key_here\n")
        print("[OK] Created .env template.")

    print("\n" + "="*40)
    print(" Setup Completed Successfully! ")
    print("="*40)
    print("\nTo start analysis:")
    if os.name == "nt":
        print("1. .\\venv\\Scripts\\activate")
    else:
        print("1. source venv/bin/activate")
    print("2. jupyter lab")
    print("="*40)

if __name__ == "__main__":
    main()