# Art Style Classification

画風分類のための深層学習モデル。EfficientNet B7をベースに、8種類の画風を分類します。

## サポートする画風

1. **anime** - アニメ塗り
2. **brush** - ブラシ塗り
3. **thick** - 厚塗り
4. **watercolor** - 水彩
5. **photo** - 写真
6. **3dcg** - 3DCG
7. **comic** - 白黒マンガ
8. **pixelart** - ピクセルアート

## 機能

- **EfficientNet B7**ベースの高精度な画風分類
- **MSA-Net** (Multimodal Style Aggregation Network) のオプション統合
- **仮想データセット分割**でディスク容量を節約
- クラス不均衡データセットに対する自動重み付け
- TensorBoardによる学習過程の可視化
- スタイル空間の2D可視化
- 単一画像・バッチ推論のサポート

## 環境構築

### 前提条件

- Windows 11 with WSL2
- CUDA 12.8
- Python 3.10+

### WSL2のセットアップ

```bash
# WSL2のインストール (PowerShellで管理者権限)
wsl --install

# Ubuntuを起動
wsl
```

### CUDA 12.8のインストール

```bash
# CUDA Toolkitのインストール
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# 環境変数の設定
echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# CUDAバージョン確認
nvcc --version
```

### Pythonとパッケージのインストール

```bash
# Pythonのインストール
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip

# 仮想環境の作成
python3.10 -m venv venv
source venv/bin/activate

# PyTorch (CUDA 12.1サポート) のインストール
# Note: CUDA 12.8はCUDA 12.1のバイナリと互換性があります
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# その他の依存パッケージのインストール
pip install -r requirements.txt

# CUDAが正しく認識されているか確認
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

## データセットの準備

### 方法1: 仮想分割（推奨）

データセットをクラスごとに配置するだけで、**メモリ内で**自動的にtrain/valに分割されます:

```
dataset/
├── anime/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── brush/
├── thick/
├── watercolor/
├── photo/
├── 3dcg/
├── comic/
└── pixelart/
```

学習スクリプトが自動的に80%をtrain、20%をvalidationに**仮想分割**します（比率は`--val_split`オプションで変更可能）。

**メリット:**
- **物理的なフォルダー構造を変更しない**（train/valディレクトリを作成しない）
- **ディスク容量を節約**（ファイルのコピーやシンボリックリンク不要）
- 手動でtrain/valを分ける必要がない
- 層化分割により各クラスの比率を保持
- 再現性のある分割（`--random_seed`で固定）

### 方法2: 手動分割

既にtrain/valに分割済みの場合も使用できます:

```
dataset/
├── train/
│   ├── anime/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── brush/
│   ├── thick/
│   ├── watercolor/
│   ├── photo/
│   ├── 3dcg/
│   ├── comic/
│   └── pixelart/
└── val/
    ├── anime/
    ├── brush/
    ├── thick/
    ├── watercolor/
    ├── photo/
    ├── 3dcg/
    ├── comic/
    └── pixelart/
```

学習スクリプトが自動的にtrain/valディレクトリを検出し、既存の分割を使用します。

### サポートする画像形式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- WebP (.webp)

## 学習

### 基本的な学習コマンド

```bash
# 標準的なトレーニング (MSA-Netなし)
# データセットが自動的にtrain/valに分割されます
python train.py \
  --data_dir dataset \
  --output_dir outputs \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --num_workers 4
```

### カスタム分割比率での学習

```bash
# Validationの比率を30%に設定
python train.py \
  --data_dir dataset \
  --output_dir outputs \
  --batch_size 8 \
  --epochs 50 \
  --val_split 0.3
```

### MSA-Netを使用した学習

```bash
# MSA-Netを有効化
python train.py \
  --data_dir dataset \
  --output_dir outputs \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --use_msa
```

### 学習の再開

```bash
# チェックポイントから学習を再開
python train.py \
  --data_dir dataset \
  --output_dir outputs \
  --resume outputs/latest_checkpoint.pth
```

### 学習パラメータ

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--data_dir` | データセットのディレクトリ | `dataset` |
| `--output_dir` | 出力ディレクトリ | `outputs` |
| `--batch_size` | バッチサイズ | `8` |
| `--epochs` | エポック数 | `50` |
| `--lr` | 学習率 | `1e-4` |
| `--num_workers` | データローダーのワーカー数 | `4` |
| `--use_msa` | MSA-Netを使用 | `False` |
| `--resume` | チェックポイントから再開 | `None` |
| `--val_split` | Validationの分割比率（自動分割時） | `0.2` |
| `--random_seed` | データ分割のランダムシード | `42` |

### TensorBoardで学習過程を確認

```bash
# TensorBoardの起動
tensorboard --logdir outputs/logs --port 6006

# ブラウザでアクセス
# http://localhost:6006
```

### 学習時の出力

学習時には以下のファイルが生成されます:

- `best_model.pth` - 最高精度のモデル
- `latest_checkpoint.pth` - 最新のチェックポイント
- `class_info.json` - クラス情報
- `logs/` - TensorBoardのログ

## 推論

### 単一画像の推論

```bash
# 1枚の画像を推論
python inference.py \
  --checkpoint outputs/best_model.pth \
  --image path/to/image.jpg
```

出力例:
```
============================================================
Predicted Class: anime
Confidence: 95.23%

Class Probabilities:
  anime          : 95.23% ██████████████████████████████████████████████████
  brush          :  2.15% ██
  thick          :  1.45% █
  watercolor     :  0.67%
  photo          :  0.30%
  3dcg           :  0.12%
  comic          :  0.06%
  pixelart       :  0.02%

Style Coordinates: (2.3456, -1.2345)
============================================================
```

### フォルダー内の全画像を推論

```bash
# フォルダー内の全画像を推論
python inference.py \
  --checkpoint outputs/best_model.pth \
  --folder path/to/images/
```

### JSON形式で結果を保存

```bash
# 推論結果をJSONファイルに保存
python inference.py \
  --checkpoint outputs/best_model.pth \
  --folder path/to/images/ \
  --output_json results.json
```

## スタイル空間の可視化

### 1. 学習データから参照データを作成

```bash
# 学習データの分布を記録
python inference.py \
  --checkpoint outputs/best_model.pth \
  --build_reference \
  --data_dir dataset \
  --reference outputs/style_space_reference.json \
  --visualize \
  --output_plot training_distribution.png
```

### 2. 新しい画像をスタイル空間に配置

```bash
# 単一画像をスタイル空間に可視化
python inference.py \
  --checkpoint outputs/best_model.pth \
  --image path/to/image.jpg \
  --visualize \
  --reference outputs/style_space_reference.json \
  --output_plot style_space_single.png

# フォルダー内の全画像をスタイル空間に可視化
python inference.py \
  --checkpoint outputs/best_model.pth \
  --folder path/to/images/ \
  --visualize \
  --reference outputs/style_space_reference.json \
  --output_plot style_space_batch.png
```

### 3. 座標軸の範囲を指定

```bash
# X軸とY軸の範囲を指定して可視化
python inference.py \
  --checkpoint outputs/best_model.pth \
  --folder path/to/images/ \
  --visualize \
  --reference outputs/style_space_reference.json \
  --x_range -10 10 \
  --y_range -10 10 \
  --output_plot style_space_custom_range.png
```

### 可視化パラメータ

| パラメータ | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--visualize` | 可視化を有効化 | `False` |
| `--build_reference` | 参照データを作成 | `False` |
| `--reference` | 参照データのパス | `outputs/style_space_reference.json` |
| `--x_range MIN MAX` | X軸の範囲 | 自動 |
| `--y_range MIN MAX` | Y軸の範囲 | 自動 |
| `--output_plot` | 出力画像のパス | `style_space.png` |

### スタイル空間の見方

- **色付きの点(薄い色)**: 学習データの各画風の分布
- **星マーク**: 各画風の中心点
- **破線の楕円**: 各画風の2σ信頼区間
- **赤枠のひし形**: 新しく推論した画像の位置
- **黄色の吹き出し**: 画像ファイル名

## プロジェクト構造

```
Image_Classification/
├── train.py                    # 学習スクリプト
├── inference.py                # 推論スクリプト
├── requirements.txt            # 依存パッケージ
├── README.md                   # このファイル
├── dataset/                    # データセット
│   ├── train/                  # 学習データ
│   └── val/                    # 検証データ
└── outputs/                    # 出力ディレクトリ
    ├── best_model.pth          # 最良モデル
    ├── latest_checkpoint.pth   # 最新チェックポイント
    ├── class_info.json         # クラス情報
    ├── style_space_reference.json  # スタイル空間参照データ
    └── logs/                   # TensorBoardログ
```

## モデルアーキテクチャ

### EfficientNet B7

- ImageNetで事前学習済みの高性能なCNNモデル
- 入力サイズ: 512×512
- 効率的なスケーリングにより高精度を実現

### MSA-Net (オプション)

- Multimodal Style Aggregation Network
- Average PoolingとMax Poolingを組み合わせたアテンションメカニズム
- スタイル特徴の効果的な集約

### クラス不均衡対応

- データセットの枚数を自動集計
- 逆頻度重み付けによるサンプリング
- 損失関数への重み適用

## トラブルシューティング

### CUDAが認識されない

```bash
# CUDAドライバーの確認
nvidia-smi

# PyTorchでCUDAが使えるか確認
python -c "import torch; print(torch.cuda.is_available())"
```

### メモリ不足エラー

```bash
# バッチサイズを小さくする
python train.py --batch_size 4

# またはグラデーション蓄積を使用
# (train.pyの修正が必要)
```

### 学習が進まない

- 学習率を調整: `--lr 5e-5` や `--lr 1e-3` を試す
- データ拡張の確認: データセットの品質を確認
- TensorBoardでlossの推移を確認

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 参考文献

- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- PyTorch Image Models (timm)
