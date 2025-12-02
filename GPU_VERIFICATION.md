# GPU使用確認ガイド

## ダブルチェック項目

### 1. GPU設定の確認

```bash
# GPUチェックスクリプトを実行
python check_gpu.py
```

このスクリプトは以下を確認します:
- ✅ CUDAが利用可能か
- ✅ GPUデバイス情報
- ✅ GPU上でのテンソル操作
- ✅ メモリ管理

### 2. train.pyのGPU使用箇所

train.pyでは以下の箇所でGPUを使用しています:

#### モデルのGPU配置 (train.py:521)
```python
model = model.to(device)
```

#### 損失関数の重みをGPUに配置 (train.py:530)
```python
class_weights_gpu = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_gpu)
```

#### 学習時のデータ転送 (train.py:361-362)
```python
inputs = inputs.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
```

#### 検証時のデータ転送 (train.py:414-415)
```python
inputs = inputs.to(device, non_blocking=True)
targets = targets.to(device, non_blocking=True)
```

### 3. 実行時のGPU確認メッセージ

学習開始時に以下のメッセージが表示されます:

```
[GPU Status] Model device: cuda:0
[GPU Status] Model is on GPU: True
[GPU Status] Initial GPU Memory: 0.XX GB
[GPU Status] Loss criterion weight device: cuda:0

[GPU Check] Input device: cuda:0
[GPU Check] Target device: cuda:0
[GPU Check] Model device: cuda:0
[GPU Check] GPU Memory Allocated: X.XX GB
```

### 4. GPU使用の検証方法

#### 方法1: nvidia-smiでリアルタイム監視
```bash
# 別ターミナルで実行
watch -n 1 nvidia-smi
```

学習中にGPU使用率とメモリが増加することを確認

#### 方法2: PyTorchからGPUメモリを確認
学習スクリプト内で自動的に表示されます

#### 方法3: GPU使用率の確認
```bash
# 学習実行中に別ターミナルで
nvidia-smi dmon -s u
```

### 5. トラブルシューティング

#### GPUが使われていない場合

1. **CUDAが認識されているか確認**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. **PyTorchのCUDAバージョン確認**
```bash
python -c "import torch; print(torch.version.cuda)"
```

3. **GPU上にモデルがあるか確認**
学習開始時のログで `[GPU Status] Model is on GPU: True` を確認

4. **データがGPUに転送されているか確認**
学習開始時のログで `[GPU Check] Input device: cuda:0` を確認

## セマフォリーク問題の対処

### 修正内容

1. **persistent_workersを無効化** (train.py:331, 341)
```python
persistent_workers=False
```

2. **multiprocessing_contextを削除**
spawnコンテキストがセマフォリークの原因だったため削除

3. **prefetch_factorを追加**
```python
prefetch_factor=2 if num_workers > 0 else None
```

### それでもエラーが出る場合

```bash
# ワーカー数を減らす
python train.py --num_workers 2

# またはワーカーを無効化（最も安全）
python train.py --num_workers 0
```

## 期待される動作

### 正常な学習の流れ

1. GPU情報が表示される
2. モデルがGPUに配置される
3. 各エポックで以下が表示される:
   - `[GPU Check]` メッセージ (最初のエポックのみ)
   - プログレスバー
   - GPU使用率が上昇 (nvidia-smiで確認)

### GPU学習の確認ポイント

- ✅ `device: cuda:0` と表示される
- ✅ `is_cuda: True` と表示される
- ✅ GPU Memory Allocated が増加する
- ✅ nvidia-smiでGPU使用率が上がる
- ✅ 学習が高速 (CPUの10-50倍)

## 最終確認

すべて正常に動作している場合、学習開始時に以下のような出力が得られます:

```
Using device: cuda

GPU Information:
  GPU Name: NVIDIA GeForce RTX 3090
  CUDA Available: True
  PyTorch CUDA Version: 12.1
  Number of GPUs: 1
  GPU Compute Capability: 8.6
  CUDA Test: Passed

[GPU Status] Model device: cuda:0
[GPU Status] Model is on GPU: True
[GPU Status] Initial GPU Memory: 2.34 GB
[GPU Status] Loss criterion weight device: cuda:0

Epoch 1/50
--------------------------------------------------------------------------------
[GPU Check] Input device: cuda:0
[GPU Check] Target device: cuda:0
[GPU Check] Model device: cuda:0
[GPU Check] GPU Memory Allocated: 5.67 GB

Training: 100%|██████████| 100/100 [00:45<00:00, 2.21it/s, loss=2.123, acc=45.2%]
```

これが確認できれば、GPUで正しく学習が行われています！
