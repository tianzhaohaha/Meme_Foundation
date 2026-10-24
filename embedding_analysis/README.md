# 共享二分类源与V4/V5/V6的特征可视化

把整个 `embedding_analysis/` 文件夹复制到 **Meme_Foundation 项目根目录**，与 `training/`、`data/` 并列。模块包含自己的模型构造、预处理和固定验证清单，不需要覆盖原项目文件，也不修改训练脚本或 checkpoint。

```text
Meme_Foundation/
  training/
  data/
  embedding_analysis/
    __main__.py
    extract.py
    plot.py
    _model.py
    _data.py
    manifests/
    README.md
    requirements.txt
```

## 一条命令：提特征并画图

激活原训练环境，进入项目根目录。安装画图所需依赖：

```bash
pip install -r embedding_analysis/requirements.txt
```

如果数据和模型位于仓库默认位置，直接运行：

```bash
python -m embedding_analysis
```

数据或模型目录不同时：

```bash
python -m embedding_analysis \
  --data /absolute/path/subset_50000.zip \
  --runs-root /absolute/path/runs_comparison
```

默认比较以下四个验证集选出的权重，不进行新训练：

```text
runs_comparison/shared_binary/<配置指纹>/seed_42/best.pt  # Binary共享源
runs_comparison/direct_5k/seed_42/best.pt      # V4
runs_comparison/reset_fusion/seed_42/best.pt   # V5
runs_comparison/tune_fusion/seed_42/best.pt    # V6
```

共享二分类目录只有一个候选源模型时会自动发现；存在多个候选时必须显式指定，避免误用其他实验的源模型。也可以直接指定全部四个checkpoint：

```bash
python -m embedding_analysis \
  --data /path/subset_50000.zip \
  --checkpoint Binary=/path/shared_binary/best.pt \
  --checkpoint V4=/path/V4/best.pt \
  --checkpoint V5=/path/V5/best.pt \
  --checkpoint V6=/path/V6/best.pt
```

历史 checkpoint 若因含旧 NumPy 随机状态而被受限加载拒绝，只对自己生成且可信的历史文件增加 `--trust-legacy-checkpoint`。默认 CUDA；不会自动改用本地 CPU。只执行前向推理，不训练模型。

## 产物

默认写入模块自己的 `embedding_analysis/results/`：

```text
results/
  features/      # 三版 .npz 特征与 .json 来源、验证指标
  plots/         # PCA / t-SNE PNG、PDF、坐标 CSV、投影设置
```

默认使用固定的同一批 2,000 条 validation 样本，提取分类头之前的 256 维融合表示。图按七种真实情感和真实正负情感分别着色，以2×2面板比较四个模型。四个模型的融合特征矩阵未压缩共约8.2 MB（不含标签和其他元数据）。

Binary严格保留原来的二分类头，输出维度为2；它的七类着色来自样本的真实标签，并非七分类预测。Binary仅计算二分类accuracy/F1作为权重复现检查，不与V4/V5/V6的七分类accuracy直接排名。若同时指定Binary和V5/V6，还会校验后两者记录的源checkpoint哈希与所选Binary一致。

`--include-encoded` 可额外提取1,536维融合前特征并画前后两组图；四个模型的两组特征矩阵未压缩共约57.3 MB。本次已发布的可视化只使用256维融合表示，没有额外提取encoded特征或训练探针。
`--output /path/results` 指定新输出目录；结果目录中的同名分析文件会被覆盖，训练目录不变。

## 只抽特征，发回电脑画图

```bash
python -m embedding_analysis extract \
  --data /path/subset_50000.zip \
  --runs-root /path/runs_comparison
```

把 `results/features/` 发回即可。画图无需 GPU、checkpoint 或原始数据：

```bash
python -m embedding_analysis plot \
  --features embedding_analysis/results/features/Binary_validation.npz \
             embedding_analysis/results/features/V4_validation.npz \
             embedding_analysis/results/features/V5_validation.npz \
             embedding_analysis/results/features/V6_validation.npz
```

## 比较口径

严格加载每版 best.pt，核对模型版本、标签顺序和最佳epoch；核对全部样本图片哈希和原始标签，保留训练时相同的图像JPEG预处理与文本清洗；在eval/inference_mode下提取特征。JSON记录新算的validation accuracy/F1是否与checkpoint指标一致。任一指标误差达到1e-6或缺少历史指标时立即停止，不导出该模型特征；绘图也拒绝验证未通过的文件。绘图前还校验所有模型的样本ID、顺序、标签、图片哈希、处理后输入指纹和文本配置一致。

PCA/t-SNE每个模型独立拟合，默认L2归一化；t-SNE先PCA到最多50维，perplexity 30、1500 iterations、seed 42。模型之间的坐标方向和簇间距离不能直接比较。七分类/二分类着色使用同一投影坐标，点绘制顺序按固定seed打乱，避免固定把某一类别覆盖在其他类别之上。二维图仅作描述，不能单凭簇形状证明信息丢失或负迁移。

只比较原三个七分类模型可用 `--variants V4 V5 V6`。V1可通过 `--variants Binary V1 V4 V5 V6` 加入，但它使用更多七分类标签，只作为数据量更大的参考，不是理论上限。

模型与预处理基于 https://github.com/tianzhaohaha/Meme_Foundation 。`_model.py` 包含可选 text_config 参数，以固定 revision 的真实 RoBERTa 配置严格还原 checkpoint；原项目无需修改。此包不含数据、checkpoint、下载凭据或真实特征结果。
