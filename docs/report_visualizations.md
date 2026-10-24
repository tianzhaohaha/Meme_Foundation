# 数据、错误对照与类中心图的复现

阅读图表只需打开[报告目录](../reports/README.md)。本次新增8组图，每组包含PNG和PDF：标签分布、原始样本、预处理对照、输入质量总览、分标签输入质量、错误重叠、错误案例、类中心相似度。

## 生成输入与核验

使用原训练环境中的Python依赖，再安装 `requirements-analysis.txt`；全部分析在CPU上完成，不重新训练或运行GPU推理。Pillow、transformers和scikit-learn来自训练环境。脚本只从本地缓存加载固定revision的RoBERTa tokenizer；若尚未准备模型缓存，先完成[环境和数据准备](reproduction.md)。自定义Hugging Face缓存时，设置 `HF_HOME` 和 `HF_HUB_CACHE` 指向同一缓存根与其hub子目录。

数据图需要原始 `subset_50000.zip` 和 `experiment_manifest.json` 对应的准备缓存：

```bash
python tools/analyze_dataset.py \
  --data data/subset_50000.zip \
  --cache training/prepared_comparison_v2
```

它核验完整ZIP SHA-256、29k记录的可移植指纹、validation/test清单、每条原图哈希与清洗文本，并复核历史类别/空文本数量。文本长度包含RoBERTa特殊token，清洗后总数大于96才计入截断；图片尺寸考虑EXIF方向。统计按记录加权，保留原训练清单重复。

错误与类中心图复用已经提取并核验的四个NPZ，以及公开Qwen预测：

```bash
python tools/analyze_model_outputs.py \
  --data data/subset_50000.zip \
  --features embedding_analysis/results/features
```

需要 `Binary_validation.npz`、`V4_validation.npz`、`V5_validation.npz`、`V6_validation.npz`。提取方法见[embedding模块说明](../embedding_analysis/README.md)。高维NPZ和checkpoint未发布；只有原始作者的对应文件才能逐值重绘本次历史图。脚本将NPZ哈希与 `reports/data/embedding_visualization.json` 对照，将预测accuracy/macro-F1与 `reports/data/vlm_validation.json` 对照，避免把新运行的结果误混入历史报告。重新训练产生的新checkpoint应另建核验记录和报告，不能期待哈希及分数与历史文件相同。

两条命令默认写入 `reports/figures/` 和 `reports/data/`。可用 `--output /path/to/new_reports` 保存到其他位置；脚本不会修改训练文件或报告中的手写分析段落。它们会生成少量样本缩略图和文本节选供报告使用，不会导出完整原始数据。

## 可复核的定义

- 样本墙：Target每类两个，按 `SHA256("42:" + ID)` 排序；不会按图片内容或预测结果挑选。
- 诊断案例：预先固定错误条件，条件内使用同样的ID排序；保留最大V4错误softmax样本作为最后一例，避免重复。案例不是随机抽取的准确率估计。
- 错误对照：按ID对齐同一2k validation，拒绝漏行、重复ID、错误标签或未知模型；Qwen的Invalid计错。共享二分类模型不参与七分类错误排名。
- 类中心：先对每条256维融合特征做L2归一化，按真实七类分别求均值，再计算七个均值之间的余弦。它完全不依赖PCA/t-SNE坐标，不做全局中心化。四个面板固定色标0–1；原特征在ReLU之后，故为非负。

样本的图像缩放、填充和JPEG重读与训练一致；预处理展示还逐像素核对训练缓存。Markdown中的样本文本转义后展示。输入数值CSV、逐样本预测CSV及完整7×7相似度矩阵随报告保存，高维特征仍留在本地。
