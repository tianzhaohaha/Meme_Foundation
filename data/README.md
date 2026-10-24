# 数据文件

仓库只包含样本 ID、标签和验证/测试划分清单，不包含图片、原始文本、预训练权重或模型 checkpoint。

取得有使用权限的 `subset_50000.zip` 后，将其放在本目录，或使用 `--data /absolute/path/subset_50000.zip` 指定路径。ZIP 内部应包含：

```text
subset_50000/
  all_data_id_and_label_with_polarity.txt
  all_data/<id>.jpg
  all_data/<id>.txt
```

`manifests/source_binary_20000.tsv` 和 `manifests/finetune_5000.tsv` 是固定训练清单；前者20,000行，后者5,000行，列为 `id / label / polarity`。`validation.tsv`、`test.tsv` 是已完成 seed42 实验的评估清单，各2,000行，列为 `id / emotion / image_sha256`。

默认预处理按 seed42 和原ZIP顺序重新生成相同划分。若更换数据ZIP、split seed或验证/测试规模，得到的是新实验；不要与本仓库报告的指标混合。数据指纹包含缓存绝对路径，移动目录会改变指纹，应核对固定清单中的ID和图片哈希确认划分相同。

验证/测试与训练之间、验证与测试之间均排除了完全重复图片及规范化非空文本。两份训练清单之间保留了227个重复图片哈希；本轮不声称已排除语义近重复。
