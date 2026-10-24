# Qwen视觉语言模型：同一validation集的零样本七分类

此实验补充V1–V6监督训练结果，评估两个通用视觉语言模型是否能直接完成TumEmo细粒度情绪分类。没有任务微调、训练示例、检索或提示词搜索。两个模型使用同一提示词和解码规则。

## 模型与固定版本

| 模型 | Hugging Face revision |
|---|---|
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |

使用Instruct版本、BF16权重和Transformers的SDPA注意力，无量化。4B/8B是模型名称中的规模；结果另记录实际参数数量。实现直接使用Transformers的Qwen3-VL类和官方processor，不启用远程模型代码。官方说明见[Qwen3-VL仓库](https://github.com/QwenLM/Qwen3-VL)。

## 数据与比较口径

- 仅评估已提交的 `data/manifests/validation.tsv` 中2,000条记录，顺序保持一致；不使用test集。
- 复用原预处理缓存的224×224 RGB图片，Qwen视觉输入固定224×224，即49个合并后的视觉token。没有改为高分辨率原图，因此这是受控低分辨率基线，不代表Qwen在原生高分辨率下的最好表现。
- 采用原来的URL/情绪hashtag清洗规则。按固定版本RoBERTa tokenizer、`max_length=96`（含特殊token）取得字符偏移，给Qwen输入保留的文本前缀；不额外提供原RoBERTa输入以外的后续文本。
- 标签、样本ID、文件名和polarity不进入提示词。每条只有一张图和一段文本；两种模型的输入指纹必须一致。
- 类别固定为 `Angry, Bored, Calm, Fear, Happy, Love, Sad`。不合格式的输出计错，仍进入accuracy分母和各类召回率；不从含多个类别的回答中猜标签。
- 原V1–V6用validation选最佳epoch，Qwen没有在该集合上微调或选提示词。因此可以报告同一集合的描述性差异，但不应视为独立test泛化排名，也不能把监督训练与零样本方案称为相同训练预算。通用预训练语料是否包含相关样本没有审计。

完整提示词保存在 `training/evaluate_vlm.py` 的 `PROMPT` 及每个结果的配置中。固定greedy解码、`num_beams=1`、最多16个新token；不采样、无额外推理轮次。大小写、包裹引号和末尾句号/感叹号可规范化，其余回答判为无效。

## 运行

先完成[主复现指南](reproduction.md)中的环境创建和数据准备。无需V1–V6的checkpoint；首次使用会下载两个Qwen模型，共约27 GB。模型与运行输出都被Git排除。主训练环境中的Transformers 4.57.6已包含所需模型类，无需安装vLLM或额外推理服务。

```bash
# 在仓库根目录，激活自己的虚拟环境
source .venv/bin/activate
bash training/run_vlm_validation.sh

# 可选：将聚合结果记录到自己的W&B项目
bash training/run_vlm_validation.sh --wandb-mode online
```

SSH后台运行：

```bash
mkdir -p training/background_vlm
nohup setsid bash training/run_vlm_validation.sh \
  > training/background_vlm/evaluation.log 2>&1 < /dev/null &
echo $! > training/background_vlm/worker.pid
tail -f training/background_vlm/evaluation.log
```

默认在GPU 0上先4B、后8B串行执行，batch16，避免同GPU竞争影响计时。可以设置 `CUDA_VISIBLE_DEVICES`；较小显存设备可以用 `--batch-size`，但耗时不再是同batch对照。已有缓存同时指定 `HF_HOME` 和 `HF_HUB_CACHE`，避免继承的环境变量指向其他缓存。缓存完整后可以设置 `HF_HUB_OFFLINE=1`。

进程意外中断后，以相同参数添加 `--resume` 重新启动；已完成模型会跳过，未完成模型从最后一个完整batch继续。每批结果原子保存，正在计算但未保存的batch会重算。不要同时启动多个相同输出目录的进程，文件锁会阻止覆盖。单独运行或检查：

```bash
python training/evaluate_vlm.py --model 4b --cache training/prepared_comparison_v2
# 仅做小规模接口检查，独立输出；不能作为完整validation结果
python training/evaluate_vlm.py --model 4b --limit 16 --output training/runs_vlm_smoke
```

`training/runs_vlm/qwen3_vl_4b/` 和 `qwen3_vl_8b/` 各包含配置、逐样本预测进度、最终指标、效率和可选W&B链接。更改模型、提示词、batch、输入或硬件时使用新的 `--output` 目录。论文/报告结果应要求 `full_validation=true` 且样本数为2,000。

## 计时与结果文件

推理前用合成灰色图片和固定无标签文本执行一个batch预热。`evaluation_seconds`为完整batch耗时之和，包含图片读取、processor、数据传输、生成和解码；不包含数据核验、权重下载/加载、预热、结果写盘和W&B同步。模型加载与预热时间另列；中断的未完成batch时间不计入。`amortized_seconds_per_sample`是batch吞吐折算，不能解释为单请求延迟。

Qwen两种规模之间使用同一GPU、batch、精度和输入协议。V1–V6的既有计时是训练成本，不与本次推理时间直接相除。原有测试集报告保持不变，新增validation结果单独呈现。

完成后的导出和重绘：

```bash
python tools/collect_vlm_results.py
python tools/plot_vlm_results.py \
  --input training/runs_vlm/analysis/vlm_validation.json \
  --output training/runs_vlm/analysis/figures
```

导出工具重新计分、核对2,000条样本的ID/顺序/标签及两模型协议一致性；输出不包含原始图片、文本或模型权重。已发布[结果与分析](../reports/experiment_review.md#qwen零样本七分类同一2000条validation)可以直接阅读，运行 `python tools/plot_vlm_results.py` 可重绘仓库中的参考图表。
