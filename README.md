# PCB-EviGate：基于证据卡与双门控的 PCB 缺陷检测结果多模态大模型复核

这个仓库包含 PCB 缺陷检测结果复核系统的全部代码、配置、实验结果和论文源文件。系统流程如下：YOLOv8s 检测器产生检测结果，差异候选模块补充可能的漏检，再为每个候选生成证据卡交给多模态大模型复核。学习式**复检门**决定哪些候选值得调用模型，**接受门**决定是否采纳模型给出的修改。

数据集为 [DeepPCB](https://github.com/tangsanli5201/DeepPCB)。评价采用五折 GroupKFold，按模板组隔离；官方划分存在模板泄漏，只作对比参考。

## 主要结果

| 方法（五折均值 ± 标准差） | F1 | TPRR | HCR |
|---|---:|---:|---:|
| B1 检测器 + 工作点 | 0.9118 ± 0.0344 | | |
| B7 全量复检，无门控 | 0.8660 ± 0.0368 | 0.9572 | 0.9326 |
| B8 手工线性风险门 | 0.8787 ± 0.0471 | 0.9717 | 0.9607 |
| **Full 学习式双门控** | **0.9117 ± 0.0343** | **0.9998** | 0.4000 |
| Oracle-C 候选上界 | 0.9555 ± 0.0254 | | |
| Oracle-A 接受上界 | 0.9143 ± 0.0336 | 1.0000 | 0.0000 |

- **RQ1 证据消融**：只有正常模板（A2）带来稳定增益，F1 相对 A0 提升 0.1178，95% CI [+0.026, +0.230]。
- **RQ2 端到端**：Full 相对 B1 的差值为 -0.000064，图像聚类 bootstrap 95% CI 为 [-0.000273, +0.000094]，两者没有统计差异。双门控挡住了有害修改，但净收益接近零；Oracle-A 分析表明瓶颈在于大模型本身的判断质量。
- **RQ3 预算曲线**：在 5% 调用预算下，复检门的复核召回为 0.4649，随机策略为 0.0315。有接受门时，F1 在任何预算下都保持在约 0.9115。
- **RQ4 鲁棒性**：容差 XOR 把漏检恢复率从 6.71% 提高到 23.62%。

完整结论见 `docs/最终实验总结.md`，论文见 `paper/main.pdf`。

## 获取代码与数据

git 仓库里没有模型权重和 DeepPCB 原始图像，这两部分放在 [Releases](../../releases) 中：

| Release 附件 | 内容 | 大小 |
|---|---|---|
| `pcb-evigate-full-v1.0.zip.part00` 到 `part04` | 完整复现包：本仓库全部内容，外加权重和 DeepPCB 数据 | 约 450 MB |
| `pcb-evigate-weights-v1.0.zip.part00` 到 `part01` | 只含权重：五折 YOLOv8s `best.pt`、RT-DETR-R18 fold0、`yolov8s.pt` 预训练权重 | 约 200 MB |
| `SHA256SUMS.txt` | 两个合并后 zip 的校验和 | |

两个 zip 都按 95 MB 切成分片上传。下载全部分片后，先合并再校验：

```bash
cat pcb-evigate-full-v1.0.zip.part* > pcb-evigate-full-v1.0.zip
cat pcb-evigate-weights-v1.0.zip.part* > pcb-evigate-weights-v1.0.zip
sha256sum -c SHA256SUMS.txt
```

**方式 A：使用完整包（推荐）**

```bash
unzip pcb-evigate-full-v1.0.zip && cd pcb-evigate
```

**方式 B：clone 仓库，再补齐权重和数据**

```bash
git clone https://github.com/Clay-HHK/pcb-evigate.git && cd pcb-evigate
unzip /path/to/pcb-evigate-weights-v1.0.zip -d .
git clone https://github.com/tangsanli5201/DeepPCB.git data/raw/DeepPCB
git -C data/raw/DeepPCB checkout 08e98c4db5922613fb97176eb3d6497d48260cb1
```

## 环境要求

### 硬件

| 项目 | 实验所用配置 | 最低要求 |
|---|---|---|
| GPU | 4 × NVIDIA Tesla T4 16 GB | L1 不需要 GPU；L2、L3 需要 1 块 ≥ 12 GB 显存的 GPU |
| CPU / 内存 | 64 核 / 251 GB | 8 核 / 32 GB |
| 磁盘 | | L1 约 1 GB；L2、L3 重新生成证据卡还需约 2 GB |

### 软件

| 组件 | 版本 |
|---|---|
| OS | Linux（Ubuntu，kernel 6.8） |
| NVIDIA 驱动 / CUDA | 570.144 / CUDA 12.4（PyTorch wheel 自带运行库） |
| Python | 3.11.15 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| ultralytics | 8.4.117 |
| transformers | 4.55.4（RT-DETR-R18VD） |
| scikit-learn / LightGBM | 1.9.0 / 4.7.0 |
| OpenCV / scikit-image | 5.0.0 / 0.26.0 |
| codex CLI（只在需要新调用大模型时用到） | 0.155.1 |

### 安装

```bash
# conda（推荐）
conda create -n pcb-evigate python=3.11.15 -y
conda activate pcb-evigate
pip install -r requirements.txt

# 也可以完整还原实验时的环境
conda env create -f environment.yml
# 或 pip install -r requirements-lock.txt
```

`requirements.txt` 只固定直接依赖；`requirements-lock.txt` 和 `environment.yml` 是实验环境导出的完整依赖清单。

### 环境变量

```bash
export YOLO_OFFLINE=1               # 避免 ultralytics 联网校验，直接使用本地 yolov8s.pt
export MPLCONFIGDIR=/tmp/mpl        # 可选
# 只有发起新的大模型调用时才需要：
export DASHSCOPE_API_KEY=...        # configs/mllm/openai_compatible.yaml 读取此变量
```

所有命令都在仓库根目录下执行。仓库中不含任何 API 密钥。

## 复现

复现分三个层级，越往后成本越高。

### L1 结果核验（CPU，约 1 分钟）

```bash
python scripts/verify_release.py
```

脚本先按 `MANIFEST.sha256` 校验所有文件的哈希，然后用打包的五折逐图预测（`experiments/rq2_correction/fold_outputs/`）和 DeepPCB 标注，重新计算 Full 方法的 precision、recall、F1、mAP@0.5 和 mAP@0.5:0.95，并与 `full_results.json` 比对。全部一致时输出 `PASS`。只 clone 了仓库、没有下载完整包时，用 `--skip-hash` 跳过哈希校验。

### L2 免训练复现（GPU，不调用大模型）

打包的权重和中间结果可以跳过检测器训练；`cache/mllm_outputs/codex_cli.jsonl` 缓存了全部 7185 次真实大模型响应。缓存键由候选、证据配置、模型名和温度决定（`src/mllm/client.py: cache_key`），使用默认配置重跑时会全部命中缓存，不会发起新的调用。证据卡没有打包，运行时会自动重新生成；`examples/evidence_cards/` 中保留了 A0、A2、A6 各 20 张样例。

| 论文内容 | 命令 | 输出 |
|---|---|---|
| 门控训练（阶段 C） | `python experiments/stage5_gates/run.py` | `experiments/stage5_gates/` |
| RQ1 全量证据消融 | `python experiments/rq1_evidence/run_full.py` | `experiments/rq1_evidence/full_*` |
| RQ1 证据诊断（B2） | `python experiments/rq1_evidence/run_diag.py` | `experiments/rq1_evidence/diagnosis_*` |
| RQ2 端到端 + RQ3 预算曲线 | `python experiments/rq2_correction/run_full.py` | `experiments/rq2_correction/`、`experiments/rq3_gating/` |
| RQ4 扰动鲁棒性 | `python experiments/rq4_robustness/run.py` | `experiments/rq4_robustness/perturbation_*` |
| RQ4 检测器替换（RT-DETR） | `python experiments/rq4_robustness/run_rtdetr_full.py` | `experiments/rq4_robustness/rtdetr_*` |
| RQ4 门控模型替换 | `python experiments/rq4_robustness/run_model_replacement.py` | `experiments/rq4_robustness/model_replacement.json` |
| 论文图 | `python scripts/make_paper_figures.py` | `paper/figures/` |
| 统一结果表 | 由以上脚本汇总生成 | `experiments/results_summary.csv` |

### L3 从头复现

```bash
# 数据管线：manifest、五折划分、泄漏检查
python scripts/build_manifest.py
python scripts/build_splits.py
python scripts/check_leakage.py

# 五折 YOLOv8s 训练、预测、校准、误差分解（4 × T4 并行，约 70 分钟）
bash scripts/run_full_training.sh

# 差异候选参数扫描（阶段 A）
python scripts/sweep_diff_candidates.py

# RT-DETR-R18VD fold0 微调（RQ4，首次运行需从 HuggingFace 下载 PekingU/rtdetr_r18vd）
python experiments/rq4_robustness/train_rtdetr_r18.py
```

之后按 L2 的顺序运行各实验。

**关于大模型调用的可复现性**：主实验使用 codex CLI 调用 `gpt-5.6-sol`，`temperature=0`、`reasoning_effort=low`（`configs/mllm/codex_cli.yaml`）。即使温度为 0，外部模型的输出也不保证完全确定，模型版本也可能更新，所以删除缓存重新调用得到的结果会与论文数字有偏差。论文数字以打包的缓存为准。换成其他 OpenAI 兼容接口时，可参考 `configs/mllm/openai_compatible.yaml`。

### 随机性控制

所有实验的随机种子固定为 `20260811`（检测器训练、分层抽样、门控模型、bootstrap）。阈值只在 val 折上选取，测试折不参与调参。接受门的安全下限预先固定为 remove 0.90、relabel 0.75、add 0.80。

## 目录结构

```
src/                    核心代码
  detector/             YOLOv8s 训练、预测、校准、误差分解
  candidates/           差异候选生成与融合
  registration/         ECC 配准与配准质量分
  evidence/             外观、拓扑证据与证据卡渲染
  mllm/                 大模型客户端、prompt 与缓存
  gates/                复检门、接受门与安全规则
  evaluation/           检测指标、纠错指标、Oracle、动作应用
configs/                检测器、证据、大模型配置
scripts/                数据管线、训练驱动、论文作图、结果核验
experiments/            各阶段实验脚本与全部结果（json/csv/md/log）
  results_summary.csv   统一结果表（6169 行）
  figures/              实验结果图
cache/
  mllm_outputs/         全部真实大模型响应缓存
  gate_oof/             五折 OOF 候选（51,222 个）
  detector_predictions/ detector_eval/ detector_runs/   检测器预测、评价与训练记录
  detector_dataset/     五折 YOLO 数据集（图像为指向 data/raw 的相对软链接）
  rtdetr_r18_fold0/     RT-DETR fold0 结果
data/
  manifests/ splits/    样本清单与五折划分
  raw/DeepPCB/          原始数据（需单独获取，见上文）
docs/                   开发规格书、各阶段实验报告、问题台账
paper/                  论文 LaTeX 源文件、图和 PDF
examples/evidence_cards/  证据卡样例
```

## 许可

DeepPCB 数据集使用 MIT 许可，版权归原作者所有，见 `data/raw/DeepPCB/LICENSE`。使用时请引用原论文：

> Tang, S., He, F., Huang, X., & Yang, J. Online PCB Defect Detector On A New PCB Defect Dataset. arXiv:1902.06197, 2019.
