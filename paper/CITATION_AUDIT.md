# 引用核实报告

核实日期：2026-08-12。核实原则是以 arXiv、出版社论文页、会议论文集或 DOI 元数据为准。`refs.bib` 仅收录下列已确认存在的文献，并删除或更正来源清单中没有依据的字段。`paper/CITATION_NEEDED.md` 中的五类需求均可由已确认条目满足，建议映射为：AOI 复判使用 `deng2018autovrs`、`chen2023pcbreview`；PCB 深度检测使用 `tang2019deeppcb`、`chen2023pcbreview`；参考引导检测使用 `tang2019deeppcb`、`fridman2021changechip`；工业变化检测使用 `fridman2021changechip`；工业 MLLM 能力与风险使用 `jiang2025mmad`、`xu2025anomalyov`、`andrade2026agreementbias`。这些映射也写入了 `refs.bib` 顶部注释，未改动其他作者负责的章节文件。

## 逐条核实

| 编号 | 标题 | 结论 | 依据 |
|---:|---|---|---|
| 1 | Online PCB Defect Detector On A New PCB Defect Dataset | 已确认 | [arXiv:1902.06197](https://arxiv.org/abs/1902.06197) 的标题、四位作者和提交年份一致。 |
| 2 | A PCB Dataset for Defects Detection and Classification | 已确认 | [arXiv:1901.08204](https://arxiv.org/abs/1901.08204) 的标题、作者和年份一致。 |
| 3 | Building an Automatic Defect Verification System Using Deep Neural Network for PCB Defect Classification | 已确认 | [DBLP 记录](https://dblp.org/rec/conf/icfsp/DengLD18)与 DOI `10.1109/ICFSP.2018.8552045` 共同确认 ICFSP 2018、页码 145--149。 |
| 4 | Cost-Sensitive Siamese Network for PCB Defect Classification | 已确认 | [PubMed Central 全文记录](https://pmc.ncbi.nlm.nih.gov/articles/PMC8526275/)确认作者、2021 年、文章号和 DOI。原清单把期刊写为 Scientific Programming，实际期刊为 Computational Intelligence and Neuroscience，已更正。 |
| 5 | ChangeChip: A Reference-Based Unsupervised Change Detection for PCB Defect Detection | 已确认 | [IEEE Xplore](https://ieeexplore.ieee.org/document/9707699)与 [arXiv:2109.05746](https://arxiv.org/abs/2109.05746)相互印证。 |
| 6 | A Comprehensive Review of Deep Learning-Based PCB Defect Detection | 已确认 | [DBLP](https://dblp.org/rec/journals/access/ChenWHM23.html)确认 IEEE Access 11、页码 139017--139038 和 DOI。原清单作者“I. M. Chen et al.”不符，实际作者为 Xing Chen、Yonglei Wu、Xingyou He、Wuyi Ming，已更正。 |
| 7 | A Dataset for Deep Learning Based Detection of Printed Circuit Board Surface Defect | 已确认 | [Scientific Data 论文页](https://www.nature.com/articles/s41597-024-03656-8)确认 DOI、作者和 2024 年卷次，PDF 标明文章号 811。 |
| 8 | Industrial Printed Circuit Board Surface Defect Dataset for Object Detection | 已确认 | [Scientific Data 论文页](https://www.nature.com/articles/s41597-026-07684-4)确认 2026-06-27 发表、作者与 DOI。页面未给出卷期和文章号，BibTeX 未填写。 |
| 9 | RefDiffNet: Learning to Expose Subtle PCB Defects Before Detection | 已确认 | [arXiv:2606.00852](https://arxiv.org/abs/2606.00852)确认 2026-05-30 提交，作者为 Vinay Edula、Nilesh Badwe、Priyanka Bagade。 |
| 10 | SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation | 已确认 | [DBLP](https://dblp.org/rec/conf/eccv/ZouJPZD22)确认 ECCV 2022、页码 392--408；论文 DOI 为 `10.1007/978-3-031-20056-4_23`。 |
| 11 | AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models | 已确认 | [AAAI 官方论文页](https://ojs.aaai.org/index.php/AAAI/article/view/27963)确认作者、卷期、页码和 DOI。 |
| 12 | VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection | 已确认 | [arXiv:2409.20146](https://arxiv.org/abs/2409.20146)确认标题、作者和 2024 年提交记录。仅按预印本收录，未加入后续期刊字段。 |
| 13 | MMAD: A Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection | 已确认 | [arXiv:2410.09453](https://arxiv.org/abs/2410.09453)注明 Accepted by ICLR 2025，[OpenReview 论文](https://openreview.net/forum?id=JDiER86r8v)可对应确认。 |
| 14 | Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models | 已确认 | [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.html)确认 CVPR 2025 与作者；对应 [arXiv:2502.07601](https://arxiv.org/abs/2502.07601)。未核实页码，BibTeX 留空。 |
| 15 | EMIT: Enhancing MLLMs for Industrial Anomaly Detection via Difficulty-Aware GRPO | 已确认 | [arXiv:2507.21619](https://arxiv.org/abs/2507.21619)确认标题、六位作者和年份。 |
| 16 | UniPCB: A Unified Vision-Language Benchmark for Open-Ended PCB Quality Inspection | 已确认 | [arXiv:2601.19222](https://arxiv.org/abs/2601.19222)确认 2026-01-27 提交及六位作者。 |
| 17 | DifferAD-R1: A Difference-Guided Industrial Anomaly Localization with Multimodal Large Language Models | 已确认 | [arXiv:2606.16601](https://arxiv.org/abs/2606.16601)确认 2026-06-15 提交及八位作者。标题在 arXiv 页面存在排版粘连，BibTeX 按正常空格恢复。 |
| 18 | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | 已确认 | [PMLR/ICML 记录](https://proceedings.mlr.press/v97/geifman19a.html)确认作者、卷 97 和页码 2151--2159。 |
| 19 | On Calibration of Modern Neural Networks | 已确认 | [PMLR/ICML 记录](https://proceedings.mlr.press/v70/guo17a.html)确认作者、卷 70 和页码 1321--1330。 |
| 20 | Consistent Estimators for Learning to Defer to an Expert | 已确认 | [PMLR/ICML 记录](https://proceedings.mlr.press/v119/mozannar20b.html)确认作者、卷 119 和页码 7076--7087。 |
| 21 | Conformal Risk Control | 已确认 | [ICLR 2024 官方论文集](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)确认标题和五位作者。 |
| 22 | When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs | 已确认 | [arXiv:2406.01297](https://arxiv.org/abs/2406.01297)确认标题、作者和年份。本次按清单指定的预印本版本收录。 |
| 23 | Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification | 已确认 | [arXiv:2507.11662](https://arxiv.org/abs/2507.11662)确认论文与作者；[项目页](https://mshalimay.github.io/agreement-bias-sgv/)和 OpenReview 记录确认 ICLR 2026 接收。 |
| 24 | Predicting When to Trust Vision-Language Models for Spatial Reasoning | 已确认 | [arXiv:2601.11644](https://arxiv.org/abs/2601.11644)确认 2026-01-14 提交，作者为 Muhammad Imran、Yugyung Lee。 |
| 25 | Image Quality Assessment: From Error Visibility to Structural Similarity | 已确认 | [IEEE 论文页](https://ieeexplore.ieee.org/document/1284395)确认 TIP 13(4)、600--612 和 DOI `10.1109/TIP.2003.819861`。 |
| 26 | Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization | 已确认 | DOI `10.1109/TPAMI.2008.113` 的出版元数据确认 TPAMI 30(10)、1858--1865。 |
| 27 | A Fast Parallel Algorithm for Thinning Digital Patterns | 已确认 | [ACM DOI](https://doi.org/10.1145/357994.358023)确认 Communications of the ACM 27(3)、236--239。 |
| 28 | DINOv2: Learning Robust Visual Features without Supervision | 已确认 | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)与 [Meta AI 项目页](https://dinov2.metademolab.com/)相互印证。 |
| 29 | Qwen2.5-VL Technical Report | 已确认 | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)确认标题、作者和年份。 |
| 30 | Qwen3-VL Technical Report | 已确认 | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631)确认标题、作者和 2025 年提交记录。 |
| 31 | MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe | 已确认 | [arXiv:2509.18154](https://arxiv.org/abs/2509.18154)确认论文实际为 2025 年预印本。没有找到支持“CVPR 2026: 11704--11715”的官方记录，该部分已删除，只以 arXiv 版本收录。 |

## 存疑或无法核实文献

无。31 篇文献的作品实体均已确认存在。编号 4、6、31 的原始书目信息含错误或未获支持的字段，已在上表说明并在 `refs.bib` 中纠正或删除；这些元数据错误不影响作品实体的“已确认”结论。
