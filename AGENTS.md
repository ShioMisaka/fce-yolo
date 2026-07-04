# AGENTS.md

本文件为 ZCode / Claude Code / Codex 等 agent 提供本仓库的开发指引。
本仓库是 **ultralytics（YOLOv11）的 fork**，上游框架（engine/models/nn/cfg）尽量不动；
本项目改动集中在 **FCE 自定义模块** 与 **`script/` 训练实验工具链**。

> 红线：不要重构 `ultralytics/` 上游代码。本项目只新增 / 覆盖配置，改动落在 `ultralytics/nn/modules/fce_block.py`、`ultralytics/cfg/models/11/yolo11-*.yaml`、`ultralytics/cfg/default.yaml`、`script/` 这几处。

## FCE 自定义模块

实现于 `ultralytics/nn/modules/fce_block.py`，是本项目的核心创新点：

| 模块 | 作用 | 典型 YAML 参数 |
|------|------|----------------|
| **BiFPN_Concat** | 可学习加权多尺度特征融合 | `[-1, 6], 1, BiFPN_Concat, []` |
| **CoordAtt** | 坐标注意力（H/W 方向空间依赖） | `[-1, 1, CoordAtt, [256, 16]]`（oup, reduction） |
| **CoordCrossAtt** | 坐标交叉注意力（跨方向交互） | `[..., CoordCrossAtt, [256, 16, 2]]`（+num_heads） |
| **BiCoordCrossAtt** | 双向坐标交叉注意力（对称 H↔W） | `[..., BiCoordCrossAtt, [512, 16, 8]]` |

消融阶梯（`ablation_config.yaml` 的 models 顺序）：`baseline` ① → `+BiFPN` ② → `+BiFPN+Attn` ③ → `FCE (+WIoU)` ④。

**新增模块的完整流程**：见 `.agents/skills/add-module/SKILL.md`（实现类 → 更新 `__all__` → `tasks.py` 导入 → `parse_model()` 加参数解析 → YAML 引用 → 改文档）。

## 训练 / 实验工具链（`script/`）

模块化训练架构，支持单模型训练、多模型对比、配方驱动消融、论文图表生成。详细 CLI 文档见 `script/README.md`。

| 文件 | 角色 |
|------|------|
| `config.py` | 配置系统：`StageConfig`/`TrainConfig`/`ModelConfig` dataclass，`MODEL_CONFIGS`（baseline/bifpn/fce/fce_wiou），`DATASET_PRESETS`（default/coco/coco_hq） |
| `trainer.py` | `YOLOv11Trainer`：按 `ModelConfig.is_two_stage()` 自动跑单阶段或两阶段（stage1 预热 + stage2 微调） |
| `train.py` | 单模型训练 CLI：`python script/train.py <MODEL> --scale <n/s/m/l/x> [opts]` |
| `compare.py` | 多模型对比 CLI：`--models baseline fce --scale s` |
| `run_ablation.py` | **公平消融编排器**：读 `ablation_config.yaml` 配方，一键训练 + 整理 + 出图，产物落入 `runs/outputs/fair_<timestamp>/` |
| `ablation_config.yaml` | 消融配方（shared 统一变量 / stage1/stage2 / scales / models / iou_override）。**重做实验只改 YAML，不改代码** |
| `paper_figs.py` + `paper_figs_config.yaml` | 论文图表生成（A 收敛曲线 / B 消融柱状 / C 检测样例 / D PR+混淆矩阵） |
| `paper_plots.py` | 底层绘图函数（被 paper_figs 调用） |
| `analysis.py` | 无状态纯函数：`load_results`/`extract_metrics`/`plot_comparison_curves`，可独立 import |
| `pack_results.py` | 实验结果打包（跨机器回传） |
| `test.py` | 配置自检：`python script/test.py` → `✓ 所有测试通过!` |

### 两阶段训练约定（重要）

含随机初始化模块的模型（bifpn/fce/fce_wiou）**必须两阶段**，否则新增模块（BiFPN_Concat/BiCoordCrossAtt 从 yolo11m.pt 按层名匹配不到预训练权重）单阶段学不充分，会导致消融倒挂（M1≥M2≥M3）。

- **stage1（50 ep）**：`freeze=0`（不冻结 backbone，避免冻住层 5/8 的注意力），`lr0=0.001`，`cos_lr=True`，`close_mosaic=0`——给新增模块预热。
- **stage2（250 ep）**：`lr0=0.001`，`cos_lr=True`，`close_mosaic=20`——端到端微调。
- fair 消融中 baseline 也注入同样的两阶段配置（`dataclasses.replace`），保证 4 组变量完全统一。

根因分析见 `analysis_log/` 与近期 commit（如 `d5679f30 fix(ablation): 恢复两阶段训练`）。

### 跨机器工作流

工作站（Linux，全英文路径，有 GPU）训练 → 打包 → 本地（Windows）出图：

```bash
# 工作站
python script/run_ablation.py --scale m        # → runs/outputs/fair_<ts>/
zip -r fair_<ts>.zip runs/outputs/fair_<ts>     # 单文件夹自包含回传

# 本地
python script/run_ablation.py --replot fair_<ts>   # 从已有文件夹重出图，写回原位
```

⚠️ **`paper_figs.py` 输出全英文**——Linux 工作站无 CJK 字体，中文会渲染成方框。

## 关键文件位置

- FCE 模块实现：`ultralytics/nn/modules/fce_block.py`
- 模型 YAML：`ultralytics/cfg/models/11/{yolo11,yolo11-bifpn,yolo11-fce}.yaml`
- IoU 损失：`ultralytics/utils/loss.py`（`BboxLoss`）+ `metrics.py`（`bbox_iou`/`bbox_wiou`），配置项 `iou_type` 在 `cfg/default.yaml`
- 模型注册：`ultralytics/nn/tasks.py` 的 `parse_model()`（加新模块必须在此加参数解析）
- 配置覆盖链：`default.yaml` → `model.args` → `script/config.py` 的 `MODEL_CONFIGS`
- 结果输出：`runs/detect/`（ultralytics 原生）与 `runs/outputs/fair_<ts>/`（消融整理后）

## Git 提交规范

Conventional Commits + **中文描述**，详见 `.agents/rules/git-commit.md`。
常用 scope：`ablation`、`run_ablation`、`config`、`fce`、`paper_figs`、`script`。示例：`fix(ablation): 恢复两阶段训练修复 fair 消融倒挂`。

## 代码风格

- Python 类型注解 + Google 风格 docstring（Args/Returns/Examples）
- `script/` 下用 dataclass 管理配置，CLI 参数走 argparse，分析逻辑保持无状态纯函数
- 图表配色与 display_name 集中在 `paper_figs_config.yaml` / `config.py` 的 `MODEL_CONFIGS`，不散落硬编码
