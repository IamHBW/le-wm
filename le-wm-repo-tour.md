【输出文件】

`/media/gtb/UBUNTU_DATA/le-wm/le-wm-repo-tour.md`

【仓库目标】

这是一个很薄的研究仓库：它不想自己重写训练框架、环境系统和规划器，而是站在 `stable-pretraining` 与 `stable-worldmodel` 之上，只保留 LeWorldModel 的核心贡献，也就是 JEPA 架构、两项损失和最小训练/评测胶水。主目标不是“做一个完整平台”，而是把论文方法尽量直接地落到可复现实验代码里。

【当前问题】

用户想先理解这个仓库整体在做什么、主路径怎么走、哪些文件是真正重要的。

【第 1 层结构】
这一层的作用：

- `README.md`: 论文摘要、安装方式、数据放置约定、训练命令、评测命令。
- `train.py`: 训练入口，负责 Hydra 配置加载、数据集构建、模型装配、Lightning 启动。
- `eval.py`: 评测入口，负责从 checkpoint 构建 cost model，用 MPC/CEM 在 latent space 里规划。
- `jepa.py`: LeWM 的核心模型接口，定义编码、预测、rollout 和 planning cost。
- `module.py`: 真实的网络积木，尤其是 `ARPredictor`、`SIGReg`、`Embedder`。
- `utils.py`: 图像预处理、数值列归一化、对象 checkpoint 导出回调。
- `config/train/`: 训练配置，含默认超参和不同数据集切换。
- `config/eval/`: 评测配置，含不同环境、solver 和启动器。
- `assets/`: README 展示素材。
- `Maes 等 - 2026 - ...pdf`: 论文本体，可直接拿来做 paper-code 对照。

【优先展开】

先看 `train.py`、`eval.py`、`jepa.py` 和 `config/`。因为这个仓库的真实主干不是目录结构，而是两条执行链：训练链和规划评测链。

【第 2 层结构】
这一层的作用：

- `train.py`: 训练链的 orchestration 文件。它自己不写 Trainer，而是把数据、模型和损失塞给 `stable_pretraining`。
- `config/train/lewm.yaml`: 训练默认超参；默认数据集是 `pusht`，默认 world model 历史长度 3，预测器 6 层，SIGReg 权重 0.09。
- `config/train/data/*.yaml`: 不同任务的数据视图，只描述 HDF5 名字、frameskip、要加载哪些列。
- `eval.py`: 评测链入口。它主要干三件事：取评测数据、构造 `WorldModelPolicy`、把指标和视频写出来。
- `config/eval/*.yaml`: 环境相关配置，决定 `env_name`、评测 budget、goal offset、怎么把 dataset 中的状态写回环境。
- `config/eval/solver/*.yaml`: 规划器配置，默认是 CEM，也支持梯度优化。

【优先展开】

继续看 `jepa.py` 和 `module.py`。因为训练/评测入口本质只是接线，真正的论文贡献在模型和损失。

【第 3 层结构】
这一层的作用：

- `jepa.py::JEPA.encode`: 把像素序列展平成 `(B*T, ...)` 喂给 ViT encoder，取 CLS token，再投影成 latent embedding。
- `jepa.py::JEPA.predict`: 把历史 latent 和动作嵌入送入自回归 predictor，输出下一步 latent 预测。
- `jepa.py::JEPA.rollout`: 评测时在 latent space 里自回归展开候选动作序列。
- `jepa.py::JEPA.criterion`: 用终点 latent 与 goal latent 的 MSE 当规划 cost。
- `jepa.py::JEPA.get_cost`: planner 侧统一入口，先编码 goal，再 rollout，再出 cost。
- `module.py::SIGReg`: anti-collapse 正则，沿随机投影方向做高斯性检验。
- `module.py::Embedder`: 动作编码器，把动作序列压到与 latent 对齐的 embedding 空间。
- `module.py::ARPredictor`: 带 AdaLN-zero 条件调制的 causal Transformer，用动作条件预测未来 latent。
- `module.py::MLP`: encoder/projector 和 predictor projector 的简单头部。

【优先展开】

如果你下一轮想深挖“为什么这样设计能防 collapse”，就继续追 `SIGReg` 和 `lejepa_forward`。如果你想看“规划时到底怎么和外部库接上”，就去追 `stable_worldmodel.policy.WorldModelPolicy` 的实现。

【关键主路径】

训练主路径：

1. Hydra 载入 `config/train/lewm.yaml`，再组合某个 `config/train/data/*.yaml`。
2. `train.py` 用 `swm.data.HDF5Dataset` 打开 HDF5 数据，并对 `pixels` 做 ImageNet 标准化和 resize，对 `action/proprio/state/...` 做列归一化。
3. `train.py` 构建 ViT encoder、动作 `Embedder`、`ARPredictor`、两个 projector，然后装成 `JEPA`。
4. `lejepa_forward` 先编码整段序列，再取前 `history_size` 步作为上下文，预测未来 embedding，并计算
   - `pred_loss`: 预测 latent 和目标 latent 的 MSE
   - `sigreg_loss`: 对整段 embedding 做 SIGReg
   - `loss = pred_loss + lambda * sigreg_loss`
5. `stable_pretraining` 的 `Manager` 和 Lightning `Trainer` 执行训练，`utils.py` 的回调每个 epoch 导出一个可直接加载的对象 checkpoint。

评测主路径：

1. Hydra 载入某个 `config/eval/*.yaml`，默认 solver 是 `config/eval/solver/cem.yaml`。
2. `eval.py` 创建 `swm.World` 环境，并从数据集拟合 `StandardScaler`，供 action/state/proprio 等数值列标准化。
3. 若 `policy != random`，则通过 `swm.policy.AutoCostModel` 直接加载训练导出的对象 checkpoint。
4. solver 反复提出候选动作序列，模型通过 `JEPA.get_cost` 在 latent space 里 rollout，并计算终点到目标的 cost。
5. `world.evaluate_from_dataset` 从离线数据集采样起点和 goal，执行 MPC 评测，写指标和视频。

【论文-代码映射】

| 论文概念 | 代码位置 | 说明 |
| --- | --- | --- |
| `enc_theta(o_t)` | `train.py` 中 ViT encoder 装配 + `jepa.py::JEPA.encode` | 像素经 ViT 提取 CLS token，再经 projector 变成低维 latent。 |
| `pred_phi(z_t, a_t)` | `module.py::ARPredictor` + `module.py::Embedder` + `jepa.py::JEPA.predict` | 动作先编码，再通过带 AdaLN-zero 的 causal Transformer 条件化预测。 |
| `L_pred` | `train.py::lejepa_forward` | 直接做 predicted embedding 与 target embedding 的 MSE。 |
| `SIGReg(Z)` | `module.py::SIGReg` | 随机投影后做 Epps-Pulley 风格统计，逼近各向同性高斯。 |
| `L = L_pred + lambda SIGReg(Z)` | `train.py::lejepa_forward` | 这就是本仓库真正的训练目标，代码和论文对得上。 |
| 终点 goal-matching cost | `jepa.py::JEPA.criterion` | 规划时比较 rollout 末端 latent 与 goal latent。 |
| MPC / CEM planning | `eval.py` + `config/eval/solver/cem.yaml` | 规划逻辑主体在 `stable_worldmodel` 外部库中。 |

【风险 / 疑点】

- 这个仓库的大量关键行为藏在外部依赖里，尤其是 `stable_worldmodel` 的 `World`、`AutoCostModel`、`WorldModelPolicy` 和 solver。只看本仓库会低估系统复杂度。
- `SIGReg` 的注释明确写了 `single-GPU!`，但训练配置里 `trainer.devices: auto`，如果真上多卡，正则统计是否仍与论文假设一致，需要额外确认。
- 训练逻辑非常依赖 HDF5 列名与外部环境约定，比如 `episode_idx/ep_idx`、`goal_state`、`goal_qpos`。这部分没有本地 schema 校验，属于“配错就炸”的研究代码风格。
- 规划评测只用了 terminal latent cost，没有本地 reward model 或多项 cost 设计；这是论文选择，不是平台能力缺失。
- 没有独立 `tests/`，最小验证手段仍是跑一次训练和一次评测 smoke run。

【未解决问题】

- `stable_worldmodel` 内部如何把对象 checkpoint 适配成 `AutoCostModel`，以及它对 `get_cost` 的接口约束是什么。
- `num_preds > 1` 时训练监督的严格语义是什么，虽然配置上支持，但默认实验只用 `1`。
- `keys_to_merge`、`keys_to_cache` 等 HDF5 细节在外部库里具体怎么影响数据读取性能与字段拼装。
