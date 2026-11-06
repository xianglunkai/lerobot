# π0.5 分拣螺丝：FlashRT 落地手册

按阶段执行。上一阶段的门禁没过，不要进入下一阶段，也不要上真机。

本手册只覆盖这一份权重：

`checkpoints/outputs/pi05_policy_screw_sorting_single_sft_ep279_v30_annotated-pi07/checkpoints/030000/pretrained_model`

当前 HIL 入口是 `scripts/run_hil_collect_data.sh` 的 spacemouse 分支。FlashRT 只替换 10 步去噪。分位数反归一化、相对动作还原、夹爪排除、OSQP、动作队列和 `fixed_inference_delay=6` 留在 LeRobot。

## 0. 先看结论

这台工作站是 NVIDIA GeForce RTX 3090，compute capability 8.6。FlashRT 的 `detect_arch()` 对 SM 8.6 直接抛错，π0.5 的 torch 前端只登记了这些架构：

| 卡 | SM | FlashRT 名称 |
| --- | --- | --- |
| RTX 4090 | 8.9 | `rtx_sm89` |
| RTX 5090 | 12.0 | `rtx_sm120` |
| Jetson Orin | 8.7 | `rtx_sm87` |
| Thor | 11.0 | `thor` |

3090 上不要编译、不要 `hardware="rtx_sm89"`、不要开 FP8。Ampere 没有 FP8 tensor core，强行指定架构只会在 kernel 启动时失败。3090 上继续用现在的 `torch.compile` 路径；FlashRT 换到上表里的卡再做。

README 里的 8–13 ms 是 RTX 5090 的数。4090 / Orin 用对应行的公开数字做预期，本机测完再记自己的数。

## 1. 这份模型的固定合同

下面这些数在后续命令里原样使用，不要沿用 FlashRT 的默认值。

| 项 | 值 |
| --- | --- |
| 相机 | `observation.images.high`，然后 `observation.images.right`。各 480×640，模型输入 224×224。`num_views=2`，`empty_cameras=0` |
| state / action | 各 7 维。模型内部 pad 到 `max_action_dim=32`，有效维是前 7 维 |
| 相对动作 | `use_relative_actions=true`。`right_joint6.pos`（夹爪）排除，保持绝对 |
| 归一化 | state、action 都是 QUANTILES。统计在同目录的 `policy_preprocessor_step_3_normalizer_processor.safetensors` 和 `policy_postprocessor_step_0_unnormalizer_processor.safetensors` |
| chunk | `chunk_size=32`，`n_action_steps=32` |
| 去噪 | `num_inference_steps=10`，bf16，`gemma_2b` + `gemma_300m` |
| 文本 | `tokenizer_max_length=200`。任务句见 HIL 脚本：`Please sort and return the silver screws in the grey box to their proper places.` 离散化后的 state 写在 prompt 里 |
| RTC | `rtc_config` 为空。`rtc_training_config.enabled=true`，`max_delay=10`。HIL 固定 `fixed_inference_delay=6` |
| 队列 | `execution_horizon=10`，`action_queue_size_to_get_new_actions=12`，`fps=30`，`interpolation_multiplier=1`，`use_torch_compile=true` |
| 后处理顺序 | quantile 反归一化 → `absolute_actions_processor` → OSQP（fps 30）→ cpu |

Training-time RTC 的语义以 `src/lerobot/policies/pi05/modeling_pi05.py` 的 `sample_actions` 为准，每一步都做两件事：

1. 调用速度场之前，把 `x_t` 的前 `d` 步写成上一 chunk 的剩余动作，这些位置的 flow time 设为 0，其余位置用当前步的 time。`d = fixed_inference_delay = 6`。
2. Euler 更新 `x_t = x_t + dt * v_t` 之后，再把前 6 步写回同一段剩余动作。`dt = -1/10`，time 从 1.0 走到 0.1。

第一帧没有上一 chunk，`prev_chunk_left_over is None`，走普通 10 步去噪。从第二帧起才钉前 6 步。

## 2. 阶段 0：确认卡

在准备跑 FlashRT 的机器上执行：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
python - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability()
print(f"SM {major}.{minor}")
PY
```

`8.6`：停在这里。后面的阶段换到 8.9 / 8.7 / 12.0 / 11.0 的机器再做。

`8.9`、`8.7`、`12.0`、`11.0`：继续。记下 SM，后面 BF16 和 FP8 的选择用得到。

FlashRT README 的安装要求写着 GPU「SM80+（A100、30xx+）」，CMake 也有 `-DGPU_ARCH=86`（RTX 3090）。那是 FA2 注意力 kernel 的编译目标：cubin 能在这张卡上编出来。π0.5 能不能加载由 `detect_arch()` 决定，它只认 8.7、8.9、12.0、11.0。8.6 会抛 `unsupported GPU SM 8.6`，调度表里没有 `rtx_sm86` 的 π0.5 前端。README 硬件表把 3090 标成 “Build target”，同页也写了未登记的 SM 在分发时失败。所以这张卡可以编译部分 kernel，不能把这份 π0.5 跑进 `load_model`。

## 3. 阶段 1：在支持的卡上安装

按 FlashRT 仓库当前的安装说明编译 torch 版。本阶段只验证导入和架构识别：

```bash
python - <<'PY'
from flash_rt.hardware import detect_arch
print(detect_arch())
PY
```

输出必须是 `rtx_sm89`、`rtx_sm87`、`rtx_sm120`、`thor` 之一。出现 `unsupported GPU` 就回到阶段 0。

权重目录原样拷过去，七个文件都要在：

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_preprocessor_step_3_normalizer_processor.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors`
- `train_config.json`

`model.safetensors` 是 LeRobot 键名（带 `model.` 前缀）。前端加载时会去掉这个前缀，目录名保持 `pretrained_model`，不要先转成别的格式。

## 4. 阶段 2：用对构造函数

`flash_rt.load_model()` 不转发 `chunk_size`，也不转发 `max_prompt_len`。直接调用它会落在前端默认值上：chunk 10、prompt 最长 48。这份模型是 chunk 32、tokenizer 200，用默认值跑出来的动作序列和训练分布不是同一个策略。

在支持的卡上直接构造 RTX 前端。第一轮强制 BF16，用来做数值对齐；FP8 留到阶段 6，对齐通过之后再开。

```python
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

CKPT = "/abs/path/to/pretrained_model"

pipe = Pi05TorchFrontendRtx(
    CKPT,
    num_views=2,
    chunk_size=32,
    num_steps=10,
    max_prompt_len=200,
    state_prompt_mode="fixed",
    use_fp8=False,
    use_cuda_graph=True,
    denoise_trace=True,
)
```

Thor 用对应的 `Pi05TorchFrontendThor`，参数同样显式写成上面这组，不要靠类默认值。

`state_prompt_mode="fixed"` 让变长的 state prompt 垫到固定长度，CUDA graph 只捕获一次。`denoise_trace=True` 会在结果里给出归一化空间的 `raw_actions`，形状 `(32, 32)`。机器人指令不要用返回值里的 `actions`：那个字段已经做完 quantile 反归一化，并切成前端内置的 7 维，里面没有「关节相对、夹爪绝对」这一步，也没有 OSQP。

## 5. 阶段 3：先对齐「没有 RTC」的第一帧

这一阶段 `prev_chunk` 为空，和 LeRobot 第一帧的普通去噪对齐。目的是确认权重、相机顺序、prompt、归一化统计一致。

准备一条离线样本，字段和 HIL 批一致：

- 图像：`observation.images.high`、`observation.images.right`，HWC uint8，480×640。
- 状态：7 维，关节顺序与数据集 feature 名一致，夹爪在最后。
- 同一条 prompt。
- 同一份初始噪声，形状 `(1, 32, 32)`，bf16。后 25 维是 pad，两边填同一个噪声。

LeRobot 侧调用 `PI05Pytorch.sample_actions`，`inference_delay=None`，`prev_chunk_left_over=None`，`num_steps=10`。取出 Euler 结束、后处理之前的张量，切前 7 维，得到 `(32, 7)`。

FlashRT 侧把 high 送给 `image`，right 送给 `wrist_image`，把同一份噪声的前两维去掉 batch 后传入 `infer(..., noise=noise[0], return_noise=True)`。读 `raw_actions[:, :7]`。

门禁，在归一化空间、7 个有效维上算：

- 32 步的最大绝对误差 `< 1e-2`（BF16）。
- 余弦相似度 `> 0.999`。

不过门禁时只查这四件事，查完再改代码：

1. `chunk_size` 是否真是 32，`num_steps` 是否真是 10。
2. 相机是否 high 在前、right 在后。若整体余弦很低而每路单独看正常，对调两路再测一次。
3. prompt 是否被截到 48。`max_prompt_len` 必须是 200。
4. 归一化统计是否来自这个目录里的两份 safetensors，而不是 LIBERO 自带统计。

这一阶段不要上机器人。

## 6. 阶段 4：改 FlashRT 的 decoder style，固定 d=6

这是落地 training-time RTC 的必要改动，仓库里没有开关。

`Pi05TorchFrontendRtx._precompute_decoder_styles` 对每个去噪步只算一个 time embedding，再 `expand` 到 chunk 的全部 32 个位置。Training-time RTC 要求同一步里前 6 个位置的 time 是 0，后 26 个位置才是该步的 time。style 是 AdaRMS 的输入，time 错了，速度场就不是同一个模型。

改法保持静态，这样 CUDA graph 仍然只捕获一张：

对 `step = 0 .. 9`：

- `t_suffix = 1.0 + step * (-1.0 / 10)`
- 位置 `0:6` 使用 `t = 0` 的 time MLP 输出
- 位置 `6:32` 使用 `t = t_suffix` 的 time MLP 输出
- `time_emb`、`style_attn`、`style_ffn`、`style_final` 都按这个位置规则写，不要再 `expand` 成整段同一个向量

`d=6` 写死在这张图里。不要为 `max_delay` 的 0–10 各捕获一张图，也不要在图里做 Python 分支。HIL 的 `fixed_inference_delay` 保持 6，和这张图一致。

去噪循环里还要补两处拷贝，对应 LeRobot 在 Euler 前后各写一次前缀。前缀来自调用方传入的 buffer，形状 `(32, 32)`，前 6 行的前 7 列是上一 chunk 的归一化剩余动作，其余 pad 为 0：

- 每个 step 算速度之前：`x_t[:6, :] = prefix[:6, :]`
- Euler 之后再写一次同样的 6 行

这两次拷贝放进已捕获的 graph，用固定长度的 copy，不在 host 上按延迟分支。`infer` 增加参数 `prev_action_norm`，形状 `(32, 32)` 或 `(6, 32)`。第一帧传 `None`，走阶段 3 的普通图；从第二帧起传剩余动作，走这张 d=6 的图。两张图都是静态的。

不要把 `torch.compile` 套在同一段去噪上。FlashRT 接管之后，HIL 脚本把 `--use_torch_compile=false`。

## 7. 阶段 5：对齐「有 RTC」的第二帧

用阶段 3 同一条样本再跑一帧。

1. 第一帧：两边都不带前缀，保存 LeRobot 的归一化 chunk，记为 `prev`，形状 `(32, 32)`。
2. 第二帧：换一张图和一份新噪声。LeRobot 调用 `sample_actions(..., inference_delay=6, prev_chunk_left_over=prev)`。FlashRT 调用 `infer(..., noise=new_noise, prev_action_norm=prev)`。
3. 比较两边后处理之前的 `(32, 7)`。

门禁：

- 前 6 步与 `prev[:, :7]` 按元素相等，最大绝对误差 `< 1e-5`。这 6 步是钉进去的，不是网络重新采样的。
- 后 26 步最大绝对误差 `< 1e-2`，余弦 `> 0.999`。

前 6 步对不上，说明前缀没写进 `x_t`，或写进之后被 Euler 冲掉。后 26 步对不上、前 6 步是对的，说明 style 仍是整段同一个 time。两种情况都回到阶段 4，不要调 HIL 队列阈值来掩盖。

通过之后才允许打开 FP8（`use_fp8=True`，且仅 SM89 / SM120 / Thor）。FP8 再跑同一条第二帧：后 26 步余弦仍要 `> 0.995`。低于这个数就固定用 BF16 上机。SM87 的 Orin 若前端没有可用的 FP8，保持 BF16。

## 8. 阶段 6：只换 HIL 推理线程里的去噪

替换点是 `examples/rac/hil_data_collection.py` 推理线程里对 `predict_action_chunk` 的调用所进入的 `sample_actions`。队列合并、`ActionQueue`、`fixed_inference_delay=6`、`action_queue_size_to_get_new_actions=12`、`execution_horizon=10` 不动。

数据流：

1. LeRobot preprocessor 继续做相机顺序、resize、quantile 归一化、state 写入 prompt。
2. 把 preprocessor 之后、`sample_actions` 之前的图像、语言 token、归一化状态和噪声交给 FlashRT。
3. 取 `raw_actions`，形状 `(32, 32)`，切前 7 维，还原成 `(1, 32, 7)`。
4. 把这个张量送进现有 postprocessor：quantile 反归一化 → 相对动作还原（夹爪 `right_joint6.pos` 保持绝对）→ OSQP → cpu。
5. 后处理后的 chunk 仍交给原来的 `ActionQueue.merge`。

不要把 FlashRT 的 `actions`（已经反归一化的 7 维）直接发给机械臂。那样会跳过相对动作还原和 OSQP，关节量纲和现在的策略不一致。

第一帧 `prev_action_norm=None`。之后每一帧把「尚未执行、仍留在队列里的归一化动作」按时间对齐后作为 `prev_action_norm`，长度不足 6 的位置用上一 chunk 里已经对齐的剩余补齐，pad 维填 0。传给 FlashRT 的前缀必须是归一化空间的动作，和 `sample_actions` 里的 `prev_chunk_left_over` 是同一份数。不要把后处理之后的绝对关节角喂回去。

上机前把脚本改成：

```bash
--use_torch_compile=false
--fixed_inference_delay=6
--rtc.execution_horizon=10
--action_queue_size_to_get_new_actions=12
```

`rtc.max_guidance_weight` 在 training-time RTC 开启时不会进入去噪，留着无影响。不要把它改成推理时 guidance 来「配合」FlashRT。

## 9. 阶段 7：上机时看什么

先空跑一个 episode，手不要放在示教器上。

日志里应看到：

- 第一次推理没有前缀，之后每次推理的 delay 是 6。
- 推理耗时稳定。丢掉前 200 次（含校准和 graph capture），再取中位数。这个中位数才是这张卡的数，不要和 README 的 5090 数字比较。
- `delayqueue` 在一次推理之后降到阈值 12 附近，再触发下一次推理。它若停在 28 左右，说明几乎只执行了被钉住的前缀，机械臂会停在当前关节。这是队列阈值问题，不是 FlashRT 算错；阈值保持 12。

动作抽查：记录里 `|action - state|` 的均值应明显高于 `0.001` rad。若回到 `0.001` 附近，前 6 步仍是 hold-pose，后 26 步没有被队列发出去。先看 `action_queue_size_to_get_new_actions` 是不是 12，再看传给 FlashRT 的前缀是不是上一 chunk 的真实剩余，而不是「停在当前位」的占位。

延迟预算：30 Hz 下一步是 33 ms。BF16 对齐已经通过、但中位数仍高于约 25 ms 时，先保持 OSQP 和现有队列，不要为了压延迟去减 `num_steps` 或改 chunk。减步数会换成另一个策略。

## 10. 做完的判定

同时满足才算落地：

1. 阶段 3 的第一帧门禁通过。
2. 阶段 5 的前 6 步被钉住，后 26 步过余弦门禁。
3. 真机一个 episode 里，队列会消费到阈值 12，手臂离开当前关节。
4. 干预数据仍由原来的 spacemouse 分支写入，后处理链仍是反归一化、绝对动作、OSQP。

## 11. 不要做的事

- 在 RTX 3090 上编译或运行这份前端。
- 用 `load_model(config="pi05")` 的默认 chunk 10、prompt 48 直接 `predict()`。
- 把 FlashRT 反归一化后的 7 维 `actions` 当作关节指令。
- 在 style 仍是「一步一个 time、铺满 32 个位置」时打开 training-time RTC。那样前缀时间不再是 0，钉住的动作和训练分布不一致。
- 在 FlashRT 去噪之外再套一层 `torch.compile`。
- 为了延迟把 `num_inference_steps` 改成不是 10，或把 chunk 改成不是 32。
- 修改 `fixed_inference_delay`，除非阶段 4 的静态图一起改成同一个 `d`。
