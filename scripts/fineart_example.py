# 一条 t=10s 的帧，从 render 的输出字段交到训练 forward 和推理。
# 键名与源码一致，方便在调试器里对。
#
#   RenderTrainingMessagesStep 写出 complementary:
#       messages_rendered, message_streams, target_message_indices
#   FineARTVLATextTokenizerStep 读上面三份，写出:
#       observation["observation.language.tokens"]
#       observation["observation.language.attention_mask"]
#       complementary["text_labels"]
#       complementary["predict_actions"]
#   ActionTokenizerProcessorStep 写出:
#       complementary["action.tokens" / "action.token_mask" / "action.code_token_mask"]
#   FineARTVLAPolicy.forward 只读这几份张量，不再看见 subtask 字符串。

frame = {
    "timestamp": 10.0,
    "index": 0,
    "task": "fasten the zip tie",
    "observation.images.high": "img_h",
    "observation.state": "state_raw",
    "action": "action_raw (50, d)",
    # 整集列表复制在每一帧上。t=10 时 active subtask 是 7.5 那句。
    "language_persistent": [
        {"style": "subtask", "content": "pick up the zip tie tail", "timestamp": 0.0, "camera": None},
        {"style": "memory", "content": "I picked up the zip tie.", "timestamp": 7.5, "camera": None},
        {"style": "subtask", "content": "insert the zip tie tail through the lock", "timestamp": 7.5, "camera": None},
        {"style": "memory", "content": "I picked up the zip tie. I threaded its tail through the lock.", "timestamp": 17.5, "camera": None},
        {"style": "subtask", "content": "release the zip tie and retract the arm", "timestamp": 17.5, "camera": None},
    ],
    "language_events": [],
}

# ---------------------------------------------------------------------------
# A. 低层样本 index=0
#    blake2b("0")/2**64 ≈ 0.516 >= 0.30 → low_level_execution
#    active_at(10, style=subtask) → "insert the zip tie tail through the lock"
# ---------------------------------------------------------------------------

# A1. RenderTrainingMessagesStep 的返回值就是这三份，写进 complementary。
#     同时删掉 language_persistent / language_events。
complementary = {
    "messages_rendered": [
        {"role": "user", "content": "insert the zip tie tail through the lock"},
    ],
    "message_streams": ["low_level"],
    "target_message_indices": [],
}

# A2. FineARTVLATextTokenizerStep 读 complementary 这三份。
#     stream 含 low_level → 归一化 state 离散成 256 档，接到最后一条 user。
prompt = "User: insert the zip tie tail through the lock, State: 128 131 140 96;\n"
#     target_message_indices 为空 → 没有任何 token 是监督目标。
#     message_streams 含 low_level → predict_actions=True。
observation = {
    "observation.images.high": "img_h",
    "observation.state": "state_norm",
    "observation.language.tokens": "tokenizer(prompt)  # (1, L)",
    "observation.language.attention_mask": "(1, L) bool",
}
complementary["text_labels"] = "(1, L) 全是 -100"
complementary["predict_actions"] = "tensor([True])"

# A3. FAST 编码的是归一化 action，与 subtask 字符串无关。
complementary["action.tokens"] = "(1, F) FAST 码"
complementary["action.token_mask"] = "(1, F)"
complementary["action.code_token_mask"] = "(1, F) 哪些位置进交叉熵"

# A4. forward 的读取点（modeling_fineart_vla.py FineARTVLAPolicy.forward）
#     text_labels 存在、predict_actions 为 True、FAST 默认开
#     → run_flow / run_text / run_fast 都为真
#     → _compute_all_losses_fused 一次前向：
#         embed_prefix(images, observation.language.tokens)   # 前缀里是那句 subtask + State
#         末尾再接 action.tokens
#         x_t = t*eps + (1-t)*action ; u_t = eps - action
#         flow = MSE(action_out_proj(suffix), u_t)            # 真实动作维
#         text = CE(语言段, text_labels)                      # -100 = ignore_index，全忽略则分子为 0
#         fast = CE(前缀末尾, action.tokens)                  # 非代码位也改成 -100，那些位不进分母
# 逐步计算见 docs/fineart_vla_loss_calc.html
#     total = 10*flow + 1*text + 1*fast

# ---------------------------------------------------------------------------
# B. 高层样本 index=2，单独一个 batch
#    blake2b("2")/2**64 ≈ 0.109 < 0.30 → high_level_subtask
#    同一句 subtask 这次是 Assistant 的目标，不是 User 的条件。
# ---------------------------------------------------------------------------

complementary_high = {
    "messages_rendered": [
        {"role": "user", "content": "fasten the zip tie"},
        {"role": "assistant", "content": "insert the zip tie tail through the lock"},
    ],
    "message_streams": ["high_level", "high_level"],
    "target_message_indices": [1],  # 只监督第 1 条消息
}
prompt_high = (
    "User: fasten the zip tie\n"
    "Assistant: insert the zip tie tail through the lock<eos>\n"
)
# 没有 low_level，不加 State。
# text_labels：User 段是 -100，Assistant 段是 subtask 的 token id。
# predict_actions = False，因为没有 low_level 流。
#
# forward：predict_any=False → run_flow=False, run_fast=False, run_text=True
# → _compute_text_and_fast_loss，不建动作后缀
#    位置 i 的隐状态预测位置 i+1，只在 subtask token 上算 CE + 1e-4 * z-loss
#    total = 1 * text_loss

# ---------------------------------------------------------------------------
# C. 推理不走 RenderTrainingMessages，也不读 language_persistent。
#    训练时写进 messages_rendered 的那句 subtask，推理时变成任务槽里的字符串。
# ---------------------------------------------------------------------------

# C1. 生成 subtask。rollout 在进 preprocessor 之前打上：
runtime_complementary = {
    "query_kind": "next_subtask",
    "query_text": "fasten the zip tie",  # /autosteer 的目标，不是数据集 task 列
}
# RenderTrainingMessagesStep 看见 query_kind 就 return，不调用 render_sample。
# RenderRuntimeMessagesStep 用 prompt_turns("subtask")，只留下目标回合之前的 User：
runtime_complementary = {
    "messages_rendered": [{"role": "user", "content": "fasten the zip tie"}],
    # 不写 message_streams，也不写 target_message_indices。
}
# 分词器因此在末尾补 "Assistant:"，并且不加 State（没有 low_level）。
prompt_gen = "User: fasten the zip tie\nAssistant:"
# select_message 读的是分词结果，不是 messages_rendered：
#   observation.language.tokens / observation.language.attention_mask + 当前图像
#   逐 token lm_head，屏蔽 FAST 词表，EOS 停止
generated = "insert the zip tie tail through the lock"
engine_task = generated  # set_task：这就是下一步动作条件的全部语言

# C2. 出动作。这一拍的 complementary 没有 query_kind，没有 language_persistent，没有 action。
# RenderTrainingMessagesStep：没有语言列且没有 action → return。
# 分词器看到 messages_rendered 缺失，用任务槽造消息，并标成 low_level：
action_complementary = {
    "task": engine_task,
    "messages_rendered": [[{"role": "user", "content": engine_task}]],
    "message_streams": [["low_level"]],
    "target_message_indices": [[]],
}
prompt_act = f"User: {engine_task}, State: <当前离散状态>;\n"
# sample_actions 只读 observation.language.tokens 和图像。
# 前缀 KV 缓存一次；x 从噪声出发，10 步 denoise_step；每步 crop 掉 suffix cache。
# 后处理反归一化，送给机器人。这里没有 text_labels，也不算三份训练损失。
