# 独立 TopK Tensor Capture 使用指南

用于隔离/canary 实例的低频诊断：在真实 Indexer TopK 边界保存输入与原始输出，离线校验或独立调用 CUDA 算子，不需要回放整个模型。实现依据：`rtp_llm/utils/tensor_capture.py`、`rtp_llm/utils/tensor_capture_replay.py`、`rtp_llm/server/server_args/profile_debug_logging_group_args.py`。

## 1. 已实现范围与部署前提

- 默认关闭；仅 `indexer.topk`、指定层、**纯 eager prefill，非 CP、非 microbatch、非 MTP**。支持 `dsv4_per_row` 和 `topk_v3_tie_break`；不采集 MQA、decode、混合 prefill/decode、warmup、fake、verify/draft。
- **owner-only**：只在 record owner（TP rank 0）采集，不向 TP peer 传播采集身份，不生成 peer 样本。使用 C++ ExecutionRecorder 冻结的 batch 身份，不在 Python 中猜请求归属。
- `MtpExecutor` 的 target 或 draft 配置启用采集会直接报错：`tensor capture is not supported by MtpExecutor; use non-speculative eager prefill`；不是静默跳过。
- **不强制关闭 CUDA graph，也不强制 fallback**。graph 仍开启时，只有正常执行路径自然 fallback 到 eager 且满足其他条件的调用才可能采集；graph capture/replay 不采集。需要确定获得样本时，在已验证的 canary 启动配置中显式选择非 graph eager 路径。
- **必须同步重编译并部署 Python、C++ executor、配置/输入 pybind 和 CUDA 算子库**。只替换 Python 不够；旧 bindings 可能没有 `tensor_capture_config` 或采集 context 字段。

激活数据即使请求 ID 已匿名化仍然敏感。`authorized_capture=true` 表示操作者已授权**整个 batch/chunk 的所有请求**，不是逐请求/逐行授权过滤器；禁止让未授权流量混入实例。文件只写本地，目录/文件分别采用 0700/0600；仍需部署侧限制访问、传输和保留期。

## 2. 配置与启动参数片段

以下 JSON 可复制为 `/data0/luhuanbing.lhb/RTP-LLM/local-lab/topk_capture_config.json`。**此路径仅为示例，本指南没有创建该文件**；`layers=[0]` 也需改为实际执行 Indexer TopK 的层。未知字段、非正整数 limits、缺少授权或 selector 都会被拒绝。

```json
{
  "enabled": true,
  "authorized_capture": true,
  "layers": [0],
  "ops": ["indexer.topk"],
  "phases": ["prefill"],
  "rank_filter": "record_owner",
  "max_calls": 2,
  "max_bundle_bytes": 268435456,
  "max_total_bytes": 536870912,
  "max_inflight_bytes": 2147483648
}
```

在现有、已验证的模型启动环境中设置以下变量；**不是完整模型启动命令**，不包含 checkpoint、并行度等部署参数：

```sh
export RTP_LLM_RECORD_DIR=/data0/luhuanbing.lhb/RTP-LLM/local-lab/topk-capture-records
export RTP_LLM_RECORD_SESSION=topk-canary-001
```

向现有启动命令追加这个参数片段（不要单独当作 shell 命令运行）：

```text
--tensor_capture_config /data0/luhuanbing.lhb/RTP-LLM/local-lab/topk_capture_config.json
```

也可使用 `RTP_LLM_TENSOR_CAPTURE_CONFIG` 环境变量传配置路径，CLI 优先。`RTP_LLM_RECORD_DIR` 必须是绝对路径，不能经过符号链接或含 `..`；session 使用简单安全标识符并与 ExecutionRecorder 保持一致。启用后 owner 必须成功初始化 ExecutionRecorder，否则启动报错。默认空配置路径不启用；显式配置 `{"enabled": false}` 也可关闭。

### 预算怎么算

设 chunk 有 `R` 行、logits 有 `C` 列、TopK 为 `K`：

- raw payload `P = 4*R*C + 8*R + 4*R*K`（FP32 logits、两个 INT32 行区间、INT32 indices）。
- 实际预留 `B = P + identity JSON 字节数 + parameters JSON 字节数 + 4096`，即包含 **manifest 保守估算**，不是只算 `.bin`。
- 单次要满足 `B <= max_bundle_bytes`、剩余 total budget 足够，并且 **`max_inflight_bytes >= 4*B`**；并发未写完的 bundle 要累计。4 倍覆盖快照、D2H/连续化临时区、编码和 JSON；出队不立即释放，必须等持有的存储释放。
- 上例单 bundle 上限 256 MiB、总量 512 MiB、在途 2 GiB，足以覆盖两个达到单 bundle 预算上限的调用。它不是任意长上下文都适用的配置：例如 `R=1024,C=32768,K=2048` 的 raw 约 136 MiB，再加 manifest 估算可容纳；应按真实 chunk/层的形状核算。
- writer/loader 目前有 **单 bundle 所有 raw tensor 合计 1 GiB 的硬上限**，不是每个 bin 1 GiB；提高配置不能绕过。replay 重建原始 stride/offset 的 storage 另有合计 1 GiB 限制。
- `max_calls` 是**每个 recorder/model 实例**成功预留过的调用数，跨层/chunk 累计，不是每层、每请求或整个服务的限额；预留后的取消也不返还次数。多个实例预算分别计算。
- total budget 保守计费，已发布和已尝试写盘的失败调用仍保留额度；写前取消可退字节额度。预算不含 `run.json`、文件系统开销、Python/Torch allocator 开销，不能当作进程 RSS/磁盘总额保证。

同步 producer-stream D2H 有意阻塞采集路径；后台写盘有界，超预算/队列满跳过，不截断 bundle。不要据此测线上无采集性能。

## 3. 文件与回放合同

实际目录（尖括号为运行时值，不是字面目录名）：

```text
RTP_LLM_RECORD_DIR/
  owner-<ExecutionRecorder owner_id>/
    tensor-capture/
      sink-<每个 recorder 实例的 UUID hex>/
        run.json
        call-00000000/
          logits.bin
          row_starts.bin
          row_ends.bin
          indices.bin
          manifest.json
```

没有额外的 session 目录；session 在 metadata 中。owner 由 ExecutionRecorder 生成，当前为 PID-时间戳；sink 按实例隔离，call 是实例内预留序号，失败可能留下编号空洞。只消费 `complete=true` 的 `manifest.json`：它最后原子发布，孤立 `.bin`/`.tmp` 不代表有效样本。

- `.bin` 为 little-endian、逻辑连续 row-major 数据；manifest 保存 dtype/shape、原始 stride/storage offset、字节数、SHA-256、backend 参数、execution/batch/chunk 映射。
- **有效区间 `[row_starts[row], row_ends[row])` 保留原始数值位；区间外的无效域仅在 CPU 快照中置零**，不修改真实算子输入。回放/参考计算**仍必须按区间 mask**，不能把这些 0 当候选（合法分数可能全是负数）。
- `indices` 是 TopK 后处理前的原始输出：相对 row start 的索引，`-1` padding；不是加 offset、排序之后的全局索引。
- 当前源码合同为 FP32 bit-order key 降序、同 key 时相对索引升序选择；长行输出顺序不保证。短行按 `0..length-1` 输出再补 `-1`。V3 有效候选长度为 `min(row_end-row_start, max_seq_len)`，K 仅支持 512/1024/2048。正负零有区别；有效域 NaN 校验器明确不支持，Inf 可支持，采集本身不预筛非有限值。
- `run.json` 记录配置/hash、Python/Torch/CUDA build 版本等，GPU metadata 仅在 CUDA 已初始化时尝试记录。**`code_version`、`model_provenance` 当前为 null，不包含 checkpoint 来源**。部署侧必须额外保留 checkpoint 标识/修订、源码 commit、实际加载 `.so` 路径及 hash、构建配置、启动配置与 GPU 信息；不能仅凭算子名/时间戳推断二进制来源。
- SHA-256 检查完整性，不提供真实性认证；同时改写 payload 和 manifest 无法被这种校验发现。

从仓库根目录运行，下例路径是展示 CLI 用法的**未生成示例**；必须替换成实际完整 manifest 路径：

```sh
/opt/conda310/bin/python3 -m rtp_llm.utils.tensor_capture_replay /data0/luhuanbing.lhb/RTP-LLM/local-lab/example-topk-bundle/manifest.json
/opt/conda310/bin/python3 -m rtp_llm.utils.tensor_capture_replay /data0/luhuanbing.lhb/RTP-LLM/local-lab/example-topk-bundle/manifest.json --device cuda
```

CPU 默认只校验文件、mask、索引范围/唯一性/padding 和当前源码选择合同，输出 `mode=cpu_contract_validation, kernel_replayed=false`，**不等于 CUDA 验证通过**。CUDA 模式先校验捕获结果，再恢复原始布局调用真实 backend，比较选中集合及短行顺序；`raw_order_equal=false` 不必然是失败。读取无需 capture 授权配置，不加载模型权重。退出码：0 成功，1 无效/失败，2 不支持，3 CUDA/真实算子不可用。

## 4. 测试与真实模型验收

以下均在 `/data0/luhuanbing.lhb/RTP-LLM/github-opensource` 运行，使用已安装匹配 PyTorch 的 `/opt/conda310/bin/python3`。CPU 测试无需权重/新编译扩展；Indexer orchestration 测试用 mock kernel，不能证明 CUDA 数值正确。未 opt-in 的 CUDA 项会 skip。

```sh
/opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_test -v
/opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_replay_test.TensorCaptureReplayTest -v
/opt/conda310/bin/python3 rtp_llm/models_py/modules/base/cuda/test/indexer_tensor_capture_test.py -v
```

Indexer CPU 测试应直接运行文件：用 `-m unittest rtp_llm.models_py...` 会先加载父包，可能在测试的 mock 生效前因缺少 `rtp_kernel` 失败；这不是 CUDA 算子验证失败。

GPU synthetic opt-ins（非真实模型验证）：

```sh
RTP_LLM_TEST_TENSOR_CAPTURE_CUDA=1 /opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_test.TensorCaptureTest.test_cuda_nondefault_stream_noncontiguous_owned_snapshot -v
RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1 /opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_replay_test.TensorCaptureReplayCudaTest -v
# 单独检查 forced-radix
RTP_LLM_TEST_CAPTURE_REPLAY_CUDA=1 /opt/conda310/bin/python3 -m unittest rtp_llm.utils.test.tensor_capture_replay_test.TensorCaptureReplayCudaTest.test_dsv4_forced_radix -v
```

**已有验证记录的边界：**CUDA synthetic forced-radix 已实测通过；insertion 在旧 loaded `.so` 上表现为不同 tie 语义，V3 在已加载库中缺少导出。它们不能据此归因为“当前源码 kernel 坏了”，也不能宣称全部 CUDA backend 已通过。C++ 尚未完整 build/验证：当前仓库 `.bazelrc` 有 `cuda13`，未定义 `sm120`，且现有 bindings 较旧。以上是既有验证状态，不是本文重新执行 GPU/C++ 验证的声明。

换测试机器后，先确认其 Bazel rc/工具链同时提供 `cuda13`、`sm120`，再执行有界测试；缺配置应报告并修复测试环境，**不要偷偷换成 cuda12、sm9x/sm10x 或删除 sm120 参数**。配置就绪后可从以下集成目标开始（尚未在此环境通过）：

```sh
bazelisk test --config=cuda13 --config=sm120 --test_timeout=300 \
  //rtp_llm/server/server_args/test:server_args_test \
  //rtp_llm/cpp/normal_engine/test:recording_engine_test \
  //rtp_llm/cpp/normal_engine/speculative/test:mtp_executor_test
```

真实模型验收顺序：

1. 先同步构建/部署新 C++、bindings、Python 和算子库；记录实际库来源及 checkpoint，不只检查源码目录。通过配置字段/pickle 与 executor 集成测试后再启动模型。
2. 选独立、全部流量已授权的非 MTP canary，关闭 CP/microbatch；为确定取样选择 eager，层号必须实际执行 TopK。使用真实部署参数，不照抄不明来源的完整模型启动命令。按本地 E2E 规程在模型切换前完成 page-cache 预热。
3. 设置上面的 record 环境和 JSON 路径，发少量已授权纯 prefill 请求；覆盖短/长于 K、prefix/cache 与多请求 batch 等实际路径。不要为得到样本修改 kernel/backend 的选择。
4. 正常结束实例并检查 `Tensor capture close complete=... stats=...`，确认 writer 已完成；看 `selected/completed/skipped/io_error/writer_error/bytes_published` 及 `skip_reasons`，不要只看目录存在。`bytes_published` 只计完整 bundle 的 tensor+manifest。
5. 找到对应 owner/sink 下完整 manifest，核对 layer/backend、execution、chunk 与请求映射；CPU 校验后再做 CUDA replay。保留命令、退出码、JSON summary 和部署来源；重复采样时确认多个实例的预算不会放大到不可接受水平。

### 无样本或缺库如何定位

- 无样本：先检查层号/真实 TopK 分支、纯 prefill、owner、graph/CP/microbatch/MTP 与 recorder 初始化。结合 `layer_filter`、`graph`、`cp`、`microbatch`、`missing_context`、`invalid_context` 等实际 skip 原因；执行根本未进入采集点时不保证每次都有 skip 统计。
- `oversize`/`inflight_budget`/`total_budget`/`payload_limit`：按 B 和 4 倍规则重算；不要靠突破硬上限解决。`io_error`/`writer_error`：查目录权限、可用空间和初始化/关闭日志，不消费半成品。
- `unavailable`/退出 3、`lib*.so: cannot open shared object file` 或 undefined symbol：先检查 CUDA 可用性、`rtp_llm.ops` 实际导入位置、实际 `.so` 的 `ldd` 缺项及 ABI/运行库版本，再修复匹配的库搜索路径/安装包。不要用 mock、CPU fallback 或 skip 冒充 CUDA 通过。
- 缺 `topk_v3_tie_break` 导出或 `tensor_capture_config` 字段：检查旧 wheel/`.so` 遮蔽源码；同步重编译、部署 bindings 和算子库，而不是仅换 Python。insertion tie mismatch 先对照采集端与回放端的源码/二进制来源；CPU 当前源码合同不承诺兼容任意旧 kernel 的 arrival-slot tie 语义。
