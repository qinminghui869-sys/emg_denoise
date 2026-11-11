# 架构与接口约束

## 三项能力

1. **EMG 时间戳重建**：采集 tick 量化时间戳 → 双锚点分段插值
2. **EMG 去噪**：20 Hz 高通 → 450 Hz 低通 → 50 Hz 梳状（BW=2 Hz，Q=25），2 kHz
3. **可视化**：`viz/viewer.html` 离线浏览（EMG 默认与文件去噪同链；去直流只用于显示和频谱）

## 包布局

```text
emg_denoise/
  dsp/         filters / offline / RealtimeDenoiser
  pipeline/    retime / replay / batch / timebase / meta
  qc/          只读质量检查
  io.py        边界 I/O
  cli.py       命令行
```

## 数据流

```text
采集文件（parquet/CSV：tick量化时间戳 + 通道）
    ↓ pipeline.retime：估计tick、定位2-tick候选锚点、固定双端点均分时间
    → 分段重建时基文件；诊断合并进 meta/info.json（可选 .timebase.json）
```

```text
时间戳递增的 CSV/parquet（timestamp + 通道）
    ↓ pipeline.replay（默认）：dsp.offline（镜像 → 双向 20/450 Hz → 周期对齐 → 周期延拓 → 双向梳状 → 端点消振；不去直流）
    ↓ 可选 --causal：RealtimeDenoiser 流式因果
    → 同目录 <stem>.denoised.*（信号列 = timestamp + 通道 + 元数据；不含 valid/segment_id）
    → 注册 features.<key>.denoised，并写入 episodes_stats 中该派生特征的统计
    → 可选 --sidecar：*.report.json + *.mask.parquet（valid/segment_id）
```

- **文件去噪**（`process` / `batch --denoise`）与浏览页 EMG 默认手动链路对齐：离线零相位。去直流只属于浏览页的显示和频谱。
- **实时 API** `RealtimeDenoiser` 仍为因果 `sosfilt`（可加 `--causal` 让文件路径与之对齐）。
- 信号 parquet 不得把 `valid`/`segment_id` 写成伪通道；上游 LeRobot 读取器会把非 timestamp/episode_index 列全部拼成信号。
- 离线 EMG 按超过三个标称周期的间隔分段；段内非均匀时基通过内部均匀网格线性插值、滤波、映回处理，保存的时间戳和样本数不变。短于 32 点的分段保留原信号并标记 invalid。因果模式仍要求均匀时基。

### 阶段一（EMG 默认 `anchor_interpolation`）

约两个 tick 的跳变（容差 ±0.05 tick）是候选校正点，更大的缺口不是校正点。相邻校正点 `(a, t_a)`、`(b, t_b)` 之间固定两端，内部按 `t_i = t_a + (i-a)*(t_b-t_a)/(b-a)` 均分。间距可以是 9、18 或其他样本数。只改内部时间戳；校正点本身、未被双端点覆盖的首尾保持原样。不足两个校正点时整个文件不变。局部周期偏离标称超过 20%，或区间内有超过三个标称周期的间隔时，该区间保持原样并告警。信号值逐位不变，也不做降采样。

`uniform` 为显式旧模式；`stitch`、`non_negative` 只对旧模式有效。命令行和批处理默认使用 `anchor_interpolation`。

### 阶段二

固定滤波链，不去直流。离线 EMG 在超过三个标称周期的间隔处分段；段内时间戳不均匀时，映到内部均匀网格上滤波后再映回，保存的时间戳和样本数不变。短于 32 点的分段保留原信号并标记 invalid。因果模式仍要求均匀时基，并且不做镜像、`filtfilt`、周期对齐、周期延拓和端点消振。

| 采样率 | 顺序 | 作用 |
|---|---|---|
| 约 2000 Hz | 镜像 → 双向高通 20 Hz（二阶 Butterworth）→ 双向低通 450 Hz（四阶 Butterworth）→ 周期对齐 → 周期延拓 → 双向梳状 50 Hz（带宽 2 Hz，单向 Q=25）→ 端点消振 | 去慢漂移和带外噪声，并抑制工频及其谐波。低通不能消除已经发生的混叠。双向滤波零相位，并使截止和陷波更深、更宽。周期对齐会改写端部样本；端点消振是启发式的，也可能衰减真实激活。 |

## 模块职责

| 模块 | 职责 | 不承担的职责 |
|---|---|---|
| `dsp.filters` | EMG 固定 DSP、梳状、输入矩阵校验 | 数据采集、文件读写 |
| `dsp.offline` | 文件离线 filtfilt（镜像、周期对齐、周期延拓、端点消振；不去直流） | 流式状态机 |
| `dsp.core` | 会话配置、因果处理、原时间戳、预热与断流 | 命令行、设备驱动 |
| `pipeline.timebase` | tick估计、候选锚点定位、分段时间轴重建 | 读写文件、修改样本值 |
| `pipeline.retime` | 阶段一：读采集文件、写分段时基 | 滤波、信号重采样 |
| `pipeline.replay` | 阶段二文件去噪（默认 offline，可选 causal） | 参数调优 UI |
| `pipeline.batch` | 展开glob、镜像/就地编排两阶段 | 单文件算法细节 |
| `pipeline.meta` | 合并诊断进 `meta/info.json`、sidecar 回填 | DSP 计算 |
| `qc` | 只读发现、窗口指标、报告 | 修改信号、时基或 meta |
| `io` / `cli` | 校验、原子写入、可读摘要 / `--json` | 内嵌滤波系数选择 |
| `viz/` | 离线波形浏览器 | 实时设备驱动、写回去噪文件 |

## 关键约束

- 阶段一只重建时间轴，样本值逐位不变。约两个 tick 的跳变只作为候选校正点；首尾未覆盖部分、不足两个校正点的文件、异常区间和疑似断流都保持原样。
- `modality` 在一次会话内只读。`reset()`清空信号/时间状态并重新预热（因果 API）。
- 因果路径：每次输入先完整校验再改动状态；向前缺样开启新分段；倒退/重复时间被拒绝。
- 输入输出保持同一单位。波形时间戳是采样时刻，不能换成处理完成时间。
- 批量输出按输入相对公共父目录镜像，避免同名文件静默覆盖。
- 诊断默认写入 `meta/info.json`；`.timebase.json` / `.report.json` 仅在 `--sidecar` 时写出。
- 输出CSV及单个JSON文件各自原子替换；输入/输出/报告路径不得冲突。

## 时间和预热

- EMG：2 kHz，因果块长 20 点（10 ms），预热 1 s（仅 `--causal` / `RealtimeDenoiser`）
- 离线 filtfilt 路径无预热窗口；掩码仅出现在可选 `*.mask.parquet`，不进入信号 feature


## 离线浏览与文件去噪

`viz/viewer.html` 的 EMG 默认手动链路与文件 `process` 相同：镜像延拓 + 双向高通/低通，然后周期对齐、周期延拓、双向梳状和端点消振。去直流只减去显示和频谱中的通道均值，不写入去噪文件，也不代替 20 Hz 高通。陷波的 filtfilt、周期延拓、周期对齐、端点消振默认不执行。实时因果主链仅用于 `RealtimeDenoiser` 与 `--causal`；实验自适应 FIR 只在浏览页预览。

## EMG QC

`qc/` 是只读诊断入口，`QcOptions`/`run_qc` 返回含 `exit_code` 的严格 JSON 兼容字典。`discovery.py` 负责发现与宽容读取（保留异常值），`signal.py` 提供纯数组窗口和频谱指标，`report.py` 负责终端及原子 JSON/CSV 写入。QC 不调用 retime/denoise，不更新会话 meta。按文件顺序加载源、备份和派生；窗口禁止跨非有限值、时间倒退或疑似缺口。源数据、关联 sidecar、配置和 meta 路径参与输出冲突检查（含符号/硬链接别名）。因果比较要求 `causal_stream` 记录和 timestamp 对齐的布尔 valid mask；缺处理记录时不假定为离线。
