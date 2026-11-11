# Changelog

## 0.3.0

### 包

- 项目定名为 `emg-denoise`（包 `emg_denoise`，CLI `emg-denoise` / `eed`）。`emg-eeg-denoise` 不再使用。
- 移除 EEG 去噪。文件名含 `eeg` 的记录会被拒绝；`--modality` 只接受 `emg`。

### 时基

- `retime` 默认 `anchor_interpolation`：在相邻校正点之间固定两端，并按实际样本数均分。首尾未覆盖、不足两个校正点、局部周期偏离标称超过 20%，或区间内出现超过三个标称周期的间隔时，保持原时间戳。信号值不变。`--method uniform` 仍是显式旧模式。
- 支持 parquet 与 CSV，可 `--inplace`（源文件改为 `*.bak`）或写出新文件。诊断默认写入会话 `meta/info.json`；`.timebase.json` 仅在 `--sidecar` 时写出。

### 文件去噪

- `process` / `batch --denoise` 默认离线零相位，顺序为：镜像延拓 → 双向 20 Hz 二阶 Butterworth 高通 → 双向 450 Hz 四阶 Butterworth 低通 → 周期对齐 → 周期延拓 → 双向 50 Hz 梳状（带宽 2 Hz，单向 Q=25）→ 端点消振。不去直流。
- 超过三个标称周期的间隔处分段。段内时间戳不均匀时，先映到内部均匀网格上滤波，再映回原时间戳，样本数不变。短于 32 点的分段保留原信号并标记 invalid。
- `--causal` 使用 `RealtimeDenoiser`，要求均匀时基，不做镜像、`filtfilt`、周期对齐、周期延拓和端点消振。
- 信号文件只保留 `timestamp`、通道和元数据，不写 `valid` / `segment_id`。逐样本掩码仅在 `--sidecar` 时写入 `*.mask.parquet`。
- 采样率优先 `--rate`，否则读 `meta/info.json` 的 `fps`，再否则测时间戳。去噪后向 `episodes_stats.jsonl` 写入派生特征统计。
- CLI 默认打印可读摘要；`--json` 输出完整 JSON。批量总览写 `batch.json`。

### 浏览与 QC

- 浏览页 EMG 默认手动链路与文件去噪相同。梳状的 filtfilt、周期对齐、周期延拓、端点消振可单独关闭；陷波的对应项默认关闭。去直流只影响显示和频谱。实验 FIR 只在浏览页预览。
- 新增只读 `eed qc`：通道质量、时间轴、备份一致性与去噪对照。不修改信号、时间轴或 meta。

## 0.1.0

- 实现 2 kHz 多通道因果肌电去噪与 50 Hz IIR 梳状陷波。
- 提供原时间戳、预热/断流状态、流式特征和 CSV 回放工具。
- 将命令行、数据 I/O、回放服务、质量指标与 DSP 核心分层。
- 增加严格类型检查、统一格式、分支覆盖率门槛、跨 Python 版本 CI 及打包检查。
- 防止文件路径别名覆盖输入，解析失败时保留已有输出。
