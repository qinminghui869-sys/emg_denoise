# emg-denoise

[![CI](https://github.com/qinminghui869-sys/emg_denoise/actions/workflows/ci.yml/badge.svg)](https://github.com/qinminghui869-sys/emg_denoise/actions/workflows/ci.yml)

EMG 时基重建与去噪工具，以及浏览器对照页。当前版本 0.3.0。文件去噪默认离线零相位；`RealtimeDenoiser` 仍为因果流式。

**三项能力**

1. EMG 时间戳重建（tick 量化 → 双锚点分段插值）
2. EMG 去噪：高通 20 Hz → 低通 450 Hz → 梳状 50 Hz（−3 dB 带宽 2 Hz，Q=25）
3. 可视化：EMG Viewer

> 本仓库不含采集驱动。文件 `process` 与浏览页 EMG 默认手动链路同为离线 filtfilt（镜像、双向 20/450 Hz、周期对齐、周期延拓、双向梳状、端点消振）。去直流只影响浏览页的显示和频谱，不写入去噪文件。`RealtimeDenoiser` 为因果流式。本机回放不能证明端到端延迟。

[架构说明](docs/architecture.md) · [处理链路 DAG](dag.html) · [开发规范](CONTRIBUTING.md) · [变更记录](CHANGELOG.md)

---

## 安装

需要 Python ≥ 3.10。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'   # 含 parquet / 测试 / lint
```

CLI 入口：`emg-denoise`（短别名 `eed`）。若提示找不到命令，用：

```bash
.venv/bin/eed --help
```

---

## 仓库结构

```text
emg_denoise/
├── src/emg_denoise/
│   ├── cli.py              # 命令行（可读摘要；--json 输出完整 JSON）
│   ├── io.py               # CSV/JSON 校验与原子写入
│   ├── dsp/                # 滤波与去噪
│   │   ├── filters.py      # EMG 带通与 50 Hz 梳状（因果积木）
│   │   ├── offline.py      # 文件离线 filtfilt
│   │   └── core.py         # RealtimeDenoiser（流式因果）
│   ├── pipeline/           # 离线两阶段流水线
│   │   ├── timebase.py     # tick → 双锚点插值算法
│   │   ├── retime.py       # 阶段一
│   │   ├── replay.py       # 阶段二（默认 offline）
│   │   ├── batch.py        # 批量编排
│   │   └── meta.py         # 写入 / 回填 meta/info.json
│   └── qc/                 # 只读质量检查
├── tests/
├── viz/                    # 浏览页；experimental_emg.js 仅预览
├── docs/                   # 架构说明与历史核查记录
├── dag.html                # 处理链路图
└── .github/workflows/ci.yml
```

---

## 处理链路

| 路径 | 采样率 | 处理 |
|---|---:|---|
| `process` 默认 / 浏览页 EMG 手动默认 | 约 2000 Hz | 镜像 → 双向高通 20 Hz（二阶）→ 双向低通 450 Hz（四阶）→ 周期对齐 → 周期延拓 → 双向梳状 50 Hz（带宽 2 Hz，单向 Q=25）→ 端点消振 |
| `--causal` / `RealtimeDenoiser` | 约 2000 Hz | 同一截止的因果高通、低通和梳状；建议块长 20 点，预热 1 s |

文件名含 `eeg` 的记录会被拒绝。`--modality` 只接受 `emg`。

**通用 CLI 选项**

| 选项 | 作用 |
|---|---|
| （默认） | 打印可读摘要 |
| `--json` | 标准输出打印完整 JSON |
| `--sidecar` | 额外写 `.timebase.json` / `.report.json` |
| `--no-update-meta` | 不修改会话 `meta/info.json` |

诊断默认写入 LeRobot 会话的 `meta/info.json`（原 feature 增加 `timebase`，并注册 `<feature>.denoised`）。批量总览写 `batch.json`。

---

## 阶段一：时间戳重建

```bash
eed retime observation.left_wrist_emg.parquet --output retimed.parquet
eed retime observation.left_wrist_emg.parquet --inplace
```

默认 `--method anchor_interpolation`。**只改时间戳，信号值不变。**

- 相邻校正点之间固定两端时间，按两点间的样本数均分。校正点是约两个 tick 的跳变（容差 ±0.05 tick）。间距不固定，可以是 9、18 或其他数量。
- 第一个校正点之前、最后一个校正点之后保持原时间戳。不足两个校正点时，整个文件保持原样。
- 局部周期偏离标称超过 20%，或区间内出现超过三个标称周期的间隔时，该区间保持原样并告警。

`--method uniform` 是旧的固定周期模式；`--allow-negative` 只对它生效。`--inplace`：源文件改名为 `*.bak`，重建后的时间轴写回原名。`--json` 与 `--sidecar` 列出校正点、已均分区间和跳过区间。

---

## 阶段二：去噪

```python
import numpy as np
from emg_denoise import DenoiserConfig, RealtimeDenoiser

emg = RealtimeDenoiser(DenoiserConfig(channels=2))

for block in emg.process(np.zeros((20, 2)), np.arange(20) / 2000.0):
    waveform, ready = block.samples, block.valid
```

```bash
# parquet：默认在同目录写出 <stem>.denoised.parquet
# 默认离线链路见下文；不去直流
eed process observation.left_wrist_emg.parquet

# 需要流式因果结果时：
eed process observation.left_wrist_emg.parquet --causal

# CSV；可显式指定输出
eed process retimed.csv --output denoised.csv
```

默认离线链路按下面的顺序处理每个足够长的分段。文件结果不去直流；浏览页的「去直流」只减去显示和频谱里的通道均值，不代替 20 Hz 高通，也不写入去噪文件。

1. **镜像延拓。** 整段偶对称反射后再滤波，然后裁回原长度，减轻边界不连续引起的瞬态。
2. **20 Hz 二阶 Butterworth 高通。** 衰减低频基线漂移及部分运动伪迹，保留主要表面肌电。Butterworth 通带幅频响应最大平坦。
3. **450 Hz 四阶 Butterworth 低通。** 衰减带外高频噪声，限定分析频带。它不能消除采样或降采样时已经产生的混叠。
4. **双向滤波（`filtfilt`）。** 高通、低通和后面的梳状都先正向、再反向。内部区间合成响应为零相位。幅频响应是单向响应的平方，等效阶数加倍，截止频率处的衰减也随之增加。
5. **周期对齐。** 比较相邻工频周期，把不像周期的端部改写成内侧干净周期。这一步直接改写边界样本，可能影响真实瞬态。
6. **周期延拓。** 按 50 Hz 周期重复首尾片段，给梳状滤波提供边界数据，减轻初始化瞬态。效果取决于这些片段对周期性干扰的代表性。
7. **50 Hz 梳状，带宽 2 Hz。** 在 50 Hz 及其整数倍附近形成周期性陷波。单向设计满足 Q = 50 / 2 = 25。双向处理后陷波进一步加深、加宽，并同时削弱对应频段内的真实肌电。
8. **端点消振。** 若端部能量明显高于内部并逐渐衰减，就向内部反射交叉淡化。这是启发式处理，可能同时衰减真实的肌肉激活。

间隔超过三个标称周期时分段。段内时间戳不均匀时，先映到内部均匀网格上滤波，再映回原时间戳，样本数不变。短于 32 点的分段不滤波，计入 `unfiltered_samples` 并标记 invalid；走内部网格的段数记为 `resampled_segments`。`--causal` 仍要求均匀时基，并且不做镜像、`filtfilt`、周期对齐、周期延拓和端点消振。

输出信号列仅为 `timestamp` + 通道（parquet 另保留 `episode_index` 等元数据）；**不含** `valid`/`segment_id`，以免上游按列数读成多出的伪通道。逐样本掩码仅在 `--sidecar` 时写入旁路 `*.mask.parquet`。采样率优先 `--rate`，否则读 `meta/info.json` 的 `fps`，再否则测时间戳。去噪后会写入 `episodes_stats.jsonl` 的派生特征统计。文件去噪默认 **离线零相位**；Python `RealtimeDenoiser` API 仍为因果流式。

---

## 批量处理

推荐对采集会话就地重建 + 去噪：

```bash
eed batch \
  '/path/to/bcigo/L3-*/data/chunk-000/episode_000000/observation.*_wrist_emg.parquet' \
  --inplace --denoise --modality emg
```

每个文件产物：

| 文件 | 含义 |
|---|---|
| `observation.*_wrist_emg.parquet` | 分段重建时基（已覆盖） |
| `observation.*_wrist_emg.parquet.bak` | 原始备份 |
| `observation.*_wrist_emg.denoised.parquet` | 去噪结果 |
| `meta/info.json` | 写入 `timebase` 与 `*.denoised` feature |

镜像输出（不改原数据）：

```bash
eed batch '数据集/**/observation.*_wrist_emg.parquet' --output-dir out/ --denoise
```

已有 `.bak` 时需加 `--force` 才能再次 `--inplace`。

**参考吞吐（8 进程，按文件并行；i7-14700KF / 62 GiB RAM）**

| 阶段 | 吞吐（h 数据 / h 墙钟） | 峰值内存增量 |
|---|---:|---:|
| 仅重建时基 | 5700 | ~7.8 GiB |
| 仅去噪 | 1510 | ~10.6 GiB |
| 重建 + 去噪 | 1120 | ~11 GiB |

> 当前 `eed batch` 为单进程；上表为同 API 的多进程实测。单进程「重建 + 去噪」约 370 h/h、峰值约 0.15 GiB。

---

## meta/info.json

就地处理后会更新会话 `meta/info.json`，例如：

- `features.observation.left_wrist_emg.timebase`：重建诊断（方法、测得速率、tick、首尾是否保持原样、告警、备份名等）
- `features.observation.left_wrist_emg.denoised`：去噪 feature（`source`、`pipeline`、valid 样本数等）
- `episodes_stats.jsonl`：为每个 `*.denoised` 特征写入独立的 min/max/mean/std/count

若目录里还留着旧的 sidecar，可回填并删除：

```bash
eed meta-backfill /path/to/L3-002-2026-10-19-17-49-37
# 保留 sidecar：加 --keep-sidecars
```

---

## 实时接口约束

| 规则 | 说明 |
|---|---|
| 采样 | 2000 Hz（可经 `sample_rate_hz` / `--rate` 覆盖，须仍接近 2 kHz）；单次调用不可跨缺样区 |
| 分块 | 建议 20 点；更长输入内部按块切分 |
| 断流 | 向前时间跳变重置状态并开新分段；倒退/重复拒绝 |
| 预热 | 启动/重置后前 1 s 标记 `valid=False`，滤波仍持续 |
| 无效输入 | 在状态更新前拒绝，避免 NaN 污染 |

详情见 [架构说明](docs/architecture.md)。

---

## 可视化

页面：[`viz/viewer.html`](viz/viewer.html)。

```bash
python3 -m http.server 8765 --directory viz
# 浏览器打开 http://127.0.0.1:8765/viewer.html
```

在页面中「打开文件夹」，选择 episode 目录，例如：

```text
.../L3-002-2026-10-19-17-49-37/data/chunk-000/episode_000000
```

| 入口 | 行为 |
|---|---|
| 打开文件夹 | 自动读取左右腕 EMG；也可选 `*.denoised.parquet` |
| 打开 Parquet | 单文件或多选腕部 EMG |
| JSON/CSV | 示例 JSON 或带 `timestamp` 列的 CSV |

浏览页 EMG 默认手动链路与上面的文件 `process` 相同。去直流只影响波形和频谱，不进入去噪文件。梳状的 filtfilt、周期对齐、周期延拓、端点消振，以及陷波的对应项，仍可单独关掉。实验 FIR 与 `--causal` / `RealtimeDenoiser` 不是这条链路。

---

## 开发

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
make check
```

| 路径 | 职责 |
|---|---|
| `dsp/` | 因果积木、离线 filtfilt、流式 `RealtimeDenoiser` |
| `pipeline/` | 时基重建、文件去噪、批量、meta 更新 |
| `io` / `cli` | 校验、原子写入、命令行 |
| `viz/` | 离线波形浏览器 |

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 快速肌电 QC（只读）

```bash
# 会话或 episode 目录：自动发现 EMG，关联 .bak 和 .denoised
eed qc /path/to/session --output-dir artifacts/emg-qc
# 文件或 glob；终端摘要可切换为完整 JSON
eed qc '/path/to/data/**/*_emg.parquet' --json
eed qc /path/to/session --require-denoised --strict
```

支持 Parquet/CSV；不修改信号、时间轴或 meta。检查全量数据的非有限值、恒值、时间顺序、疑似间隙、采样率，以及 2 秒窗/1 秒步的 RMS 和频谱。对齐成功后比较去噪前后的工频、非工频带内能量与首尾 RMS。缺备份、处理方法或因果 mask 时明确标记未验证，不推断处理成功；现存备份不保证是首次采集原件。

`--rate` 覆盖 feature.fps，否则默认 2000 Hz；`--window-s`、`--step-s` 调整窗口。目录/glob 扫描排除派生与 mask 文件。显式文件名无需含 emg；目录只发现文件名含 emg 的信号。

`--output-dir` 写入 `summary.json`（完整结果）、`channels.csv`（通道指标与对照）和 `findings.csv`（问题及样本/时间区间，样本区间右端不包含）。默认只打印摘要，`--json` 打印完整结果。退出码 `0` 表示无错误（仍可能有未验证项或告警）；`1` 表示错误或 strict 下的告警；`2` 表示检查未完成/参数错误。单文件读取失败不会中止整批。

可用 `--config thresholds.json` 覆盖下列门限；未列出的键会报错：

```json
{
  "rate_warn_fraction": 0.01,
  "rate_error_fraction": 0.05,
  "gap_periods": 3.0,
  "uniform_tolerance_s": 0.000001,
  "flatline_s": 1.0,
  "rms_mad_multiplier": 8.0,
  "line_fraction_warn": 0.20,
  "adc_min": null,
  "adc_max": null,
  "clipping_fraction_warn": 0.01
}
```

门限属于初始工程策略，须经人工标注校准。ADC 上下限必须同时给出，单位与输入一致；未提供时不判断削顶。工频指标是 20–450 Hz 内各 50 Hz 谐波 ±2 Hz 的能量占比，窗内 Welch 使用最长 1 秒的子段；低频为 <20 Hz，高频为 >450 Hz。不可用频带、零功率和短片段的指标写 null，不能当作零污染。原始量化时间轴按标称采样率估算频谱，不执行插值；间隙只标为疑似断流，不估计丢样数。谱变化并不代表真实 SNR 或肌电保真度。
