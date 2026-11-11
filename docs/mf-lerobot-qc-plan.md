# MF-LeRobot 兼容性核查与 QC 脚本方案

核查日期：2026-11-06。样例会话：`L3-001-2026-10-09-17-53-45`。

> 本文是当时的只读核查记录，不是现行设计说明。2026-11-11 起，包名为 `emg-denoise`，EEG 去噪已移除，文件名含 `eeg` 的记录会被拒绝。信号 parquet 不再写入 `valid` / `segment_id`；下文「读成 10 通道」描述的是核查当时的导出，不是现行契约。建议中的 0.5/50 Hz EEG 链和 250 Hz EEG 采样率也不再对应代码。

本次只读核查样例和代码，不修改数据、不重跑就地处理。下述 `eed qc` 是拟议接口，尚未实现。结论只适用于已检查样例及代码路径，不代表全部 L3 数据。

## 1. 结论与判定口径

**当前产物不能判为完全满足目标保存要求：原有目录与 EMG 重建结果基本兼容，但新增 denoised 特征存在确定的读取维数错误。** EEG 样例没有去噪产物，且其 1000 Hz 采样率不适用于当前固定 250 Hz 的 EEG 处理链。还需补齐派生统计、处理后 QC 与处理溯源。

区分三个判定维度：

- `storage_compatibility`：目录、schema、meta 与目标读取器是否一致。当前 FAIL（去噪特征维数）。
- `processing_integrity`：样本、时基、滤波参数及派生关系是否正确。EMG 已检查项目通过；EEG 去噪未执行，不记为通过。
- `signal_quality`：通道质量、伪迹及同步可信度。当前待复核，原采集 QC 为 WARN；不能把滤波成功或 `valid=True` 当成信号质量合格。

依据：[采集项目打包说明](https://github.com/Koorye/Embodied-Brain-Collect#数据打包详解mf-lerobot)、[上游 Parquet 读取实现](https://github.com/Koorye/Multi-Frequency-LeRobot/blob/main/src/mf_lerobot/parquet.py)。目标说明强调各流原生频率独立存储、统一 RUN_START 原点、30 Hz 主时钟，以及随包携带采集与 QC 信息。核查时同时检查了本机 MF-LeRobot，commit 为 `b27215d6a8a5187dc9ec593b682fb3c40224bd37`；网页 main 会变化，正式 CI 应固定目标版本。

## 2. 样例实测

### 已满足的部分

- `meta/info.json`、tasks、episodes、episodes_stats、collect_info、qc_reports 均存在；一个 episode，主索引 1810 行，8 路视频。扫描了 30 个 parquet（包含主索引），均可用 PyArrow 读取，时间轴无重复、无回退。
- 主索引起点 0，终点约 60.300 秒；marker 241/242 对应 0 与 60.274456501 秒。30 Hz 末帧覆盖 RUN_END，不能将最后一帧比信号长约 25.5 ms 一律误判为掉流。
- 左右 EMG 各 120549 行、8 通道，均为 float32 信号、float64 时间戳。重建后的间隔为 0.5 ms，起点 0，终点约 60.274 秒。
- 与各自 `.parquet.bak` 逐列比较：信号值和现有 episode_index 完全一致，无删样或插值。左右最大时间改变量分别为 0.306263 ms、0.321714 ms；这是相对于旧时间轴的变化量，不是外部同步误差的测量值。
- 两份去噪结果均与重建输入等长、时间戳逐项一致；8 路信号均有限值，无全程恒值通道。`valid` 全为 true，`segment_id` 全为 0，与离线处理记录一致。
- EEG 为 60275 行、132 通道，实测约 1000 Hz；没有 NaN/Inf、没有全程恒值通道。没有 EEG `.denoised.parquet` 或对应 feature 注册，不能声称 EEG 已处理。
- 阻抗是 timestamp=0 的单行常量流，应豁免连续采样率与全时段覆盖检查。

### 阻断：去噪结果从 8 通道读成 10 通道

`src/emg_denoise/pipeline/replay.py` 的 `_write_denoised_parquet` 把 `valid`、`segment_id` 写进同一个特征 parquet；`pipeline/meta.py` 保持 shape=[8]。

目标 `ParquetFeature.load()` 只排除 timestamp、episode_index，其余列全部拼成信号。故实际选列为 dim_0…dim_7、valid、segment_id，总计 10 列，且掩码被转成 float32。该行为已通过文件 schema 与本机/在线上游代码共同确认；本次环境无 torch，未运行完整 Dataset 解码测试，不能将代码级复核描述成端到端运行通过。

建议采用兼容旧读取器的导出契约：信号 parquet 只含 timestamp、episode_index 与声明的信号列；逐样本 valid/segment_id 移至独立 `qc/` 附属 parquet，以 episode、feature、sample_index 关联并校验同长度与同时间戳。另一选择是修改并固定读取器，仅按 names/shape 选信号列、单独返回掩码；这要求消费端同步升级。不能简单将 shape 改为 10，否则掩码会作为肌电输入训练。

### 阻断处理能力：Neuroscan EEG 1000 Hz 与固定预设不匹配

`dsp/filters.py` 中 EEG_FS=250，`pipeline/replay.py` 用该预设分段和设计滤波器。样例约 1 ms 的间隔与预期 4 ms 不符，会被拆成单样本段。本次用前 20 行、2 通道只在内存执行，复现 `ValueError: The length of the input vector x must be greater than padlen, which is 21.`。

更严重的潜在误用是先以 250 Hz 对此 EEG 执行 retime：只改时间戳不会实现降采样，会把约 60 秒拉长为约 241 秒。不能通过该方式让输入“匹配”预设。

建议从 feature.fps 获取并交叉验证实际采样率，将采样率参数贯穿滤波器和分段器，原生保留 1000 Hz；按 names 区分 EEG 与 VEOG/HEOG/EKG/EMG 辅助通道。若训练另需 250 Hz，应另建明确记录抗混叠降采样方法的派生特征，保留原生流。

### 完整性缺口：统计和处理溯源

- `episodes_stats.jsonl` 不包含两个 denoised 特征。应独立计算其 min/max/mean/std/count，不能复制原信号统计；否则训练归一化缺信息。当前读取是否立即报错取决于消费路径，不能声称所有读取器都会报错。
- 原 EMG 值未改变，因此其信号统计不因 retime 自动失效；涉及时间范围、对齐检查的数据需要重新核查。
- `qc_reports.jsonl` 是原采集报告，整体 WARN，包含 EEG ch85 MAD 异常、头环抖动和部分运动信号告警，不是本次处理后的验收报告。ch85 不直接映射为当前列名，需先核实原始通道编号规则。
- 项目将 timebase/pipeline 写入 info.features，通常可被宽松读取器接受，但属于自定义扩展。推荐标准特征声明留在 info；逐 episode 的处理参数、输入输出哈希、版本、采样率、时间映射、质量结果写入 `meta/processing_info.jsonl` 与 `meta/processing_qc_reports.jsonl`。这是建议扩展，不冒称上游强制文件名。
- 多 episode 场景需逐 episode 记 provenance，不能用一个全局 feature.pipeline 的 samples/segments 表示所有 episode；部分处理成功时不得注册一个实际缺文件的全局派生特征。

### 原有媒体与对齐问题，不能归因于 EMG 去噪

ffprobe 容器元数据显示八路均为 30 fps，帧数分别与各自 timestamp parquet 行数一致。但左右腕各 1791 帧、时长 59.700 秒，第三视角 1792 帧、59.733333 秒；其时间戳跨度约 60.234、60.229、60.252 秒。头环各 1809 帧、60.300 秒；眼动视频 1808 帧、60.266667 秒。

本机视频读取器先查 timestamp parquet 找帧序号，再解码 idx/fps；这种路径可以保留时间戳到帧的对应关系。但按 MP4 播放时钟直接与主轴对齐，会产生不同结果。不能只因帧数相等就判“符合最新内容锚定截段要求”，也不能仅因视频帧数少于主索引就判文件损坏。应分别检查两种读取方式，结合原始录像或同步事件验证内容；本次仅检查容器头信息，未做全量解码、黑屏冻结或内容同步验证。

主索引 timestamp 实际为 float32，而 info 中声明 float64；该差异来自已有数据，建议统一并在 schema 检查报出。其最大 30 Hz 间隔量化误差约 3.31 μs，本样例没有观察到因此导致的排序问题。

## 3. QC 脚本设计

### 接口与范围

拟增加 `eed qc SESSION --profile mf-lerobot --output-dir REPORT_DIR`，默认只读、绝不自动 retime 或修复。支持 `--stage pre|post|all`、`--scope emg-eeg|all`、`--deep-media`、`--strict`、`--config`，以及 `--require-processed emg,eeg` 显式声明必须存在的派生结果。未要求处理 EEG 时，缺少 EEG 去噪属于 NOT_RUN，而不是保存结构错误。

批量入口以 session/meta/info.json 为发现依据；匹配文件时排除 .bak、报告和派生结果，防止重复滤波。按会话并行、同一会话顺序归并，避免 meta 写竞争；QC 本身不回写源 meta。

拟议模块：`qc/discovery.py` 识别 episode/feature；`qc/schema.py` 验证保存契约；`qc/timebase.py` 检查时基与窗口；`qc/processing.py` 检查派生关系；`qc/signal.py` 计算质量指标；`qc/media.py` 做可选媒体验证；`qc/report.py` 输出结果；CLI 只负责参数和退出码。

### A. 文件与 schema：硬性验收

以 info 中 data_path/index_path/chunks_size、episodes 清单定位文件，不能固定 chunk-000。检查 meta JSON/JSONL 可解析、episode 唯一、任务引用存在、总数一致、各 feature 的 episode 覆盖符合目标版本的 dropped 约定。

普通信号列从 names 或 shape 推导（dim_N 必须数值排序），核对列顺序、shape、dtype、null、NaN/Inf、episode_index。额外列通过白名单，且必须运行目标读取器列选择兼容性检查。对 video、audio、稀疏 marker、单行 impedance、主索引分别使用专用 schema；不能要求每类都有 dim_N，也不能把事件稀疏视为掉采样。

对所有新增训练特征检查统计是否存在、维数/count 是否一致，抽验或重算均值方差。缺文件、shape 错、通道顺序错、非有限信号默认 ERROR；统计缺失在 training profile 为 ERROR，在 archive profile 为 WARN。strict profile 对声明 dtype 与实际 dtype 不符判 ERROR。

### B. 时间轴与多模态覆盖

检查有限时间戳、非负、严格递增（事件同刻重复按事件 profile 配置），不要求每路首样本都恰好为 0。采样率用 (N−1)/(t_last−t_first) 和 dt 分位数共同描述；分段计算，防止平均速率掩盖缺口。

对已经声明均匀重建的 EMG，检查 abs(dt−1/fs)≤1 μs（与当前处理链容差一致）；对原始异步流使用设备 profile 的抖动/间隙标准，不强行均匀化。候选默认：有效采样率偏离标称 >1% WARN，>5% ERROR，仅适用于标称稳定的连续流，阈值需用设备基线校准。

对每个主时刻按实际 window 计算样本数、空窗、部分窗及覆盖比例；输出首尾边界与中间断流两类结果。原生流只需覆盖 RUN_START…RUN_END 的应有范围，主轴向上取整的末帧允许不足一主帧的尾部余量；不能把这一余量从真实中段缺口中扣掉。单行 impedance 使用常量语义；marker 不用普通连续流空窗规则。

retime 前后逐样本比较，输出偏移分位数、最大偏移、首尾偏移和线性漂移；建议初始门限 1 ms WARN、5 ms ERROR（工程策略，不是上游统一规定，且不应自动等同于 tolerance_s）。无原始时间戳时该项 NOT_VERIFIED。原始大缺口不能因重建成均匀轴就消失，应由 pre-QC 保留缺口与丢样证据。

### C. 处理正确性

retime 必须保持 N、通道顺序、数值与 episode 归属不变，并保存原时间轴的可追溯位置/哈希。检查输入实测 fs、meta fs 与滤波 fs 一致，在滤波前阻止本样例 1000→250 的误用。

离线 denoise 默认必须保持 N 和 timestamp 一致，验证 finite、信号 dtype、pipeline 方法；segment_id 应连续且只在真正断流处变化。causal 模式允许实现明确记录的未输出尾块，必须报告数量、原因和对应输入索引，不能静默丢尾。valid 的预热规则按方法验证，离线无预热不等同无边缘伪迹。

所有处理记录包含工具/git版本、输入输出 SHA256、实际采样率、通道选择、滤波器阶数/频率/Q、因果或零相位、padding、分段和丢样策略。高频窗口查询得到的数组应保持真正信号维数，valid 必须作为掩码单独消费。

### D. 信号质量：告警优先

逐通道、逐窗口（建议 2 秒窗、1 秒步）计算 flatline 持续时间、重复值比例、RMS、峰峰值、稳健异常比例、已知 ADC 轨道上的削顶比例、PSD、工频与谐波能量、带内能量。单位不明时不套用 μV/mV 绝对阈值，削顶判定缺 ADC 范围时标为不可确认。

分别报告原始/重建/去噪；同一有效区间比较 50 Hz 及 EMG 各谐波抑制量、有效带宽内保真和首尾能量变化。EEG 0.5–50 Hz 的指标排除辅助通道或另设 profile。没有干净参考时不报告“真实 SNR 提升”，只报告定义清楚的谱比值变化。动作期 EMG/手套的 MAD 异常不能自动当坏样本删除。

候选策略：连续恒值 ≥1秒 WARN，全程恒值 ERROR；抑制不足和异常过度衰减先 WARN，经人工标注批次校准后再升级门禁。离线零相位结果可用于离线分析，作为严格在线预测输入时需另评估未来信息泄漏。

### E. 消费端与媒体验证

固定 mf_lerobot commit，测试 Dataset 首帧、中段、末帧及每个缺口前后；验证 shape、样本时间范围、mask、stats 可用性，不仅测试 pq.read_table 能打开。CI 中加入 8 通道 + valid/segment_id 被错读成 10 通道的回归用例。

默认 ffprobe 核对媒体路径、编码信息、帧数、分辨率、音频采样率；深度模式完整解码计数并检查黑帧/冻结，audio 验证索引 num_samples 与实际样本数。同步结论分为时间戳一致、帧映射一致、内容锚定验证，缺原始参照时最后一项 NOT_VERIFIED。

### 报告与退出码

输出 `summary.json`、逐 episode/feature 的 `findings.jsonl`、`metrics.jsonl`、自包含 `report.html`。报告记录 schema_version、profile/阈值版本、目标读取器版本、检查时间、数据指纹、执行/跳过项；每条 finding 含 rule_id、severity、status、episode、feature、channel、时间/样本区间、observed、threshold、证据路径、建议动作。

规则状态用 PASS/FAIL/NOT_RUN/NOT_VERIFIED；严重度用 INFO/WARN/ERROR；总体结果不能因未执行项被默认置为 PASS。退出码建议：0=无 ERROR（可有 WARN），1=存在 ERROR 或 strict 下的 WARN，2=执行失败/依赖缺失导致要求的检查未完成。批量继续扫描并汇总失败会话。

### 实施顺序与验收

1. P0：实现只读 schema、采样率预检、derived shape、backup 对照、stats 覆盖；本样例应准确抓出 10≠8、1000≠250、派生统计缺失。时间变化约 0.32 ms 不应误判失败。
2. P1：修复信号文件与 QC 列隔离、可配置原生 EEG 采样率、按 episode 写处理记录、派生统计；随后在输出镜像上运行目标读取器测试，原数据保留。
3. P2：加入信号谱指标、HTML 复核、全媒体解码和同步内容验证；用人工标注样本校准告警阈值。

必要测试数据：单/多 episode、部分派生缺失、具名/无名通道、dim_10 排序、NaN、重复/倒退时间戳、真正缺口、量化抖动、负起点、单行阻抗、稀疏事件、1000 Hz EEG、因果尾块和视频帧映射。验收同时覆盖“能抓住真实问题”和“不会误报合法的异步频率/末帧余量”。
