# 开发与贡献

## 环境

运行时支持Python 3.10–3.13。开发和严格类型检查使用Python 3.12，以匹配当前NumPy类型定义。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make check
```

不激活环境时可运行 `make check PYTHON=.venv/bin/python`。`make format`会修复导入并格式化；`make check`只检查，不修改源码。覆盖率门槛为85%，包含分支覆盖率。

## 代码约定

- 使用`src/`布局，公共接口写明输入形状、单位、时间戳和状态语义。
- 用NumPy类型别名和只读数据协议表达模块边界；数据加载和报告序列化边界允许显式`Any`。
- 核心算法不依赖命令行、文件系统或全局日志配置。文件和参数错误在命令行边界转为清晰消息及退出码。
- 不逐窗口归一化肌电，不改写原采样时间，不在流式实现中调用`filtfilt`。
- 命名优先清楚的业务含义，注释解释约束和原因，避免重复复述代码。

## 验证要求

改变滤波或时基算法时，补充与 SciPy 对照、任意分块一致性、异常输入不污染状态、断流与路径冲突等回归测试。CI 验证功能、数值正确性、类型及打包；本机性能观测不作为共享 CI 硬实时承诺。

## 提交与审查

提交信息采用 `type: concrete change`，如 `fix: preserve replay output when CSV parsing fails`。PR说明问题、最终行为和验证结果。修改公共接口时同步README及架构说明。

本地 `.venv/`、`artifacts/`、`.bench_tmp/`、缓存、测试覆盖率文件与密钥均不提交。仓库保留验证报告快照和可复现命令，不保留大体积生成数据。项目未附授权许可证；贡献前请确认你有权提交相应代码或数据。
