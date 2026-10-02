# CodeFlow

CodeFlow 是一个面向代码仓库的轻量本地 coding agent。它直接跑在终端里，先看当前工作区，再用一组受约束的工具去读文件、改文件、跑命令，并把新会话保存在本地 `.codeflow/` 目录里。

它更像一个能在仓库里持续工作的命令行助手，不是纯聊天窗口。你可以拿它做代码排查、测试修复、仓库分析，或者让它在当前项目里执行一次性的工程任务。

> **项目来源说明：** CodeFlow 基于获得上传许可的原始代码进行改造。本仓库统一使用 CodeFlow 品牌，并继续增强上下文管理、混合记忆检索、工具治理和任务验证能力。原始代码及第三方依赖仍按各自适用的购买约定和许可证处理；本仓库未附加统一开源许可证。

## 适合做什么

- 在本地仓库里排查测试失败
- 读取当前代码结构并给出修改建议
- 基于现有文件做小步迭代，而不是脱离仓库空想
- 在会话中保留上下文，支持继续上一次工作

## 主要特性

- 发布包名是 `codeflow-agent`
- CLI 命令是 `codeflow`
- 主 Python 导入路径为 `codeflow`；旧的 `pico` 导入路径和 `Pico` 类名继续作为兼容别名
- 新会话保存在 `.codeflow/sessions/`
- 每次运行的工件保存在 `.codeflow/runs/<run_id>/`
- `Relevant memory` 使用中文语义向量与 BM25 召回历史轮次摘要，再以 RRF 合并，最多注入三条
- 支持四类模型后端：
  - Ollama
  - OpenAI 兼容 Responses API
  - Anthropic 兼容 Messages API
  - DeepSeek Anthropic 兼容 API

运行 `uv run codeflow --help` 可以查看当前 CLI 帮助和可用参数。

## 安装

需要 Python 3.10+ 和 Node.js 20.11+。`run_shell` 默认通过 Anthropic Sandbox Runtime（SRT）执行。

如果你用 `uv`，直接安装依赖：

```bash
uv sync
```

如果你已经在自己的 Python 环境里工作，也可以直接装成可编辑模式：

```bash
pip install -e .
```

安装锁定版本的 SRT：

```bash
npm install
```

Windows 第一次使用还要以管理员权限初始化一次（会出现一次 UAC 提示）：

```powershell
npx @anthropic-ai/sandbox-runtime windows-install
```

## 快速开始

在当前仓库里启动交互模式。默认 provider 是 DeepSeek：

```bash
uv run codeflow
```

指定另一个工作目录：

```bash
uv run codeflow --cwd /path/to/repo
```

直接跑一次性任务：

```bash
uv run codeflow "inspect the test failures and propose a fix"
```

旧模块入口仍保留，也可以直接这样启动：

```bash
python -m codeflow
```

## 模型后端

CodeFlow 启动时会读取项目根目录的 `.env`。本地真实 key 放在 `.env`，仓库只保留 `.env.example`。配置优先级是：

```text
显式 CLI 参数 > .env 里的 CODEFLOW_* 变量 > 旧环境变量 > 代码默认值
```

Provider 选择的具体顺序是：

```text
--provider > CODEFLOW_PROVIDER > 旧 PICO_PROVIDER > 代码默认 deepseek
```

不传 `--provider` 且没有 `CODEFLOW_PROVIDER` 或旧 `PICO_PROVIDER` 时默认使用 `deepseek`。这是推荐配置路径：DeepSeek 的 Anthropic-compatible endpoint 比本地 Ollama 更少依赖本机模型环境，也比 OpenAI-compatible/Anthropic-compatible 代理少一层默认 gateway 假设。其他 provider 仍然保留，可以在 `.env` 里写 `CODEFLOW_PROVIDER=openai`、`CODEFLOW_PROVIDER=anthropic`、`CODEFLOW_PROVIDER=ollama`，也可以显式传 `--provider openai`、`--provider anthropic` 或 `--provider ollama`。

`.env` 会在构建 provider client 前加载，并覆盖当前进程里的同名环境变量。模型名和 base URL 可以通过 `--model`、`--base-url` 临时覆盖；API key 只从环境变量读取。

本地第一次配置：

```bash
cp .env.example .env
```

然后把要使用的 provider key 填进去。`.env` 已经被 `.gitignore` 忽略，不要提交真实 key。

### 推荐配置：DeepSeek

最小配置只需要 key：

```bash
CODEFLOW_DEEPSEEK_API_KEY="your-api-key"
```

默认模型和接口是：

```bash
CODEFLOW_DEEPSEEK_API_BASE="https://api.deepseek.com/anthropic"
CODEFLOW_DEEPSEEK_MODEL="deepseek-v4-pro"
```

所以常规情况下 `.env` 里只填 `CODEFLOW_DEEPSEEK_API_KEY` 就能直接启动：

```bash
uv run codeflow
```

如果你需要临时切模型或代理地址，不必改 `.env`，可以直接覆盖：

```bash
uv run codeflow --model deepseek-v4-pro --base-url https://api.deepseek.com/anthropic
```

DeepSeek 当前走 Anthropic-compatible Messages API，所以 runtime 里复用的是 Anthropic-compatible client；这只影响 HTTP 协议，不影响 CLI 用法。

### OpenAI provider：DeepSeek Flash

`--provider openai` 默认不再指向 right.codes，而是走 DeepSeek 的 OpenAI-compatible Responses API：

- base URL：`https://api.deepseek.com`
- model：`deepseek-v4-flash`
- API key：优先读取 `CODEFLOW_OPENAI_API_KEY`；为空时回退复用 `CODEFLOW_DEEPSEEK_API_KEY`

因此同一把 DeepSeek key 可以让两个 provider 分别使用两种协议和模型：

```bash
# Pro：DeepSeek provider + Anthropic-compatible Messages API
uv run codeflow --provider deepseek

# Flash：OpenAI provider + OpenAI-compatible Responses API
uv run codeflow --provider openai
```

### 可选配置：right.codes

right.codes 在 CodeFlow 里有两条可选 provider 路径：

- `--provider openai`：仍可通过环境变量覆盖为 right.codes 的 OpenAI-compatible `/responses`，例如 `https://www.right.codes/codex/v1`
- `--provider anthropic`：走 Anthropic-compatible `/messages`，默认 base URL 是 `https://www.right.codes/claude/v1`，默认模型是 `claude-sonnet-4-6`

如果 right.codes 给你的是一把共享 key，推荐只填这一项：

```bash
CODEFLOW_RIGHT_CODES_API_KEY="your-right-codes-key"
```

然后按需要选择 provider：

```bash
uv run codeflow --provider openai
uv run codeflow --provider anthropic
```

如果你想显式区分两条 provider 的 key，也可以分别配置：

```bash
CODEFLOW_OPENAI_API_KEY="your-right-codes-key-for-codex"
CODEFLOW_ANTHROPIC_API_KEY="your-right-codes-key-for-claude"
```

不要在 `.env` 里写 `CODEFLOW_OPENAI_API_KEY=$CODEFLOW_RIGHT_CODES_API_KEY` 这种 shell 展开形式；CodeFlow 的 `.env` 解析器只读取字面量，不展开变量引用。要么只写 `CODEFLOW_RIGHT_CODES_API_KEY`，要么把 key 字符串分别填到 provider-specific 变量里。

如果请求 right.codes 返回 `API Key额度不足`，说明协议和 endpoint 已经打通，但当前 key 没有可用额度；换一把有额度的 key，或到 right.codes 后台处理额度。

当前 provider 环境变量：

| provider | base URL | API key | model |
| --- | --- | --- | --- |
| `deepseek` | `CODEFLOW_DEEPSEEK_API_BASE`，回退 `DEEPSEEK_API_BASE`，默认 `https://api.deepseek.com/anthropic` | `CODEFLOW_DEEPSEEK_API_KEY`，回退 `DEEPSEEK_API_KEY` | `CODEFLOW_DEEPSEEK_MODEL`，回退 `DEEPSEEK_MODEL`，默认 `deepseek-v4-pro` |
| `openai` | `CODEFLOW_OPENAI_API_BASE`，回退 `OPENAI_API_BASE`，默认 `https://api.deepseek.com` | `CODEFLOW_OPENAI_API_KEY`，回退 `OPENAI_API_KEY`、`CODEFLOW_RIGHT_CODES_API_KEY`、`RIGHT_CODES_API_KEY`、`CODEFLOW_ANTHROPIC_API_KEY`、`ANTHROPIC_API_KEY`、`CODEFLOW_DEEPSEEK_API_KEY`、`DEEPSEEK_API_KEY` | `CODEFLOW_OPENAI_MODEL`，回退 `OPENAI_MODEL`，默认 `deepseek-v4-flash` |
| `anthropic` | `CODEFLOW_ANTHROPIC_API_BASE`，回退 `ANTHROPIC_API_BASE`，默认 `https://www.right.codes/claude/v1` | `CODEFLOW_ANTHROPIC_API_KEY`，回退 `ANTHROPIC_API_KEY`、`CODEFLOW_RIGHT_CODES_API_KEY`、`RIGHT_CODES_API_KEY`、`CODEFLOW_OPENAI_API_KEY`、`OPENAI_API_KEY` | `CODEFLOW_ANTHROPIC_MODEL`，回退 `ANTHROPIC_MODEL`，默认 `claude-sonnet-4-6` |
| `ollama` | `--host`，默认 `http://127.0.0.1:11434` | 不需要 | `--model`，默认 `qwen3.5:4b` |

如果有额外的敏感环境变量需要从 trace/report 里脱敏，可以用 `CODEFLOW_SECRET_ENV_NAMES` 配置逗号分隔的变量名，或启动时重复传 `--secret-env-name NAME`。

### OpenAI 兼容接口

如果要改用 OpenAI-compatible `/responses` 服务，显式传 `--provider openai`：

```bash
uv run codeflow --provider openai
```

默认 OpenAI 兼容接口使用 DeepSeek Flash endpoint：

```bash
CODEFLOW_OPENAI_API_BASE="https://api.deepseek.com"
CODEFLOW_DEEPSEEK_API_KEY="your-deepseek-key"
CODEFLOW_OPENAI_MODEL="deepseek-v4-flash"
```

也可以改成其他 OpenAI-compatible 服务；这时同时显式填写对应的 key：

```bash
CODEFLOW_OPENAI_API_BASE="https://your-api.example/v1"
CODEFLOW_OPENAI_API_KEY="your-api-key"
CODEFLOW_OPENAI_MODEL="gpt-5.4"
```

### 用自然语言切换模型

交互模式中可以直接切换当前运行实例使用的模型：

```text
切换到 flash
换成 pro
切换到 deepseek-v4.1-flash
换成 4.1 flash
使用快速模型
恢复默认模型
```

其中 `flash`、`deepseek-v4.1-flash` 和 `4.1 flash` 都会映射到官方 API 模型名 `deepseek-flash`，并复用 `CODEFLOW_DEEPSEEK_API_KEY` 与 Pro 相同的 DeepSeek Anthropic-compatible 路由。这样自然语言切换不再经过 OpenAI Responses 路径。

切换只影响后续请求，不会清空当前会话历史、工作区状态或运行工件。第一版只在当前进程内生效；重新启动 CodeFlow 后，仍按 `.env` 或启动参数选择 provider。

### Anthropic 兼容接口

如果要改用 Anthropic-compatible 服务，显式传 `--provider anthropic`：

```bash
uv run codeflow --provider anthropic
```

默认 Anthropic 兼容接口使用 right.codes 的 Claude endpoint：

```bash
CODEFLOW_ANTHROPIC_API_BASE="https://www.right.codes/claude/v1"
CODEFLOW_RIGHT_CODES_API_KEY="your-right-codes-key"
CODEFLOW_ANTHROPIC_MODEL="claude-sonnet-4-6"
```

如果你的服务端对多个兼容接口复用了同一套密钥，CodeFlow 也支持从 `CODEFLOW_ANTHROPIC_API_KEY` 回退到 `ANTHROPIC_API_KEY`、`CODEFLOW_RIGHT_CODES_API_KEY`、`RIGHT_CODES_API_KEY`、`CODEFLOW_OPENAI_API_KEY` 或 `OPENAI_API_KEY`。旧的 `PICO_*` 环境变量仍然有效。

### Ollama

如果要改用本地 Ollama，显式传 `--provider ollama`：

```bash
ollama serve
ollama pull qwen3.5:4b
uv run codeflow --provider ollama --model qwen3.5:4b
```

## 常用交互命令

- `/help`：查看内置命令
- `/memory`：查看提炼后的工作记忆
- `/session`：查看当前会话文件路径
- `/reset`：清空当前会话状态
- `/workspace <目录>`：切换当前工作区

## 相关记忆检索

每轮结束后，CodeFlow 将“用户目标”和“处理结果”保存为轮次摘要。下一轮提问时，系统分别运行中文语义向量检索和 BM25 关键词检索，用 RRF 融合两路排名，并把最多三条结果放入 `Relevant memory`。使用 FastEmbed 的 `BAAI/bge-small-zh-v1.5` 模型，第一次召回时需要下载模型；模型不可用时自动退回 TF-IDF 字符向量与 BM25。

旧环境变量（`PICO_*`）、`pico` Python 导入路径和 `.pico/` 状态目录仍受支持。新工作区默认使用 `.codeflow/`；包含已有 `.pico/` 状态的工作区会继续从原目录加载会话和记忆。
- `/exit` 或 `/quit`：退出 REPL

模型请求和工具执行期间，终端会显示 `CodeFlow 思考中` 动画。它只是等待状态提示，不会输出模型的内部思考内容。

也可以在 `CodeFlow>` 后，直接从 Windows 资源管理器拖入一个项目文件夹并按回车。CodeFlow 会切换到该目录，然后你可以继续用自然语言提出需求，例如：

```text
CodeFlow> "C:\Users\ZhuanZ1\Desktop\my-project"
工作区已切换到：C:\Users\ZhuanZ1\Desktop\my-project
CodeFlow> 给这个项目增加一个登录失败次数限制，并补充测试
```

拖入单个文件时，会自动把该文件所在目录作为工作区。CodeFlow 只会在当前工作区内读取和修改文件；涉及写文件或执行命令时，仍会遵循审批和沙箱规则。

## 安全与持久化

CodeFlow 不会默认把所有动作都放开。像 shell 执行、文件写入这类高风险操作，会受审批模式控制：

- `--approval ask`
- `--approval auto`
- `--approval never`

工具按能力分组，可在启动时禁用某类动作：`filesystem.read`（列出、读取和搜索）、`filesystem.write`（写入和补丁）、`process.execute`（Shell）和 `agent.delegate`（子 Agent）。例如 `codeflow --deny-capability process.execute --deny-capability filesystem.write` 会隐藏并拒绝这些能力；运行时直接调用底层工具也会再检查一次。

可以用 `--read-path src --read-path tests` 将文件读取和搜索限制在列出的工作区路径内，用 `--write-path src` 将写入和补丁限制在 `src`。不设置路径参数时，权限范围仍是整个工作区。传入工作区外路径会被拒绝。

`--approval ask` 会在执行高风险工具前显示工具名、能力、工作区、解析后的文件路径和参数。输入 `y` 仅批准当前调用；输入 `g` 会对同一工作区中的精确文件路径或 Shell 命令授予短期权限，默认 60 秒，可用 `--approval-grant-seconds` 调整。授权保存在当前进程内，不写入会话。

Shell 和全文搜索有执行超时；Shell 子进程的 stdout、stderr 分别最多保留 128 KiB，工具结果保留前 4,000 字符并附截断提示。读取单次最多 2,000 行，写入内容最多 1 MB。每次调用的 trace 记录能力、耗时、状态、错误码、变更路径和输出是否截断；较长参数在审计 trace 中会保留摘要和哈希，敏感值会脱敏。

`run_shell` 在审批通过后仍必须经过 SRT。默认配置位于 `srt-settings.json`：命令可读写当前工作区，但不能读 `.env`、`.codeflow` 或兼容旧状态的 `.pico`，不能写 `.git`、`.env`、这两个状态目录和沙箱配置本身，网络默认关闭。SRT 缺失、配置无效或系统隔离未初始化时，CodeFlow 会拒绝执行，不会自动退回普通 shell。

仅测试或排障时可显式使用 `--shell-backend direct`；它没有操作系统级隔离，不应作为日常运行方式。

每次运行结束后，都会在 `.codeflow/runs/<run_id>/` 下写出这些文件。已有 `.pico/` 状态的工作区继续沿用原目录：

- `task_state.json`
- `trace.jsonl`
- `report.json`

这些内容默认只保存在本地，不需要跟仓库一起提交。

## 开发

常用本地检查：

```bash
uv run pytest tests -q
uv run ruff check codeflow tests scripts
```

内部代码现在按较轻的边界拆分：`codeflow/evaluation/` 放 benchmark 和 metrics，`codeflow/providers/` 放模型 provider client，`codeflow/features/` 放可选运行时能力。新代码应直接使用这些包路径；旧的 `codeflow.evaluator`、`codeflow.metrics`、`codeflow.models` 和 `codeflow.memory` import 不再作为公共入口保留。
