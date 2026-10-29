# Qwen-Agent 科研学习助手

基于 Qwen-Agent 的科研与学习助手，支持本地资料检索、原生 Codex JSONL 增量索引、日期筛选、分页原文读取，以及长期记忆保存、检索和软遗忘。

## 组成

- Qwen-Agent `Assistant`：模型调用与工具执行循环。
- `qwen_agent_live.py`：历史资料工具和记忆工具的编排适配。
- `qwen_records_bridge.py`、`local_documents.py`、`codex_chat_index.py`：资料索引、范围约束、分页检索与原文读取。
- `qwen_memory_bridge.py`：复用原项目记忆写入、SQLite 检索和软遗忘组件。
- `qwen_agent_demo.py`：Streamlit 界面，默认本机端口 8504。

## 安装运行

使用 Python 3.12，SQLite 需要支持 FTS5。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，修改模型接口和模型名：

```dotenv
ASSISTANT_BASE_URL=http://127.0.0.1:18080/v1
ASSISTANT_MODEL=Qwen3.6-27B
ASSISTANT_API_KEY=local
```

接口须提供 OpenAI 兼容的聊天与原生工具调用，支持 `chat_template_kwargs.enable_thinking`。当前默认关闭思考。模型文件另行准备，仓库不包含权重。

如模型运行在远程服务器，可先建立 SSH 隧道，保持该终端运行：

```bash
ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 -p YOUR_SSH_PORT USER@YOUR_SERVER
```

当前连接器仅允许本机回环地址；通过 SSH 隧道访问远端模型。模型服务和工具解析能力需要与你使用的推理后端匹配。

启动界面：

```bash
python -m streamlit run qwen_agent_demo.py --server.address 127.0.0.1 --server.port 8504 --client.toolbarMode viewer
```

Windows 也可运行 `./start.ps1`。打开 http://127.0.0.1:8504/，从左侧导入资料目录或原始 `.jsonl` 文件。首次运行不需要原助手预先创建数据库。

## 资料与记忆

- 默认查询所选索引目录的全部历史；可选择文件、今天或指定日期。
- 资料在本机分块、索引，模型按需调用目录、搜索和原文读取工具；分页候选不等于全部历史已阅读。
- 输入“记住：……”或“保存记忆：……”可指定保存；左侧也提供保存、查看和软遗忘操作。
- 记忆由当前用户和项目范围隔离，偏好与相关记忆进入回答上下文，普通聊天不会自动写入。
- 数据保存在 `local_assistant_data/`，相关问题、摘录和记忆会发送到配置的模型服务；请仅导入希望用于问答的资料。
- 不提供任意命令执行和任意文件写入工具。工具结果与记录正文作为数据处理。

## 验证

```bash
python verify_qwen_memory_bridge.py
python verify_release.py
```

测试使用临时资料与记忆库，不包含个人聊天或真实记忆。测试不要求模型在线，验证保存、检索、软遗忘、跨实例持久化、资料索引与范围约束。

## 复用来源及当前边界

记忆组件的 Python 源码依赖位于 `vendor/memory_core/`。`MEMORY_PROJECT_ROOT` 可指定其他兼容的原项目路径。

当前使用规则记忆写入器、特征哈希检索和 SQLite 后端，未启用语义冲突组织器。回答中的引用提示不等同于事实正确性验证；长记录覆盖、数值归属与多轮上下文仍需要评测。此仓库不包含个人数据库、资料快照、模型、服务器凭据或原项目全部功能。
