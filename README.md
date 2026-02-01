# LangChain 中文注释项目

<p align="center">
  <a href="https://github.com/xt765/LangChain-Chinese-Comment">
    <img src="https://img.shields.io/badge/GitHub-LangChain--Chinese--Comment-blue?logo=github" alt="GitHub">
  </a>
  <a href="https://gitee.com/xt765/LangChain-Chinese-Comment">
    <img src="https://img.shields.io/badge/Gitee-LangChain--Chinese--Comment-red?logo=gitee" alt="Gitee">
  </a>
  <a href="https://blog.csdn.net/Yunyi_Chi">
    <img src="https://img.shields.io/badge/CSDN-玄同765-blue?logo=c&logoColor=white" alt="CSDN">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-green?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-1.2.7-orange" alt="LangChain">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

## 项目简介

本项目是一个专为中文开发者打造的 LangChain 框架源码注释与文档库。LangChain 作为业界领先的大语言模型（LLM）应用开发框架，其架构设计精妙、功能丰富，但源码复杂度较高。本项目通过系统性地整理核心模块的中文注释，帮助开发者深入理解 LangChain 的实现原理、设计思想及最佳实践。

## 项目特色

- **源码与注释对照**：`langchain_code/` 目录包含原始源码，`code_comment/` 目录提供对应的中文注释，两者目录结构完全一致，便于对照学习
- **模块化组织**：按照 LangChain 官方包结构组织，涵盖核心库、经典组件、合作伙伴集成等多个层面
- **详细中文文档**：包含 API 参考、快速入门、学习指南、技术栈分析等辅助文档
- **版本对比**：提供 LangChain 各版本之间的差异分析

## 环境要求

- **Python**: 3.10 或更高版本
- **LangChain**: 1.2.7（本项目同步自该版本）
- **推荐依赖**: `pydantic`, `typing_extensions`, `asyncio`, `langchain-core`

## 项目结构

```
langchain_code_comment/
├── langchain_code/              # LangChain 官方源码镜像
│   └── libs/
│       ├── core/               # langchain-core: 核心抽象层
│       ├── langchain/          # langchain: 经典组件实现
│       ├── langchain_v1/       # langchain v1.x: 主应用层
│       ├── partners/           # 合作伙伴集成包
│       ├── text-splitters/     # 文本分割器
│       ├── standard-tests/     # 标准测试库
│       └── model-profiles/     # 模型配置文件
│
├── code_comment/               # 中文注释文档（与源码结构对应）
│   ├── libs/
│   │   ├── core/              # 核心模块注释
│   │   ├── langchain/         # 经典组件注释
│   │   ├── langchain_v1/      # v1.x 版本注释
│   │   ├── partners/          # 合作伙伴包注释
│   │   ├── text-splitters/    # 文本分割器注释
│   │   └── standard-tests/    # 测试库注释
│   ├── core_modules/          # 核心模块分析文档
│   └── VERSION_COMPARISON.md  # 版本对比文档
│
├── docs/                       # 辅助文档
│   ├── api_reference/         # API 参考文档
│   ├── learning_guide/        # 学习指南
│   ├── overview/              # 项目概览
│   ├── technical_analysis/    # 技术分析
│   └── usage_examples/        # 使用示例与最佳实践
│
├── README.md                   # 项目说明（本文件）
├── TERMINOLOGY.md             # 术语对照表
└── LICENSE                    # 许可证
```

## 核心模块说明

### 1. langchain-core（核心抽象层）

位于 `code_comment/libs/core/langchain_core/`，包含 LangChain 最基础的核心概念：

| 模块 | 说明 |
|------|------|
| `runnables/` | 可运行对象，LCEL 表达式语言的基础单元 |
| `prompts/` | 提示词模板系统 |
| `messages/` | 消息类型定义（System、Human、AI、Tool） |
| `tools/` | 工具定义与转换 |
| `callbacks/` | 回调机制 |
| `tracers/` | 追踪与监控 |
| `documents/` | 文档数据结构 |
| `embeddings/` | 嵌入模型接口 |
| `vectorstores/` | 向量存储接口 |
| `indexing/` | 索引系统 |

### 2. langchain（经典组件）

位于 `code_comment/libs/langchain/`，包含传统 LangChain 组件：

| 模块 | 说明 |
|------|------|
| `langchain_classic/` | 经典 API 与工具 |
| `vectorstores/` | 各类向量数据库实现（50+ 种） |
| `retrievers/` | 检索器实现 |
| `runnables/` | 可运行对象扩展 |

### 3. langchain_v1（v1.x 主应用层）

位于 `code_comment/libs/langchain_v1/`，包含新版 LangChain 功能：

| 模块 | 说明 |
|------|------|
| `agents/` | 智能体系统（含中间件架构） |
| `chat_models/` | 聊天模型接口 |
| `embeddings/` | 嵌入模型 |
| `tools/` | 工具定义 |
| `messages/` | 消息处理 |

### 4. Partners（合作伙伴集成）

位于 `code_comment/libs/partners/`，包含各大 AI 服务商的集成：

- `anthropic/` - Anthropic Claude 模型
- `openai/` - OpenAI GPT 系列
- `deepseek/` - DeepSeek 模型
- `groq/` - Groq 推理平台
- `mistralai/` - Mistral AI
- `xai/` - xAI Grok
- `fireworks/` - Fireworks AI
- `chroma/` - Chroma 向量数据库
- `qdrant/` - Qdrant 向量数据库
- `exa/` - Exa 搜索 API
- `perplexity/` - Perplexity API
- `nomic/` - Nomic AI
- `ollama/` - Ollama 本地模型
- `prompty/` - Microsoft Prompty

### 5. Text Splitters（文本分割器）

位于 `code_comment/libs/text-splitters/`，提供多种文本分割策略：

- 字符分割（Character）
- Markdown 分割
- HTML 分割
- JSON 分割
- Python 代码分割
- LaTeX 分割
- NLTK 句子分割
- SpaCy 分割
- Sentence Transformers 分割
- Konlpy（韩语）分割

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-repo/langchain_code_comment.git
cd langchain_code_comment
```

### 2. 安装依赖

```bash
pip install langchain langchain-core
```

### 3. 阅读注释文档

所有注释文档位于 `code_comment/` 目录下，与源码 `langchain_code/` 保持完全一致的目录结构。

### 4. 学习路径建议

1. **入门阶段**：阅读 `docs/learning_guide/quick_start.md`
2. **核心概念**：学习 `code_comment/libs/core/langchain_core/` 下的核心模块
3. **进阶应用**：探索 `code_comment/libs/langchain_v1/` 的智能体和工具系统
4. **实战演练**：参考 `docs/usage_examples/` 中的最佳实践

## 使用示例

### 示例 1：LCEL 基础链

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 定义提示词模板
prompt = PromptTemplate.from_template("讲一个关于 {topic} 的笑话")

# 定义处理逻辑
add_prefix = RunnableLambda(lambda x: f"AI 助手回答：\n{x}")

# 组合成链
chain = prompt | add_prefix

# 调用
print(chain.invoke({"topic": "程序员"}))
```

### 示例 2：Agent 智能体

```python
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="用于搜索信息"
    )
]

# 创建智能体
llm = ChatOpenAI()
agent = create_openai_functions_agent(llm, tools)

# 执行
result = agent.invoke({"input": "今天天气如何？"})
```

## 文档导航

### 学习指南
- [快速入门](docs/learning_guide/quick_start.md) - 5 分钟上手 LangChain
- [学习路径推荐](docs/learning_guide/learning_path_recommendations.md) - 系统化学习建议

### API 参考
- [API 参考总览](docs/api_reference/api_reference.md) - 核心 API 说明
- [关键函数 API](docs/api_reference/key_functions_api.md) - 常用函数详解

### 技术分析
- [技术栈与依赖分析](docs/technical_analysis/technology_stack_dependencies.md) - 项目依赖关系
- [版本对比](code_comment/VERSION_COMPARISON.md) - 各版本差异

### 使用示例
- [最佳实践](docs/usage_examples/usage_examples_best_practices.md) - 推荐用法
- [常见问题](docs/usage_examples/faq.md) - FAQ 解答

## 贡献指南

我们欢迎社区开发者参与本项目的完善！

### 注释规范
- 每个函数/类必须包含功能描述、参数说明、返回值说明
- 复杂逻辑需要添加实现原理说明
- 提供使用示例代码

### 术语规范
- 优先使用行业标准翻译
- 参考 [TERMINOLOGY.md](TERMINOLOGY.md) 中的术语对照表
- 保持术语一致性

### 代码规范
- 遵循 PEP 8 编码规范
- 使用类型注解
- 添加必要的错误处理

## 相关资源

### 官方资源
- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Core GitHub](https://github.com/langchain-ai/langchain-core)

### 本项目链接
- [GitHub 仓库](https://github.com/xt765/LangChain-Chinese-Comment) - 本项目源码托管地址
- [Gitee 镜像](https://gitee.com/xt765/LangChain-Chinese-Comment) - 国内访问更快的镜像
- [CSDN 博客](https://blog.csdn.net/Yunyi_Chi) - 作者技术博客，分享更多 AI 开发经验

## 许可证

本项目遵循 [MIT License](LICENSE) 开源协议。

## 更新日志

- **2026-02-01**: 完善项目文档，更新 README 和术语表
- **2026-01-29**: 初始化术语表
- **2026-01-28**: 项目初始化，完成核心模块注释

---

> **提示**：本项目仅供学习参考，生产环境使用请参考 LangChain 官方文档和最佳实践。
