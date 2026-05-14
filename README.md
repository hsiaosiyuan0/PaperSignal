# PaperSignal

**中文** · [English](README.en.md)

一个面向 LLM 的命令行工具：让模型**发现、下载、分析、发布** arXiv 论文，生成
自包含的双语研究简报 —— 浏览器里可读，Edge TTS 朗读，可直接部署到 GitHub Pages。

> [!TIP]
> **在线预览** → <https://hsiaosiyuan0.github.io/PaperSignal/>

Python 包名与 CLI 命令均为 `paperread`，项目名是 **PaperSignal**。

---

## 它能做什么

给定一个主题和目标机构，LLM 可以：

1. **检索** 该机构最近的论文（`paperread discover` → OpenAlex）
2. **下载** PDF（`paperread download` → arXiv，流式下载 + 重试）
3. **转换** 为 Markdown，保留 LaTeX（`paperread convert` → marker / markitdown）
4. **阅读** Markdown 后**撰写**分析（`report.en.md` + `report.zh.md`）
5. **构建** 发布页面（`paperread report build` → HTML + 句级 mp3）
6. **刷新** 归档索引（`paperread report index`）

推到 GitHub，Actions 自动部署到 Pages。

---

## 安装

```bash
git clone git@github.com:hsiaosiyuan0/PaperSignal.git
cd PaperSignal
uv sync              # 安装依赖并创建 .venv（首次会拉 torch + marker，约 5GB）
cp .env.example .env # 然后把 OpenAlex key 填进 .env
```

需要 Python 3.12+、[uv](https://github.com/astral-sh/uv)，以及（可选）
[OpenAlex 免费 API key](https://openalex.org/) —— 长期使用 `discover` 时需要。

---

## 快速开始

工作流的入口是 **Claude Code CLI**（或任意你常用的 coding agent）。你不需要
自己敲 `paperread` —— 用自然语言提需求，让 Claude 按需调起子命令、消费 JSON
输出、用自己的 Read/Write 写分析文档。

比如，你在 Claude Code 里说：

> 帮我找微软 2025 年关于知识检索的论文，挑一篇值得读的写成研究简报。

Claude 大致会跑这条链路（你看到的只是结果，命令是它自己调的）：

```bash
# 1. 拉候选清单，让 Claude 自己挑
uv run paperread discover "knowledge retrieval" \
  --affiliation Microsoft --year 2025- --top 10 --json

# 2. 选定 <arxiv_id> 后，下 PDF + 转 Markdown
uv run paperread convert <arxiv_id>

# 3. Claude 读全文 + 元数据
uv run paperread cache show <arxiv_id> --json

# 4. Claude 用自己的 Write 把双语分析落到这两个文件
#    （路径由 `paperread report path <arxiv_id>` 给出）
#      report.en.md
#      report.zh.md

# 5. 切句、Edge TTS、渲染 HTML
uv run paperread report build <arxiv_id>

# 6. 刷归档首页 + 浏览器查看
uv run paperread report index
uv run paperread report open <arxiv_id>
```

整条 CLI 是按 LLM 消费来设计的：所有检索/读取命令都支持 `--json`，输出稳定
的结构化数据；分析撰写直接交给 agent 的 Read/Write，不需要额外脚手架。

样本产出：[`reports/1706.03762v7/`](reports/1706.03762v7/) ——
《Attention Is All You Need》的研究简报。

---

## 命令一览

| 命令 | 用途 |
|---|---|
| `paperread discover QUERY` | 通过 OpenAlex 按机构 + 年份 + 排序检索 |
| `paperread search QUERY` | arXiv 原生搜索（不支持机构过滤） |
| `paperread info ID [--json]` | 单篇论文元数据 |
| `paperread download ID` | 流式下载 PDF，截断时自动重试 |
| `paperread convert ID [-b marker|markitdown]` | PDF → Markdown |
| `paperread cache list/search/show [--json]` | 查询本地 SQLite 缓存，LLM 友好 |
| `paperread report path ID [--lang en|zh]` | 分析 MD 应写入的目录 |
| `paperread report build ID [--no-audio]` | 渲染 HTML + 生成音频 |
| `paperread report open ID` | 浏览器打开 |
| `paperread report index` | 重建 `reports/index.html` |
| `paperread report pages-init` | 生成 `.nojekyll` + Actions workflow |

所有 search / list / show / discover 命令都支持 `--json`，便于机器消费。

---

## 配置

在 `.env`（已 gitignore）或 shell 中设置：

| 变量 | 用途 |
|---|---|
| `OPENALEX_API_KEY` | `discover` 超出免费额度后必需 |
| `PAPERREAD_REPORTS_DIR` | 报告目录根（默认 `./reports`） |
| `PAPERREAD_DATA_DIR` | PDF 存放目录（默认 `./papers`） |
| `PAPERREAD_CACHE_DIR` | SQLite 缓存路径（默认平台缓存目录） |
| `PAPERREAD_TTS_VOICE` | Edge TTS 音色（默认 `en-US-AvaMultilingualNeural`） |
| `TORCH_DEVICE` | 覆盖 marker 自动选择的设备（`mps` / `cuda` / `cpu`） |

---

## 部署到 GitHub Pages

```bash
uv run paperread report pages-init
git add -A && git commit -m "scaffold pages" && git push
```

然后在仓库 Settings → Pages → **Source: GitHub Actions**。每次 `reports/`
目录有改动，workflow 都会重新发布。

发布的目录结构按论文自包含：

```
reports/
  index.html                       ← 归档首页
  .nojekyll                        ← 保留 "_" 开头的路径
  <arxiv_id>/
    index.html                     ← 渲染好的报告
    report.en.md                   ← LLM 写的英文源
    report.zh.md                   ← LLM 写的中文源（可选）
    audio/sNNNN.mp3                ← 句级 Edge TTS
```

---

## 架构

```
src/paperread/
  arxiv_client.py     封装 arxiv Python 库；自定义 UA 规避限流；
                      流式 PDF 下载 + 重试（避免 urlretrieve 截断）
  openalex.py         /paper/search 端点，按 raw_affiliation_strings 过滤
  cache.py            SQLite 元数据缓存，自动迁移 schema（v3）
  converter.py        marker / markitdown 后端抽象，懒加载
  sentences.py        pysbd 英文断句，过滤碎片
  tts.py              edge-tts 异步封装，基于 manifest 增量重生成
  report.py           双语 HTML 渲染；按渲染后 HTML 的纯文本切句
                      使得跨 <strong>/行内公式的句子也能正确高亮
  cli.py              Typer 命令 + 给 LLM 消费的 JSON 契约
  paths.py            platformdirs + 环境变量覆盖
```

---

## 值得知道的设计选择

- **arxiv 请求加自定义 User-Agent** —— `arxiv.py/2.3.2` 被 arxiv.org
  严格限流，我们在 session 中覆盖了它写死的 UA。
- **选 OpenAlex 而非 Semantic Scholar** —— S2 申请 API key 需要学术/企业
  邮箱，机构覆盖率也稀疏（我们测试约 12%）。OpenAlex 把机构作为一等公民
  字段，任意邮箱注册即可用。
- **用 `/paper/search` 而非 `/paper/search/bulk`** —— bulk 不返回
  `authors.affiliations`；relevance 端点会返回。我们在客户端再排序。
- **默认用 marker 而非 markitdown 转换** —— 它保留 LaTeX，对 ML/CS 论文
  关键。markitdown 作为更快的回退保留。
- **在渲染后 HTML 的纯文本上切句** —— 不在源 MD 上。这样跨 `<strong>` /
  `$math$` 的句子会变成多个共享 `data-sentence-id` 的 `<span>` 片段，
  hover 时整句一起高亮。
- **选 Edge TTS，不用 OpenAI / Eleven Labs** —— 免费，无需 API key，音质够用。
- **"Editor's Desk" 视觉风格** —— 全局 Fraunces 可变衬线，骨白底色，
  单一氧化朱砂强调色，罗马数字章节序号。刻意不走 Inter + 紫色渐变那种
  AI 美学。

---

## 技术栈致谢

[arxiv.py](https://github.com/lukasschwab/arxiv.py) ·
[OpenAlex](https://openalex.org/) ·
[marker](https://github.com/datalab-to/marker) ·
[markitdown](https://github.com/microsoft/markitdown) ·
[edge-tts](https://github.com/rany2/edge-tts) ·
[pysbd](https://github.com/nipunsadvilkar/pySBD) ·
[markdown-it-py](https://github.com/executablebooks/markdown-it-py) ·
[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) ·
[Typer](https://typer.tiangolo.com/) ·
[Rich](https://github.com/Textualize/rich) ·
[Fraunces](https://fonts.google.com/specimen/Fraunces) ·
[JetBrains Mono](https://www.jetbrains.com/lp/mono/) ·
[KaTeX](https://katex.org/)

---

## 许可证

MIT（参见 `LICENSE` —— TBD）。

> **关于 marker**：依赖中的 `marker-pdf` 代码采用 GPL-3.0，模型用改版
> Open Rail-M license —— 免费用于科研、个人，以及营收 200 万美元以下的
> 创业公司。超过该阈值的商业自建部署需要从 Datalab 获得授权。
