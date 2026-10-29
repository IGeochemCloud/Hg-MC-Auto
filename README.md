# ⭐ Hg-MC-Auto

> **Element-aware automation toolkit for isotope analysis workflows** — bilingual CLI, agent-facing MCP tooling, and cross-platform Python packaging.

![alt text](<docs/Graphic abstract.png>)

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/hg-mc-auto.svg?style=flat-square&color=blue)](https://pypi.org/project/hg-mc-auto/)
[![Python versions](https://img.shields.io/pypi/pyversions/hg-mc-auto.svg?style=flat-square&color=green)](https://pypi.org/project/hg-mc-auto/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![CI - Lint, Format & Test](https://img.shields.io/github/actions/workflow/status/IGeochemCloud/Hg-MC-Auto/ci.yml?style=flat-square&label=CI)](https://github.com/IGeochemCloud/Hg-MC-Auto/actions/workflows/ci.yml)
[![PyPI - Release](https://img.shields.io/github/actions/workflow/status/IGeochemCloud/Hg-MC-Auto/pypi-publish.yml?style=flat-square&label=PyPI%20Release)](https://github.com/IGeochemCloud/Hg-MC-Auto/actions/workflows/pypi-publish.yml)
[![GitHub stars](https://img.shields.io/github/stars/IGeochemCloud/Hg-MC-Auto?style=flat-square&color=orange)](https://github.com/IGeochemCloud/Hg-MC-Auto/stargazers)

</div>

- 📄 [A data-driven, post-acquisition quality diagnostic pipeline for isotope analysis by MC-ICP-MS](https://pubs.rsc.org/ja/article-abstract/41/5/1894/1230352/A-data-driven-post-acquisition-quality-diagnostic?redirectedFrom=fulltext)

---

## ✨ Quick summary

| | |
|---|---|
| 📦 **User-ready package** | `pip install hg-mc-auto` |
| 🖥️ **CLI entry points** | `hgmc` and `hg-mc-auto` |
| 🧪 **Supported elements** | `hg`, `fe`, `cd` |
| 🖥️ **Platforms** | Windows, Linux, macOS |
| 🚀 **Release flow** | GitHub Actions → PyPI (Trusted Publishing) |

---

# 👤 For Users

> This section is for people who want to **use** Hg-MC-Auto to process isotope
> data. You do not need to read the source code or understand the internals.

## 1) 🚀 Install and start

### Install

```bash
python -m venv hg-mc-auto
source hg-mc-auto/bin/activate
pip install --upgrade pip
pip install hg-mc-auto
```

On Windows PowerShell:

```powershell
python -m venv hg-mc-auto
hg-mc-auto\Scripts\activate
pip install --upgrade pip
pip install hg-mc-auto
```

### Start the app

```bash
hgmc
```

Or:

```bash
hg-mc-auto
```

### Common CLI commands

```bash
hgmc --help
hgmc list-elements
hgmc workflow --element fe
hgmc run --element fe --task export --language zh
hgmc language --lang en
hgmc platform
hgmc project-summary
hgmc guide --element hg --language zh
```

Use `hgmc list-elements` to check maturity. At present, Hg is the production
legacy workflow; Fe and Cd are scaffolds, not validated element-specific
analytical pipelines. The generic `run` command resolves task metadata and must
not be mistaken for executing a complete scientific workflow for every element.

### Task options

The Hg workflow supports five tasks, selected with `--task`:

| # | Task | What it does |
|---|------|--------------|
| 1 | `export` | Convert proprietary `.dat` files to structured CSV, merge with instrument log files (uses RPA for vendor software interaction) |
| 2 | `calculate` | Compute δ²⁰²Hg values relative to NIST SRM 3133, mass-independent fractionation anomalies (Δ-values), and batch-process entire datasets |
| 3 | `empirical` | Apply literature-based acceptance ranges (Table 1 in manuscript), flag measurements outside 95% confidence intervals; thresholds configurable via `custom_ranges_config.json` |
| 4 | `predict` | Use pre-trained ensemble models for quality assessment, provide confidence scores, and identify probable causes for abnormal measurements |
| 5 | `train` | Train laboratory-specific models on your annotated data; supports both binary and multi-class classification, adapting to different instrument performances and sample matrices |

### Structured QC and optional LLM interpretation

The QC assistant accepts structured metrics, results, and anomaly flags. It does
not send raw instrument files by default, and an LLM response is advisory only:
the result remains `pending_human_review`.

```bash
hgmc qc --input docs/qc_packet.example.json
```

For a near-real-time, read-only directory monitor, point the command at the
directory where the calculation or prediction workflow writes result files:

```bash
hgmc monitor --directory results --element hg --once
```

Remove `--once` to keep polling. The monitor emits a `qc.v1` packet when a JSON,
CSV, or Excel result file is created or modified. It does not change instrument
settings or automatically approve a recommendation.

To request an interpretation through an OpenAI-compatible provider such as
OrcaRouter, set the provider key first:

```bash
export ORCA_KEY="your-key"
hgmc qc --input docs/qc_packet.example.json --llm --language zh
```

The assistant can explain quality signals, rank plausible causes, and suggest
checks. It must not directly change instrument parameters or start an
instrument run.

### Install with MCP support

```bash
pip install "hg-mc-auto[mcp]"
hgmc mcp --action status
```

### Connect this checkout to VS Code Agent

When opening this repository as the VS Code workspace, it includes
`.vscode/mcp.json`. Run **MCP: List Servers**, select `hg-mc-auto`, then start
the server. In Chat, choose Agent mode and enable the server's tools. If VS Code
does not discover the configuration, run **Developer: Reload Window** and
inspect **MCP: Show Installed Servers**.

The workspace configuration uses the repository-local virtual environment. If
you moved or renamed that environment, update the command in `.vscode/mcp.json`.

Try asking the Agent: “Use Hg-MC-Auto MCP to list the supported elements and
explain the Hg workflow.” It should call `list_elements` and `resolve_workflow`.

Current tools provide metadata, workflow routing, plans, and structured QC
packet creation. In particular, `run_element_task` only resolves the target
script; it does not execute Evaluation GUI automation or control the instrument.

### Notes

- If you only need normal usage, `pip install hg-mc-auto` is enough.
- `hgmc` is the recommended command for end users.
- `python src/main.py` remains available for legacy local development workflows.

---

# 👨‍💻 For Developers

> This section is for contributors and maintainers who want to **build, test,
> extend, or release** Hg-MC-Auto. It assumes you are comfortable with Python,
> git, and the command line.

## 2) 🛠️ Set up the development environment

### Clone and set up the repo

```bash
git clone https://github.com/IGeochemCloud/Hg-MC-Auto.git
cd Hg-MC-Auto
python -m venv hg-mc-auto
source hg-mc-auto/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### Run tests

```bash
PYTHONPATH=. python -m unittest tests.test_main_cli -v
```

### CLI development

```bash
hgmc --help
hgmc list-elements
hgmc workflow --element fe
hgmc run --element cd --task export --language en
```

See [docs/ELEMENT_EXPANSION.md](docs/ELEMENT_EXPANSION.md) for reviewed
MC-ICP-MS method references, element support gates, and the proposed plugin
contract.

### MCP development

```bash
hgmc mcp --action status
hgmc mcp --action tools
hgmc mcp --action json
hgmc mcp --action stdio
```

The stdio action is intended for agent/IDE integrations. It is compatible with the installed `mcp` 2.x API and exposes the project’s tool surface over standard input/output.

The MCP layer can later expose the same structured QC packet and advisory
assessment to an agent. Keep the human approval step outside the model: an
agent may prepare a review, but a qualified operator confirms any action.

### Release flow

Publishing a new version is automated with a GitHub Actions workflow
(`.github/workflows/pypi-publish.yml`). When you publish a GitHub Release, the
workflow builds the distribution and uploads it to PyPI automatically using
**Trusted Publishing** (OIDC) — no PyPI token is stored in the repository.

#### One-time setup

1. **Configure Trusted Publishing on PyPI** (one time only):
   - Open https://pypi.org/manage/account/publishing/
   - Add a pending publisher with:
     - **PyPI project name**: `hg-mc-auto`
     - **Owner**: `IGeochemCloud`
     - **Repository**: `Hg-MC-Auto`
     - **Workflow name**: `pypi-publish.yml`
     - **Environment**: `pypi`
2. **Create the `pypi` environment** in GitHub (Settings → Environments) if you
   want deployment protection rules.

#### Publish a new version

```bash
# 1. Bump the version in pyproject.toml (e.g. 0.1.1)
# 2. Commit and push your changes
git add . && git commit -m "release 0.1.1"
git push
# 3. Tag and push (optional, but recommended)
git tag 0.1.1
git push origin 0.1.1
# 4. Create a GitHub Release (triggers the workflow)
gh release create 0.1.1 --title "v0.1.1" --notes "release 0.1.1"
```

Once the Release is published, the workflow builds and uploads the package to
PyPI automatically. You can watch progress under the **Actions** tab.

> **Important**: Do **not** also run `./release.sh` or `twine upload` for the
> same version — that would upload the package to PyPI twice. The GitHub
> Actions workflow is now the single source of truth for PyPI publishing.

#### Legacy local script

The `release.sh` script and the `hgmc-release` command remain available for
local validation and manual publishing, but they are **no longer the
recommended path** for PyPI uploads:

```bash
hgmc-release check   # validate the local environment
hgmc-release build   # build dist/ locally
hgmc-release publish # manual twine upload (requires PYPI_TOKEN)
```

If you use `hgmc-release publish`, set `PYPI_TOKEN` first. Create the token at
https://pypi.org/manage/account/token/ and use `__token__` as the username.

### OrcaRouter open-source maintainer application

The project can apply through [OrcaRouter Built With](https://www.orcarouter.ai/zh-CN/built-with).
The page currently states that the repository must be public, owned by the
applicant, and contain commits from the applicant's GitHub account; approval is
manual. Listing is attribution and support, not an endorsement or an exclusive
provider agreement. Other model providers can remain supported.

Before applying, publish the repository and make the project description clear:
structured MC-ICP-MS QC, bilingual workflow support, MCP integration, and
human-confirmed recommendations. Do not include API keys, raw laboratory data,
or private instrument credentials in the repository.

---

## 3) 📁 Project structure

![alt text](<docs/Overall Architecture Diagram.png>)

```bash
Hg-MC-Auto/
├── src/
│   ├── main.py
│   ├── hg_mc_auto/
│   │   ├── cli.py
│   │   ├── mcp_server.py
│   │   ├── release.py
│   │   ├── core/
│   │   └── ...
│   └── ...
├── tests/
├── docs/
├── data/
├── model/
├── results/
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4) 🖥️ Platform notes

### Windows

```powershell
python -m venv hg-mc-auto
hg-mc-auto\Scripts\activate
pip install -e .
hgmc --help
```

### Linux

```bash
python3 -m venv hg-mc-auto
source hg-mc-auto/bin/activate
pip install -e .
hgmc --help
```

### macOS

```bash
python3 -m venv hg-mc-auto
source hg-mc-auto/bin/activate
pip install -e .
hgmc --help
```

---

## 5) 🚦 Release status

This project is currently in a release-oriented v0.1.0 state with:

- bilingual CLI support
- generalized element routing for `hg`, `fe`, and `cd`
- agent-oriented MCP tooling
- PyPI packaging support
- structured QC packets with advisory LLM interpretation
- mandatory human confirmation before operational action
- Windows / Linux / macOS compatibility notes

---

## 6) 🤖 Models and performance

### Binary classification models

Distinguish between **"Normal"** and **"Abnormal"** measurements.

- **Performance**: test F1-score 0.9960, AUC 0.999–1.0
- **Algorithms**: Random Forest, XGBoost, Bagging classifiers
- **Sampling strategies**: SMOTE, ADASYN, SMOTEENN, UnderSampling

### Multi-class diagnostic models

Identify the root cause of an abnormality:

- Possible instrument instability
- Potential concentration anomaly
- Combined factors
- Other reasons, retesting recommended

**Features**: internal precision metrics, concentration mismatch ratios.

### Performance highlights

| Metric | Binary classification | Multi-class diagnosis |
|--------|----------------------|-----------------------|
| Accuracy | 99.61% | 99.84% |
| F1-score | 0.9960 | 0.9909 (balanced) |
| Recall (Normal) | 99.8% | – |
| AUC | 0.999–1.0 | – |

Validated on **26,218 historical measurements**.

---

## 7) 📝 Citation and contact

If you use Hg-MC-Auto in your research, please cite the associated manuscript:

```bibtex
@article{zhou2025selfdriving,
  title={A Data-Driven, Post-Acquisition Quality Diagnostic Pipeline for Isotope Analysis by MC-ICP-MS},
  author={Zhou, Chufan and Huang, Qiang and Tang, Yang and Zhong, Ying and Feng, Xinbin},
  journal={Journal of Analytical Atomic Spectrometry},
  year={2025},
  doi={10.1039/D5JA00519A}
}
```

### Contact

Laboratory of Karst Environmental Evolution and Ecological Security,
Institute of Geochemistry, Chinese Academy of Sciences,
Guiyang, Guizhou 550081, China

We welcome experts from different laboratories to contribute their expertise
and make contributions to the intelligent geochemistry laboratory. Welcome to
join us and make a change together.

- **Chufan Zhou**: 📧 zhouchufan@mail.gyig.ac.cn · 🔗 [ORCID 0009-0008-0144-9017](https://orcid.org/0009-0008-0144-9017)
- **Qiang Huang** (Corresponding Author): 📧 huangqiang@mail.gyig.ac.cn · 🔗 [ORCID 0000-0003-1568-9042](https://orcid.org/0000-0003-1568-9042)

For development questions or collaboration, use the repository issue tracker
or project maintainer contact listed above.

---

## 8) 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Ensure code follows PEP 8 guidelines
5. Include tests for new functionality

## 9) 🐛 Issues and support

- **Bug reports**: Use the [GitHub Issues](https://github.com/IGeochemCloud/Hg-MC-Auto/issues) page
- **Questions**: Check the Wiki or open a discussion
- **Feature requests**: Submit via GitHub Issues with the `enhancement` label

## 10) 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.