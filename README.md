# Hg-MC-Auto

Hg-MC-Auto is an element-aware automation toolkit for isotope analysis workflows, with bilingual CLI support, agent-facing MCP tooling, and cross-platform Python packaging.
![alt text](<docs/Graphic abstract.png>)

- [A data-driven, post-acquisition quality diagnostic pipeline for isotope analysis by MC-ICP-MS](https://pubs.rsc.org/ja/article-abstract/41/5/1894/1230352/A-data-driven-post-acquisition-quality-diagnostic?redirectedFrom=fulltext)


## Quick summary

- User-ready package: `pip install hg-mc-auto`
- CLI entry points: `hgmc` and `hg-mc-auto`
- Supported elements: `hg`, `fe`, `cd`
- Platforms: Windows, Linux, macOS
- Release flow: `hgmc-release` and `hgmc release`

---

## 1) User guide

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

## 2) Developer guide

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

The recommended way to publish a new version to both GitHub and PyPI is the
one-command sync helper. It validates the version, refuses to run with
uncommitted changes, checks credentials, then builds, tags, pushes to GitHub,
and uploads to PyPI.

```bash
# 1. Bump the version in pyproject.toml (e.g. 0.1.1)
# 2. Commit your changes
# 3. Export your PyPI token (never commit it)
export PYPI_TOKEN="your-token"

# 4. Dry-run to validate without publishing
hgmc-release sync --version 0.1.1 --dry-run

# 5. Publish to GitHub and PyPI together
hgmc-release sync --version 0.1.1 --message "release 0.1.1"
```

The sync helper will fail early if the requested version does not match
`pyproject.toml`, if the git tag already exists, if the working tree is dirty,
or if `PYPI_TOKEN` / `twine` are missing.

Individual steps are also available:

```bash
hgmc-release check
hgmc-release build
hgmc-release publish
```

### PyPI publishing

```bash
python -m build
python -m twine upload dist/*
```

Set `PYPI_TOKEN` before publishing if you are using automation. Create the token
at https://pypi.org/manage/account/token/ and use `__token__` as the username.

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

## 3) Project structure

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

## 4) Platform notes

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

## 5) Release status

This project is currently in a release-oriented v0.1.0 state with:

- bilingual CLI support
- generalized element routing for `hg`, `fe`, and `cd`
- agent-oriented MCP tooling
- PyPI packaging support
- structured QC packets with advisory LLM interpretation
- mandatory human confirmation before operational action
- Windows / Linux / macOS compatibility notes

---

## 6) Citation and contact

If you use this project in published work, please cite the project and associated manuscript as appropriate for your workflow and repository policy.

For development questions or collaboration, use the repository issue tracker or project maintainer contact listed in the upstream project metadata.
│   ├── 3_Empirical_model.py           # Expert rule-based classification
│   ├── 4_ML_Predict.py                # ML model prediction interface
│   ├── 5_Exter_ML_train.py            # Binary classifier training
│   └── 6.Inter_ML_train.py            # Multi-class classifier training
│
├── custom_ranges_config.json          # User-configurable acceptance ranges
├── mouse_coordinates.config           # RPA coordinate settings
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```
## 📖 Usage Guide
Launching the Application
After installation, run:

```bash
python src/main.py
```

You will see the interactive interface:

```bash
Welcome to Hg_MC_Auto!
============================================================

Please select a task:
1. Automatically export isotope data
2. Automatically export instrument parameters, merge isotope data, and calculate isotope fractionation values
3. Classify data using an empirical model
4. Classify data using a machine learning model
5. Train your own machine learning expert model
0. Exit
```
## Task Options Explained
Option 1: Automated Data Export
Converts proprietary .dat files to structured CSV format

Merges with corresponding instrument log files

Uses RPA for vendor software interaction

Option 2: Isotope Calculation
Calculates δ202Hg values relative to NIST SRM 3133

Computes mass-independent fractionation anomalies (Δ-values)

Batch processes entire datasets

Option 3: Empirical Model Classification
Applies literature-based acceptance ranges (Table 1 in manuscript)

Flags measurements outside 95% confidence intervals

User-configurable thresholds via custom_ranges_config.json

Option 4: ML Model Prediction
Uses pre-trained ensemble models for quality assessment

Provides confidence scores for each prediction

Identifies probable causes for abnormal measurements

Option 5: Custom Model Training
Train laboratory-specific models using your annotated data

Supports both binary and multi-class classification

Adapts to different instrument performances and sample matrices

## 🤖 Models
Binary Classification Models
Purpose: Distinguish between "Normal" and "Abnormal" measurements

Performance: Test F1-score: 0.9960, AUC: 0.999-1.0

Algorithms: Random Forest, XGBoost, Bagging Classifiers

Sampling Strategies: SMOTE, ADASYN, SMOTEENN, UnderSampling

Multi-class Diagnostic Models
Purpose: Identify root causes of abnormalities

Categories:

"Possible instrument instability"

"Potential concentration anomaly"

"Combined factors"

"Other reasons, retesting recommended"

Features: Internal precision metrics, concentration mismatch ratios

## 📊 Performance Highlights
Metric	Binary Classification	Multi-class Diagnosis
Accuracy	99.61%	99.84%
F1-Score	0.9960	0.9909 (balanced)
Recall (Normal)	99.8%	-
AUC	0.999-1.0	-
Based on validation with 26,218 historical measurements

## 📝 Citation
If you use Hg-MC-Auto in your research, please cite:

bibtex
@article{zhou2025selfdriving,
  title={A Data‑Driven, Post‑Acquisition Quality Diagnostic Pipeline for Isotope Analysis by MC-ICP-MS},
  author={Zhou, Chufan and Huang, Qiang and Tang, Yang and Zhong, Ying and Feng, Xinbin},
  journal={Journal of Analytical Atomic Spectrometry},
  year={2025},
  doi={10.1039/D5JA00519A}
}
## 🤝 Contributing
We welcome contributions! Please:

Fork the repository

Create a feature branch

Submit a pull request

Ensure code follows PEP 8 guidelines

Include tests for new functionality

## 🐛 Issues and Support
Bug Reports: Use the GitHub Issues page

Questions: Check the Wiki or open a discussion

Feature Requests: Submit via GitHub Issues with the "enhancement" label

## 📧 Contact
Laboratory of Karst Environmental Evolution and Ecological Security, 
Institute of Geochemistry, Chinese Academy of Sciences,
Guiyang, Guizhou 550081, China

We welcome experts from different laboratories to contribute their expertise 
and make contributions to the intelligent geochemistry laboratory. 
Welcome to join us and make a change together.

Chufan Zhou:
📧 zhouchufan@mail.gyig.ac.cn
🔗 ORCID: [0009-0008-0144-9017](https://orcid.org/0009-0008-0144-9017)

Qiang Huang (Corresponding Author):
📧 huangqiang@mail.gyig.ac.cn
🔗 ORCID: [0000-0003-1568-9042](https://orcid.org/0000-0003-1568-9042)

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.