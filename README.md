# OMDA (OpenMind Decision Analysis)

Tools for decision analysis under the framework of OpenMind Club.

The current version only implements ROC and EWM algorithms. It's very easy to use now -- just watch and run the demos!

## Installation

Change the working directory to the folder which contains `setup.py`.

Run:

```python
pip install .
```

or

```python
pip install -e .
```

if you want to install a it in editable mode (i.e. setuptools “develop mode”; which may be more useful at the current developing stage).

### Recommendation: `uv`

Run the following command at the root directory (containing `pyproject.toml`)

```
uv pip install -e .
```

At the directory of specific analysis demo:

```
uv run run.py
```

## Background and Motivation

1. Tsing-Yu (@karanotsingyu) initially wrote the codes during the course "OMDA001" developed by @ouyangzhiping and @OpenMindClub. These codes implemented the ROC and EWM algorithms to address specific tasks required in the course.
2. Tsing-Yu employed the first version of the codes to help him solve several real-world decision problems, during autumn 2023 to summer 2024.
3. Tsing-Yu rewrite the code under the object-oriented style during the course "AIP002" also led by @ouyangzhiping and @OpenMindClub. This update (v0.0.2) aims to enhance the user experience and productivity, particularly for multi-round decision analysis.
4. Although this project originated from the "OpenMindClub Decision Analysis" (OMDA) course, it currently does not implement the decision framework created by @ouyangzhiping. The project's primary goal is merely to deepen Tsing-Yu's understanding of OMDA's concepts and principles, focusing on the algorithms covered in the course.
5. Tsing-Yu also views this project as an experience of practicing his coding ability during the current period before his completion of the course "OMDA003" or the "Huoshui Hackathon 002".
6. Tsing-Yu intends to explore the implementation of additional algorithms such as BWM, TOPSIS, and VIKOR, though full development of the project is not his current priority.

## Policies

1. The decision framework's creator, @ouyangzhiping, retains the copyrights and rights to determine the repository's use and distribution.
2. Until formal agreements, contracts, or policies are established between @karanotsingyu and @ouyangzhiping/@OpenMindClub, this project must not implement the framework explicitly. @ouyangzhiping reserves the right to interpret what constitutes "explicit implementation."
3. The project's policies will be formally reviewed and updated after the conclusion of "OMDA003" or the "Huoshui Hackathon 002". Any updates to the policies require the approval of @ouyangzhiping/@OpenMindClub. Without this consent, the project will be terminated, and the repository will be archived or deleted.
