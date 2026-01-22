# BAI_seminar

This repository contains code and experiments for the BAI seminar project.

---

## Installation

### Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## Download Additional Content

### Clone the Continuous Thought Machines repository
```bash
git clone https://github.com/SakanaAI/continuous-thought-machines.git
```

### Download the maze dataset
```bash
gdown "https://drive.google.com/uc?id=1Z8FFnZ7pZcu7DfoSyfy-ghWa08lgYl1V"
```

---

## Extract the Maze Dataset

### Windows
```powershell
Expand-Archive -Path small-mazes.zip -DestinationPath small-mazes
```

After extraction, make sure the **`small-mazes`** folder is not nested inside another `small-mazes` directory.

### Linux
```bash
unzip small-mazes.zip
```

---

## Repository Setup

### Rename the Continuous Thought Machines folder
```
continuous-thought-machines → continuous_thought_machines
```

---

## CTM Variant Setup (Important)

### Replace `ctm.py` in the Continuous Thought Machines repository

1. Copy:
```
ctmvariants/ctm.py
```

2. Into:
```
continuous_thought_machines/ctm.py
```

3. Replace the existing `ctm.py` file in the `continuous_thought_machines` repository with the one from `ctmvariants`.

---

## TensorBoard

### Launch TensorBoard
```bash
tensorboard --logdir ./maze_logs
```

