# Tools Documentation

This repository includes a comprehensive suite of industrial automation tools for ML notebook management, execution, and documentation.

## Overview

The tools are organized into several categories:

1. **Execution Tools** - Batch execution of notebooks
2. **Fixing Tools** - Repair corrupted notebooks
3. **Rebuilding Tools** - Rebuild notebooks with industrial patterns
4. **Documentation Tools** - Generate READMEs and documentation
5. **Validation Tools** - Check notebook health

## Execution Tools

### tools_execute_batch*.py

**Purpose:** Execute multiple notebooks in batch with configurable timeouts and error handling.

**Usage:**
```bash
python tools_execute_batch4.py
```

**Features:**
- Configurable timeout per notebook
- Error handling with continue-on-failure
- Progress logging
- Output preservation

**Files:**
- `tools_execute_batch2.py` - Initial batch execution
- `tools_execute_batch3.py` - Additional batch execution
- `tools_execute_batch4.py` - CPU-safe notebooks
- `tools_execute_batch5.py` - Remaining notebooks

### tools_execute_and_export.py

**Purpose:** Execute and export notebooks with proper output handling.

**Usage:**
```bash
python tools_execute_and_export.py
```

## Fixing Tools

### tools_fix_concatenated_imports.py

**Purpose:** Fix notebooks with concatenated import statements (single-line imports).

**Problem:** Some notebooks have imports like:
```python
import numpy as npimport pandas as pdimport matplotlib.pyplot as plt
```

**Solution:** Splits them into proper separate lines:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**Usage:**
```bash
python tools_fix_concatenated_imports.py
```

### tools_fix_ipynb_json.py

**Purpose:** Repair corrupted JSON in notebook files.

**Problem:** Notebooks can become corrupted with invalid JSON structure.

**Solution:** Parses and repairs JSON structure, preserving valid content.

**Usage:**
```bash
python tools_fix_ipynb_json.py
```

### tools_patch_first_cell_imports.py

**Purpose:** Ensure first code cell contains all necessary imports.

**Usage:**
```bash
python tools_patch_first_cell_imports.py
```

### tools_patch_lof_model_cell.py

**Purpose:** Patch Local Outlier Factor (LOF) model cells for compatibility.

**Usage:**
```bash
python tools_patch_lof_model_cell.py
```

### tools_patch_offline_first_batch.py

**Purpose:** Ensure offline-first execution patterns in batch processing.

**Usage:**
```bash
python tools_patch_offline_first_batch.py
```

### tools_remove_lof_unrelated_cells.py

**Purpose:** Remove unrelated cells from LOF notebooks.

**Usage:**
```bash
python tools_remove_lof_unrelated_cells.py
```

## Rebuilding Tools

### tools_rebuild_no_outputs_industrial.py

**Purpose:** Rebuild notebooks that have no outputs with industrial-grade patterns.

**Features:**
- Creates complete, executable notebooks
- Adds synthetic datasets for offline execution
- Implements proper ML pipelines
- Includes visualization and metrics

**Usage:**
```bash
python tools_rebuild_no_outputs_industrial.py
```

**Notebooks rebuilt:**
- K-Means Customer Segmentation
- SVM Image Recognition
- BERT Sentiment Analysis
- LSTM Stock Prediction
- CNN Medical Imaging
- GPT Fine-Tuning Basics
- Named Entity Recognition (spaCy)
- Isolation Forest Cybersecurity
- Autoencoders Denoising
- GAN Synthetic Data Generation
- ResNet Transfer Learning
- YOLOv8 Object Detection
- Stable Diffusion Prompt Engineering
- RL CartPole Agent

### tools_rebuild_stable_diffusion_industrial.py

**Purpose:** Rebuild Stable Diffusion notebook with CPU-safe default.

**Features:**
- Default: tiny SD pipeline (CPU-safe, no SIGKILL)
- Optional: full SD v1-5 (set `SD_FULL=1`)
- Documented execution patterns

**Usage:**
```bash
python tools_rebuild_stable_diffusion_industrial.py
```

### tools_rebuild_problematic.py

**Purpose:** Rebuild problematic notebooks with fixes.

**Usage:**
```bash
python tools_rebuild_problematic.py
```

### tools_rebuild_core_03.py

**Purpose:** Rebuild core notebooks (decision tree).

**Usage:**
```bash
python tools_rebuild_core_03.py
```

### tools_rebuild_core_gold.py

**Purpose:** Rebuild core gold-standard notebooks.

**Usage:**
```bash
python tools_rebuild_core_gold.py
```

### tools_rebuild_invalid_notebooks.py

**Purpose:** Rebuild invalid notebooks.

**Usage:**
```bash
python tools_rebuild_invalid_notebooks.py
```

### tools_rebuild_random_forest_finance.py

**Purpose:** Rebuild Random Forest Finance notebook.

**Usage:**
```bash
python tools_rebuild_random_forest_finance.py
```

### tools_rebuild_no_output_notebooks.py

**Purpose:** Rebuild notebooks with no outputs (legacy).

**Usage:**
```bash
python tools_rebuild_no_output_notebooks.py
```

### tools_repair_*.py

**Purpose:** Repair specific notebooks:
- `tools_repair_autoencoders_network_security.py`
- `tools_repair_dbscan_notebook.py`
- `tools_repair_hierarchical_notebook.py`
- `tools_repair_isolation_forest_cyber_notebook.py`
- `tools_repair_knn_recommender_notebook.py`
- `tools_repair_lof_notebook.py`

**Usage:**
```bash
python tools_repair_*.py
```

### tools_rewrite_*.py

**Purpose:** Rewrite specific cells:
- `tools_rewrite_dbscan_classifier_cell.py`
- `tools_rewrite_dbscan_visualizer_cell.py`
- `tools_rewrite_industrial_generator_cells.py`
- `tools_rewrite_nb_spam_cell.py`

**Usage:**
```bash
python tools_rewrite_*.py
```

## Documentation Tools

### tools_generate_readmes.py

**Purpose:** Generate comprehensive README files for all modules.

**Features:**
- Auto-generates README.md for each module folder
- Includes purpose, prerequisites, usage instructions
- Lists all notebooks with descriptions
- Provides troubleshooting guidance
- Documents heavy dependencies

**Usage:**
```bash
python tools_generate_readmes.py
```

**Output:**
- `01-Regression/README.md`
- `02-Classification/README.md`
- `03-Clustering/README.md`
- `04-NLP/README.md`
- `05-Computer-Vision/README.md`
- `06-Reinforcement-Learning/README.md`
- `07-Time-Series/README.md`
- `07-Time-Series-Forecasting/README.md`
- `08-Anomaly-Detection/README.md`
- `09-Generative-AI/README.md`
- `10-MLOps/README.md`
- `10-MLOps-Production/README.md`

### tools_notebooks_report.py

**Purpose:** Generate a report on notebook status.

**Usage:**
```bash
python tools_notebooks_report.py
```

## Validation Tools

### check_nb.py

**Purpose:** Check notebook health and structure.

**Usage:**
```bash
python check_nb.py
```

## Workflow Examples

### Complete Execution Workflow

```bash
# 1. Fix any corrupted notebooks
python tools_fix_concatenated_imports.py
python tools_fix_ipynb_json.py

# 2. Rebuild notebooks with no outputs
python tools_rebuild_no_outputs_industrial.py

# 3. Execute remaining notebooks
python tools_execute_batch5.py

# 4. Validate all notebooks
python validate_notebooks.py

# 5. Generate documentation
python tools_generate_readmes.py

# 6. Commit and push
git add -A .
git commit -m "Execute notebooks + generate READMEs"
git push
```

### Troubleshooting Workflow

```bash
# Check for issues
python validate_notebooks.py

# Fix specific issues
python tools_fix_concatenated_imports.py
python tools_fix_ipynb_json.py

# Rebuild problematic notebooks
python tools_rebuild_problematic.py

# Re-execute
python tools_execute_batch5.py
```

## Configuration

### Environment Variables

- `SD_FULL=1` - Enable full Stable Diffusion model (may SIGKILL on CPU)
- `TRANSFORMERS_OFFLINE=1` - Prevent model downloads
- `HF_HUB_OFFLINE=1` - Prevent HuggingFace hub access

### Timeout Settings

- Default timeout: 1800 seconds (30 minutes)
- Heavy notebooks: 3600 seconds (60 minutes)
- SD notebooks: 7200 seconds (120 minutes)

## Best Practices

1. **Always backup** before running rebuild tools
2. **Test on small subset** before full execution
3. **Monitor memory usage** for heavy notebooks
4. **Use virtual environments** for dependency isolation
5. **Commit frequently** to track changes
6. **Document changes** in commit messages

## Troubleshooting

### Common Issues

1. **SIGKILL on CPU** - Reduce steps, use tiny models
2. **Corrupted JSON** - Run `tools_fix_ipynb_json.py`
3. **Missing imports** - Run `tools_fix_concatenated_imports.py`
4. **No outputs** - Run `tools_rebuild_no_outputs_industrial.py`
5. **Memory issues** - Reduce batch size, use CPU-only

### Error Messages

- `Invalid JSON` - Run repair tools
- `SyntaxError` - Check concatenated imports
- `SIGKILL` - Reduce memory usage
- `Timeout` - Increase timeout or reduce complexity

## References

- Jupyter nbconvert documentation
- HuggingFace Transformers documentation
- PyTorch documentation
- Scikit-learn documentation
