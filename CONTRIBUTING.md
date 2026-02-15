# Contributing Guidelines

Thank you for your interest in contributing to the Advanced ML Mastery Collection! This document outlines the process for contributing.

## 🤝 How to Contribute

### 1. Fork the Repository

```bash
# Click "Fork" on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/advanced-ml-mastery-collection.git
cd advanced-ml-mastery-collection
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Add new notebooks following existing patterns
- Fix bugs or improve existing notebooks
- Update documentation
- Add tests if applicable

### 4. Submit a Pull Request

```bash
git add .
git commit -m "Add: Description of your changes"
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

---

## 📝 Coding Standards

### Notebook Structure

Each notebook should follow this structure:

```markdown
# Title

## 1. Introduction
- Problem description
- Dataset overview
- Objectives

## 2. Data Loading & Exploration
- Import libraries
- Load data
- Exploratory analysis

## 3. Data Preprocessing
- Cleaning
- Feature engineering
- Splitting

## 4. Model Development
- Model selection
- Training
- Evaluation

## 5. Results & Visualization
- Plots
- Metrics
- Interpretation

## 6. Conclusion
- Summary
- Future work
```

### Code Style

```python
# Use clear variable names
model_accuracy = 0.95

# Add comments for complex logic
# This is a custom implementation of...

# Use type hints when possible
def train_model(data: np.ndarray, labels: np.ndarray) -> model:
    pass

# Handle exceptions
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Training failed: {e}")
```

---

## 📋 Notebook Checklist

Before submitting a notebook, ensure:

- [ ] Clear title and description
- [ ] All cells executed (Kernel → Restart & Run All)
- [ ] No error outputs
- [ ] Meaningful visualizations
- [ ] Proper comments and explanations
- [ ] References to papers/concepts used
- [ ] Data sources properly attributed

---

## 🐛 Reporting Issues

When reporting issues, include:

1. **Description**: What went wrong?
2. **Steps to Reproduce**: How can we reproduce it?
3. **Expected Behavior**: What should happen?
4. **Environment**: Python version, library versions
5. **Screenshots**: If applicable

---

## 💬 Communication

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Pull Requests**: Link related issues in your PR description

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Recognition

Contributors will be recognized in the README.md Hall of Fame section!

Thank you for helping make this collection better! 🎉
