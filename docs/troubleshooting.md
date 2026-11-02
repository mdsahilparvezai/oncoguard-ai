# ONCOGUARD AI - Troubleshooting Guide

## Common Issues & Solutions

### Installation Issues

#### `ModuleNotFoundError: No module named 'oncoguard'`
**Cause:** Package not installed or PYTHONPATH not set.
**Solution:**
```bash
# Option 1: Install in development mode
pip install -e .

# Option 2: Run from project root with PYTHONPATH
cd C:\Users\LENOVO\OneDrive\Desktop\Breast Cancer
python -m src.oncoguard.data
```

#### `ImportError: cannot import name 'X' from 'src.oncoguard'`
**Cause:** Circular imports or missing `__init__.py`.
**Solution:** Check all `__init__.py` files exist and imports are correct.

#### `pip install` fails with dependency conflicts
**Cause:** Version conflicts between packages.
**Solution:**
```bash
# Create fresh venv
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Data Pipeline Issues

#### `FileNotFoundError: wdbc.data`
**Cause:** Dataset not downloaded.
**Solution:**
```bash
# Run data pipeline (downloads automatically)
python -m src.oncoguard.data
```

#### `ValueError: Schema validation failed`
**Cause:** Downloaded data doesn't match expected schema.
**Solution:**
```bash
# Force re-download
python -c "from src.oncoguard.data import load_raw_data; load_raw_data(force_download=True)"
```

#### `MemoryError` during data processing
**Cause:** Insufficient RAM for dataset (unlikely for 569 samples).
**Solution:** Check for runaway processes; dataset is small (~150KB).

#### `KeyError: 'diagnosis'` or column mismatch
**Cause:** UCI dataset format changed.
**Solution:** Check UCI URL; update `COLUMN_NAMES` in `data.py` if needed.

### Training Issues

#### `ValueError: Target contains previously unseen labels`
**Cause:** Target mapping issue or stratification problem.
**Solution:**
```python
# Verify target mapping
from src.oncoguard.data import TARGET_MAPPING
print(TARGET_MAPPING)  # Should be {'M': 1, 'B': 0}
```

#### `ConvergenceWarning: lbfgs failed to converge`
**Cause:** Logistic Regression needs more iterations.
**Solution:** Already handled (max_iter=2000 in train.py).

#### `RuntimeError: CUDA out of memory`
**Cause:** PyTorch trying to use GPU (not needed for tabular models).
**Solution:** Tabular models use scikit-learn (CPU). Only image experiment uses PyTorch.

#### Training takes too long
**Cause:** Random search with many iterations.
**Solution:** Reduce `n_iter` in ModelConfig, or use GridSearch for smaller spaces.

#### `joblib.dump` fails / PermissionError
**Cause:** Models directory permissions or disk full.
**Solution:**
```bash
# Check disk space
dir models\
# Ensure write permissions
icacls models /grant %USERNAME%:F
```

### Evaluation Issues

#### `FileNotFoundError: No model found`
**Cause:** No trained models in models/ directory.
**Solution:**
```bash
python -m src.oncoguard.train
```

#### `AttributeError: 'X' object has no attribute 'predict_proba'`
**Cause:** Model doesn't support probability estimates.
**Solution:** All baseline models configured with `probability=True` or calibration.

#### SHAP fails with `TypeError` or `AttributeError`
**Cause:** Model type not supported by TreeExplainer.
**Solution:** KernelExplainer fallback implemented; check SHAP version compatibility.

#### `ImportError: No module named 'shap'`
**Cause:** SHAP not installed.
**Solution:**
```bash
pip install shap==0.45.0
```

### Dashboard Issues

#### `streamlit: command not found`
**Cause:** Streamlit not installed or not in PATH.
**Solution:**
```bash
pip install streamlit==1.38.0
# Or run with python -m
python -m streamlit run src/oncoguard/app.py
```

#### `ModuleNotFoundError` in Streamlit
**Cause:** Streamlit running in different environment.
**Solution:**
```bash
# Ensure same venv activated
.venv\Scripts\activate
python -m streamlit run src/oncoguard/app.py
```

#### Dashboard shows "No data loaded" / "No model found"
**Cause:** Pipeline not run yet.
**Solution:** Run pipeline steps in order:
1. `python -m src.oncoguard.data`
2. `python -m src.oncoguard.train`
3. `python -m src.oncoguard.evaluate`

#### Port 8501 already in use
**Cause:** Another Streamlit instance running.
**Solution:**
```bash
# Kill existing process
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Or use different port
streamlit run src/oncoguard/app.py --server.port 8502
```

#### Blank page / white screen
**Cause:** JavaScript error or Streamlit version mismatch.
**Solution:**
```bash
# Clear browser cache, try incognito
# Check browser console for errors
# Ensure Streamlit version matches requirements
```

### Assistant Issues

#### Assistant stuck in "Thinking..."
**Cause:** LLM API timeout or rate limit.
**Solution:**
- Check internet connection
- Verify API key valid
- Check OpenAI status page
- Fallback: Assistant auto-switches to offline mode on failure

#### "OpenAI API key not configured"
**Cause:** `OPENAI_API_KEY` not set.
**Solution:**
```bash
# Option 1: Environment variable
set OPENAI_API_KEY=your_key_here

# Option 2: .env file
echo OPENAI_API_KEY=your_key_here > .env

# Option 3: Streamlit secrets
mkdir .streamlit
echo 'OPENAI_API_KEY = "your_key_here"' > .streamlit/secrets.toml
```

#### Assistant gives generic/offline responses despite API key
**Cause:** API key invalid, quota exceeded, or connection failed.
**Solution:**
```bash
# Test API key
python -c "
import openai
client = openai.OpenAI()
print(client.models.list())
"
```

#### Hindi/Hinglish not working well
**Cause:** LLM limitations or prompt issues.
**Solution:** Online mode handles better; offline mode has limited Hindi support.

### Test Failures

#### `pytest` fails with import errors
**Cause:** Tests run from wrong directory or missing PYTHONPATH.
**Solution:**
```bash
# Run from project root
cd C:\Users\LENOVO\OneDrive\Desktop\Breast Cancer
pytest tests/ -v
```

#### Tests pass individually but fail together
**Cause:** Shared state between tests (global variables, cached data).
**Solution:** Check for module-level state; use fixtures for isolation.

#### `AssertionError: CV score too low`
**Cause:** Model performance varies; test threshold too strict.
**Solution:** Adjust test thresholds; DummyClassifier should be ~0.5 ROC-AUC.

### Windows-Specific Issues

#### Path too long errors
**Cause:** Windows 260-char path limit.
**Solution:**
```bash
# Enable long paths (Windows 10 1607+)
# Run as admin:
git config --system core.longpaths true
# Or move project to shorter path
```

#### `SyntaxError: (unicode error) 'unicodeescape'`
**Cause:** Backslashes in Windows paths in Python strings.
**Solution:** Use raw strings or forward slashes:
```python
# Good
Path(r"C:\Users\path")  # raw string
Path("C:/Users/path")   # forward slashes

# Bad
Path("C:\Users\path")   # \U interpreted as unicode
```

#### PowerShell execution policy
**Cause:** Scripts disabled.
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Virtual environment activation fails
**Cause:** Execution policy or path issues.
**Solution:**
```cmd
# Use cmd instead of PowerShell
.venv\Scripts\activate.bat
```

### Performance Optimization

#### Slow SHAP computation
**Cause:** KernelExplainer on large datasets.
**Solution:**
- Reduce `sample_size` in `compute_shap_values`
- Use TreeExplainer for tree models (automatic)
- Limit to top features

#### Slow dashboard loading
**Cause:** Large model artifacts or reports.
**Solution:**
- Use `@st.cache_data` (already implemented)
- Clean old models: `del models\*.joblib` (keep latest)
- Clean old reports: `del reports\*.json`

#### High memory usage
**Cause:** Multiple large DataFrames in memory.
**Solution:**
- Dashboard loads data lazily
- Restart Streamlit to clear cache: `streamlit cache clear`

## Debugging Tips

### Enable Debug Logging
```bash
# Set log level
set LOG_LEVEL=DEBUG
python -m src.oncoguard.data
```

### Inspect Data Splits
```python
from src.oncoguard.data import load_processed_data
splits = load_processed_data()
print(splits.X_train.shape, splits.y_train.value_counts())
```

### Inspect Model Artifact
```python
from src.oncoguard.train import load_model_artifact
artifact = load_model_artifact(Path("models/latest.joblib"))
print(artifact.keys())
print(artifact["cv_results"])
```

### Check Assistant Mode
```python
from src.oncoguard.assistant import OncoguardAssistant
assistant = OncoguardAssistant()
print(f"Mode: {assistant.mode}, Provider: {assistant.provider.name}")
```

### Verbose Test Output
```bash
pytest tests/ -v -s --tb=long
```

## Getting Help

### Check Logs
- Console output from running commands
- Streamlit logs in terminal
- Browser developer tools (F12) for dashboard errors

### Verify Environment
```bash
python --version
pip list | findstr -i "sklearn torch streamlit shap"
```

### Clean Rebuild
```bash
# Remove all generated files
rmdir /s data\processed
rmdir /s models
rmdir /s reports
rmdir /s .venv

# Recreate
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.oncoguard.data
python -m src.oncoguard.train
python -m src.oncoguard.evaluate
```

### Known Limitations (Not Bugs)
- DummyClassifier ROC-AUC ~0.5 (expected)
- Small dataset → wide confidence intervals
- No external validation → unknown generalization
- Offline assistant → limited responses
- SHAP KernelExplainer → slow, approximate
- Windows paths → use forward slashes in code

## Reporting Issues

When reporting issues, include:
1. **OS & Python version**
2. **Command that failed**
3. **Full error traceback**
4. **Steps to reproduce**
5. **Expected vs actual behavior**
6. **Relevant log output**

This is a research project — check documentation first, then search existing issues.