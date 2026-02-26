# Agent Instructions for Twinship Anomaly Detection

## Project Safety Rules

### Data Protection
- **NEVER** modify or delete `data1.csv` directly
- **NEVER** delete `anomalies.db` without user confirmation
- Always create backups before bulk data operations

### Model Files
- Save new models with version tags (e.g., `clustered_svd_v2.pkl`)
- Do not overwrite existing model files without asking
- Keep at least the last 2 model versions

## Workflow Preferences

### Before Making Changes
1. Read relevant source files to understand current implementation
2. Check for existing tests or validation logic
3. Verify Python environment is activated (`.venv`)

### After Code Changes
1. Run `python detect_anomalies.py` to verify detection still works
2. Run `python analyze_results.py` to regenerate reports
3. Check for errors using `get_errors` tool

### When Modifying Detection Logic
1. Document the change rationale
2. Compare anomaly counts before/after
3. Verify recall on known constraint violations (should be 100%)

## Code Style

### Python
- Use type hints for function signatures
- Follow existing patterns in `src/` modules
- Keep functions focused and under 50 lines
- Add docstrings for public functions

### File Organization
```
src/preprocessing/  → Data loading and cleaning
src/models/         → Detection algorithms
src/database/       → Anomaly storage
```

## Common Tasks

### Adding a New Feature Column
1. Update `FEATURE_COLS` in `src/preprocessing/data_loader.py`
2. Update `get_feature_columns()` function
3. Retrain model with `python detect_anomalies.py`
4. Regenerate analysis with `python analyze_results.py`

### Adding a New Constraint Check
1. Add flag column in `load_and_clean()` in `data_loader.py`
2. Add to `classify_anomaly()` in `anomaly_db.py`
3. Update `copilot-instructions.md` with new constraint

### Changing Clustering Parameters
1. Modify `--clusters` argument in `detect_anomalies.py`
2. Consider using elbow method to validate choice
3. Document reasoning for the new cluster count

### Adding a New Plot
1. Add plot function in `analyze_results.py`
2. Call it from `main()` with appropriate step number
3. Add to "Generated Plots" section in summary report

## Terminal Commands

### Environment Setup
```bash
source .venv/bin/activate
```

### Run Detection Pipeline
```bash
python detect_anomalies.py --clusters 4 --components 3 --threshold-pct 95
```

### Run Analysis
```bash
python analyze_results.py --output-dir results
```

### Query Database
```bash
sqlite3 anomalies.db "SELECT anomaly_type, COUNT(*) FROM anomalies GROUP BY anomaly_type;"
```

### Reset and Rerun
```bash
rm -f anomalies.db && python detect_anomalies.py && python analyze_results.py
```

## Error Handling

### "ModuleNotFoundError"
- Ensure venv is activated: `source .venv/bin/activate`
- Install missing packages: `pip install <package>`

### "Database is locked"
- Close other SQLite connections
- Check for running Python processes: `ps aux | grep python`

### "Shape mismatch" in plots
- Verify cluster counts match between model and data
- Check for NaN values in feature columns

## Testing Checklist

Before considering a task complete:
- [ ] Code runs without errors
- [ ] Anomaly detection produces reasonable results (1-10% anomaly rate)
- [ ] Known constraint violations are detected (100% recall)
- [ ] Generated plots render correctly
- [ ] CSV exports contain expected columns
