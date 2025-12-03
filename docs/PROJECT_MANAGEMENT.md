# Project Management & Development Guide

## 🎯 Current Status

**Phase**: Foundation Setup  
**Goal**: Establish synthetic-to-real ML pipeline for MES price prediction

## 📋 Project Roadmap

### ✅ Phase 0: Foundation (COMPLETE)
- [x] Project structure created
- [x] Price generator (tick-based MES simulation)
- [x] Fractal state manager (hierarchical states)
- [x] Visualization tools (candlestick charts)
- [x] Analysis utilities (statistics & summaries)
- [x] Documentation (README, ARCHITECTURE)
- [x] Dependencies (requirements.txt)

### ✅ Phase 1: Synthetic Archetype Generation (COMPLETE)
- [x] Define 5-10 key archetypes (patterns to generate)
- [x] Generate clean archetype datasets
- [x] Validate archetype characteristics
- [x] Label high-probability zones procedurally
- [x] Save archetype library (parquet files)

**Archetypes to Create**:
1. **Pure Rally Day** - Sustained upward movement
2. **Pure Range Day** - Choppy, bounded movement
3. **Breakout Pattern** - RANGING → BREAKOUT → RALLY
4. **Breakdown Pattern** - RANGING → BREAKDOWN → SELLOFF
5. **Reversal Pattern** - RALLY → RANGING → BREAKDOWN (or inverse)
6. **Zombie Grind** - Slow, persistent directional movement
7. **Volatile Chop** - High volatility, no clear direction
8. **Opening Bell** - RTH open volatility spike
9. **Closing Squeeze** - End-of-day positioning
10. **News Event** - Sudden volatility spike

### ✅ Phase 2: Pattern Recognition Pre-training (COMPLETE)
- [x] Extract tick-based features from archetypes
- [x] Train state classifier (Random Forest baseline)
- [x] Train sequence encoder (LSTM/Transformer) (Deferred to Phase 4)
- [x] Evaluate on held-out synthetic data
- [x] Save pre-trained models

### ✅ Phase 3: Real Data Integration (COMPLETE)
- [x] Load continuous_contract.json
- [x] Align features with synthetic data
- [x] Validate data quality
- [x] Apply pre-trained models
- [x] Analyze predictions vs reality

### ✅ Phase 4: Fine-tuning & Optimization (COMPLETE)
- [x] Analyze Feature Drift (Synthetic vs Real)
- [x] Train Balanced Model
- [x] Apply Threshold Optimization
- [x] Visualize Optimized Results
- [ ] Cross-validation (Deferred)
- [ ] Performance metrics (Deferred)

**Synthetic Data**:
- Raw generated data: `out/data/synthetic/raw/`
- Processed archetypes: `out/data/synthetic/archetypes/`
- Training datasets: `out/data/synthetic/training/`

**Real Data**:
- Raw continuous contract: `src/data/continuous_contract.json` (already exists)
- Processed features: `out/data/real/processed/`

**Models**:
- Pre-trained: `out/models/pretrained/`
- Fine-tuned: `out/models/finetuned/`
- Production: `out/models/production/`

**Results**:
- Charts: `out/charts/`
- Metrics: `out/results/metrics/`
- Backtests: `out/results/backtests/`

**Experiments**:
- Configs: `configs/`
- Notebooks: `notebooks/`

## 🔄 Development Workflow

### Daily Workflow

1. **Start of Day**
   ```bash
   # Activate environment (already done in your case)
   # .venv312 is active
   
   # Pull latest changes
   git pull
   
   # Check what needs to be done
   cat docs/PROJECT_MANAGEMENT.md
   ```

2. **During Development**
   ```bash
   # Run tests frequently
   pytest tests/ -v
   
   # Generate synthetic data
   python scripts/generate_archetypes.py
   
   # Train models
   python scripts/train_state_detector.py
   
   # Visualize results
   python scripts/visualize_results.py
   ```

3. **End of Day**
   ```bash
   # Run full test suite
   pytest tests/ --cov=src --cov=lab
   
   # Commit changes
   git add .
   git commit -m "Descriptive message"
   git push
   
   # Update this file with progress
   ```

### Experiment Workflow

1. **Define Experiment**
   - Create config file in `configs/`
   - Document hypothesis and expected outcome

2. **Run Experiment**
   - Generate/load data
   - Train model
   - Evaluate results

3. **Analyze Results**
   - Review metrics
   - Visualize predictions
   - Compare to baseline

4. **Document Findings**
   - Update experiment log
   - Save best models
   - Note lessons learned

## 📊 Key Metrics to Track

### Synthetic Data Quality
- State distribution (% of each state)
- Tick movement statistics (mean, std)
- Volume patterns
- Session characteristics

### Model Performance
- **Classification** (State Detection):
  - Accuracy, Precision, Recall, F1
  - Confusion matrix
  - Per-state performance

- **Regression** (Price Prediction):
  - MAE, RMSE (in ticks)
  - Directional accuracy
  - Prediction horizon performance

- **Trading** (Backtest):
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Profit factor

## 🐛 Debugging Checklist

When things go wrong:

1. **Data Issues**
   - [ ] Check data shape and types
   - [ ] Verify no NaN or inf values
   - [ ] Confirm tick alignment (all multiples of 0.25)
   - [ ] Validate state labels

2. **Model Issues**
   - [ ] Check input/output dimensions
   - [ ] Verify loss is decreasing
   - [ ] Check for overfitting (train vs val)
   - [ ] Inspect predictions on sample data

3. **Performance Issues**
   - [ ] Profile code (cProfile)
   - [ ] Check memory usage
   - [ ] Optimize data loading
   - [ ] Use batch processing

## 📝 Code Standards

### Python Style
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small (<50 lines)
- Use meaningful variable names

### Testing
- Write tests for all new functions
- Aim for >80% code coverage
- Use fixtures for common setup
- Mock external dependencies

### Documentation
- Update README when adding features
- Document all config options
- Add examples for new functionality
- Keep architecture docs current

## 🎯 Next Actions (Immediate)

### This Week
1. **Create archetype generation script**
   - Script: `scripts/generate_archetypes.py`
   - Generate 10 archetype types
   - Save to parquet files
   - Visualize each archetype

2. **Validate archetypes**
   - Script: `scripts/validate_archetypes.py`
   - Check statistics match expectations
   - Compare to real data distributions
   - Document findings

3. **Set up testing framework**
   - Create test files for generators
   - Test price generator edge cases
   - Test fractal state transitions
   - Set up CI/CD (optional)

### Next Week
1. **Feature engineering**
   - Define feature extraction pipeline
   - Implement rolling window features
   - Add technical indicators
   - Normalize/scale features

2. **Baseline model**
   - Train Random Forest on synthetic
   - Evaluate on held-out synthetic
   - Apply to real data
   - Document performance

## 🤝 Collaboration Notes

### When Working with AI Assistant
- Provide context about what you're working on
- Share error messages in full
- Describe expected vs actual behavior
- Ask for explanations when unclear

### When Sharing Code
- Include relevant imports
- Show sample data
- Provide error tracebacks
- Mention environment details

## 📚 Resources

### Internal Docs
- [Architecture](ARCHITECTURE.md) - System design
- [README](../README.md) - Quick start guide
- [Price Generator Docs](PRICE_GENERATOR.md) - Generator details

### External Resources
- MES Contract Specs: [CME Group](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html)
- Machine Learning: [Scikit-learn Docs](https://scikit-learn.org/)
- PyTorch: [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 🎉 Milestones

- [x] **Milestone 0**: Project structure and generators complete
- [x] **Milestone 1**: Archetype library generated and validated
- [x] **Milestone 2**: Baseline model trained on synthetic data
- [x] **Milestone 3**: Model applied to real data with >60% accuracy
- [x] **Milestone 4**: Fine-tuned model with >70% accuracy (Balanced Model)
- [ ] **Milestone 5**: Backtested strategy with positive Sharpe ratio

---

**Last Updated**: 2025-12-01  
**Current Focus**: User Verification of Liquidity Physics  
**Blockers**: User Approval  
**Next Review**: After approval, proceed to production pipeline
