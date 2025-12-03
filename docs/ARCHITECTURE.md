# FracFire Architecture

## System Overview

FracFire is a **synthetic-to-real ML platform** for futures price prediction. The architecture follows a clear separation of concerns with three main layers:

```
┌──────────────────────────────────────────────────────────────┐
│                    LAYER 0: PHYSICS ENGINE                    │
│                  (Synthetic Data Generation)                  │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                LAYER 1: PATTERN RECOGNITION                   │
│              (Pre-training on Synthetic Data)                 │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  LAYER 2: PRODUCTION MODELS                   │
│               (Fine-tuning on Real Data)                      │
└──────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. **Generator = Physics, ML = Patterns**
- **Price Generator**: Understands tick mechanics, state configurations, session effects
- **ML Models**: Learn patterns, state transitions, entry/exit signals
- **Separation**: Generator stays pure (no ML inside), ML drives it externally

### 2. **Tick-Based from the Start**
- All prices stored as integer ticks (MES tick size = 0.25)
- No floating-point rounding errors
- Perfect for ML feature engineering
- Columns: `delta_ticks`, `range_ticks`, `body_ticks`, `upper_wick_ticks`, `lower_wick_ticks`

### 3. **Hierarchical State Management**
- **Day-level**: TREND_DAY, RANGE_DAY, BREAKOUT_DAY, REVERSAL_DAY, QUIET_DAY, VOLATILE_DAY
- **Hour-level**: IMPULSE, CONSOLIDATION, RETRACEMENT, CONTINUATION, REVERSAL, CHOPPY
- **Minute-level**: RALLY, BREAKDOWN, RANGING, FLAT, ZOMBIE, IMPULSIVE, BREAKOUT

### 4. **Labels Everywhere**
Every generated bar includes:
- Market state label
- Trading session
- Segment ID (for meso-scale patterns)
- Macro regime (day-level classification)

## Component Architecture

### Lab Module (`lab/`)

Research and experimentation code.

#### Generators (`lab/generators/`)

**PriceGenerator** - Main tick-based simulation engine
- Generates OHLCV bars from tick movements
- 7 standard market states + custom states
- Session effects (Asian, London, RTH, etc.)
- Day-of-week multipliers
- Segment-based state control

**FractalStateManager** - Hierarchical state management
- Day → Hour → Minute state cascade
- Proper transition probabilities
- Combined parameter calculation
- Ready for integration (currently standalone)

**Utils** - Analysis and validation
- `summarize_day()` - Comprehensive statistics
- `compare_states()` - State characteristic comparison
- `print_summary()` - Pretty-printed output

#### Visualizers (`lab/visualizers/`)

**ChartVisualizer** - Matplotlib candlestick charts
- Configurable colors, sizes, styles
- Volume subplot
- State change annotations
- Session background shading
- Quick chart helper function

### Source Module (`src/`)

Production code for ML pipeline.

#### Data (`src/data/`)

**Synthetic** - Synthetic data generators
- Archetype generation (clean patterns)
- Batch generation utilities
- Dataset versioning

**Real** - Real data adapters
- Continuous contract loader
- Data validation
- Feature alignment with synthetic

**Loaders** - Unified data interface
- Abstract base class
- Consistent API for synthetic and real data
- Automatic feature engineering

#### Models (`src/models/`)

**Base Model** - Abstract base class
- Standard interface for all models
- Save/load functionality
- Metrics tracking

**State Detector** - Market state classifier
- Pre-trained on synthetic archetypes
- Transfer learning to real data
- Confidence scores

**Price Predictor** - Price movement prediction
- Multi-horizon forecasting
- Tick-based targets
- Ensemble methods

#### Training (`src/training/`)

**Pretrain** - Pre-training on synthetic data
- Generate clean archetypes
- Train pattern recognition
- Save pre-trained weights

**Finetune** - Fine-tuning on real data
- Load pre-trained model
- Adapt to real market dynamics
- Continuous learning

**Train Utils** - Shared training utilities
- Data loaders
- Loss functions
- Callbacks and logging

#### Evaluation (`src/evaluation/`)

**Metrics** - Performance metrics
- Classification metrics (state detection)
- Regression metrics (price prediction)
- Custom trading metrics (Sharpe, max drawdown, etc.)

**Backtest** - Backtesting engine
- Simulate trading on historical data
- Position sizing
- Risk management
- Performance reporting

#### Utils (`src/utils/`)

**Config** - Configuration management
- YAML-based configs
- Environment variables
- Experiment tracking

**Logger** - Logging setup
- Structured logging
- File and console handlers
- Log rotation

## Data Flow

### Phase 1: Synthetic Archetype Generation

```python
# Generate clean patterns
gen = PriceGenerator(initial_price=5000.0, seed=42)

# Pure RALLY day
rally_day = gen.generate_day(
    start_date,
    state_sequence=[(0, MarketState.RALLY)],
    macro_regime='TREND_DAY'
)

# RANGING → BREAKOUT pattern
breakout_pattern = gen.generate_day(
    start_date,
    state_sequence=[
        (0, MarketState.RANGING),
        (60, MarketState.BREAKOUT),
        (120, MarketState.RALLY),
    ],
    segment_length=15,
)

# Save archetypes
rally_day.to_parquet('data/synthetic/archetypes/rally_day.parquet')
```

### Phase 2: Pattern Recognition Pre-training

```python
# Load synthetic archetypes
archetype_data = load_archetypes('data/synthetic/archetypes/')

# Extract features
features = extract_tick_features(archetype_data)

# Train state detector
state_detector = StateDetector()
state_detector.fit(features, labels=archetype_data['state'])

# Save pre-trained model
state_detector.save('models/state_detector_pretrained.pt')
```

### Phase 3: Fine-tuning on Real Data

```python
# Load real continuous contract data
real_data = load_continuous_contract('src/data/continuous_contract.json')

# Load pre-trained model
state_detector = StateDetector.load('models/state_detector_pretrained.pt')

# Fine-tune on real data
state_detector.finetune(real_data, epochs=10, lr=1e-4)

# Save production model
state_detector.save('models/state_detector_production.pt')
```

### Phase 4: Production Inference

```python
# Load production model
model = StateDetector.load('models/state_detector_production.pt')

# Real-time inference
current_features = extract_features(live_data)
predicted_state = model.predict(current_features)
confidence = model.predict_proba(current_features)

# Trading decision
if predicted_state == 'breakout' and confidence > 0.8:
    # Enter trade
    pass
```

## Configuration System

### Experiment Configs (`configs/`)

YAML files define experiments:

```yaml
# configs/synthetic_pretrain.yaml
experiment:
  name: "state_detector_pretrain_v1"
  description: "Pre-train state detector on synthetic archetypes"

data:
  source: "synthetic"
  archetypes:
    - rally_day
    - range_day
    - breakout_pattern
  num_samples: 10000

model:
  type: "RandomForestClassifier"
  params:
    n_estimators: 100
    max_depth: 10

training:
  validation_split: 0.2
  random_seed: 42
```

### State Configs

Market states defined in code:

```python
StateConfig(
    name="rally",
    avg_ticks_per_bar=18.0,
    up_probability=0.7,
    trend_persistence=0.8,
    volatility_multiplier=1.5,
)
```

## Testing Strategy

### Unit Tests (`tests/`)

- Test individual components in isolation
- Mock external dependencies
- Fast execution (<1s per test)

```python
def test_price_generator_tick_size():
    gen = PriceGenerator(initial_price=5000.0, seed=42)
    bar = gen.generate_bar(datetime.now())
    
    # All prices should be multiples of 0.25
    assert bar['open'] % 0.25 == 0
    assert bar['high'] % 0.25 == 0
    assert bar['low'] % 0.25 == 0
    assert bar['close'] % 0.25 == 0
```

### Integration Tests

- Test component interactions
- Use small real datasets
- Verify end-to-end workflows

```python
def test_synthetic_to_real_pipeline():
    # Generate synthetic data
    gen = PriceGenerator(seed=42)
    synthetic_df = gen.generate_day(start_date)
    
    # Train model
    model = StateDetector()
    model.fit(synthetic_df)
    
    # Apply to real data
    real_df = load_continuous_contract()
    predictions = model.predict(real_df)
    
    # Verify predictions
    assert len(predictions) == len(real_df)
    assert all(p in MarketState for p in predictions)
```

### Backtests

- Full system tests on historical data
- Performance metrics
- Risk analysis

## Performance Considerations

### Generation Speed
- 1 day (1440 bars): ~50ms
- 30 days: ~1.5s
- Fast enough for rapid iteration

### Model Training
- Pre-training on synthetic: Minutes to hours
- Fine-tuning on real: Minutes
- Inference: Milliseconds per bar

### Storage
- Synthetic data: Parquet format (compressed)
- Models: PyTorch checkpoints
- Results: JSON + CSV for analysis

## Future Extensions

### Planned Features

1. **Markov State Driver**
   - Fit Markov model on real data
   - Generate realistic state sequences
   - Feed to PriceGenerator

2. **Advanced Models**
   - Transformer-based encoders
   - LSTM for sequence modeling
   - Ensemble methods

3. **Real-time Pipeline**
   - Live data ingestion
   - Online learning
   - Production deployment

4. **Risk Management**
   - Position sizing
   - Stop-loss optimization
   - Portfolio allocation

### Integration Points

- **Data Sources**: Easy to add new data providers
- **Models**: Plug-and-play model architecture
- **Strategies**: Modular strategy components
- **Execution**: Broker API integration

## Best Practices

1. **Version Everything**
   - Data: DVC or similar
   - Models: Git LFS or model registry
   - Configs: Git

2. **Document Experiments**
   - Use experiment tracking (MLflow, W&B)
   - Log all hyperparameters
   - Save all results

3. **Test Thoroughly**
   - Unit tests for all components
   - Integration tests for workflows
   - Backtests for strategies

4. **Keep It Simple**
   - Start with simple models
   - Add complexity only when needed
   - Measure everything

5. **Iterate Quickly**
   - Fast feedback loops
   - Small experiments
   - Learn from failures
