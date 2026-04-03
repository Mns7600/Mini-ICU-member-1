# 🏥 Mini ICU - Physiology Engine (Member 1 MVP)

Welcome to the foundational Physiology Engine for the Mini ICU project. 

This module generates minute-level, realistic time-series data for synthetic patients. It acts as the "base reality" of the simulation, providing the baseline physiological signals that downstream disease models (Member 2) and AI models will ingest.

## ⚙️ Features
* **Core Signals:** Heart Rate, SpO₂, Movement, Temperature, and Categorical Sleep State.
* **Physiological Realism:** Implements continuity, circadian rhythms, and state transitions (e.g., movement suppression during deep sleep).
* **Advanced HRV:** Uses an Autoregressive AR(1) process for natural, organic signal wandering instead of erratic white noise.
* **Neurological Profiles:** Currently supports `healthy` and `parkinsons` profiles.
  * *Parkinson's Profile:* Features autonomic rigidity (stiffer HR variance), reduced SpO₂ capacity, and a persistent, low-amplitude high-frequency tremor.

## 📂 Project Structure
* `physiology_engine/models.py`: Contains the mathematical models for each specific vital sign.
* `physiology_engine/physiology_simulator.py`: The main coordinator class that generates the synchronized time-series data.
* `generate_dataset.py`: A test script to generate and validate 24-hour datasets.

## 🚀 Quick Start & Installation

1. **Install dependencies:**
   Ensure you have the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Baseline Vitals:**
   You can easily import the engine into your own modules:

   ```python
   from physiology_engine.physiology_simulator import PhysiologySimulator

   # 1. Initialize the engine (Set age and patient_type)
   engine = PhysiologySimulator(age=65, patient_type='healthy', seed=42)

   # 2. Generate time-series data (returns a Pandas DataFrame)
   patient_df = engine.generate_vitals(duration_hours=24.0)

   # 3. Export for dataset pipeline
   patient_df.to_csv('baseline_patient_vitals.csv', index=False)
   ```

## 🤝 Handoff Notes for Member 2 (Disease Modeler)
The output DataFrame contains pure, baseline physiological data. You can ingest this DataFrame and apply your disease-specific delta overlays (e.g., sudden SpO₂ drops, fever spikes, or tachycardia) directly on top of these baseline continuous signals.

Data is sampled at 0.1 Hz (10-second intervals) by default to maintain high-fidelity continuity constraints.

## 📊 Sample Output
```csv
timestamp,heart_rate,spo2,movement,temperature,sleep_state
2026-04-03 15:25:04.935800,73.45,96.0,0.254,35.5,awake
2026-04-03 15:25:14.935800,72.11,96.0,0.027,35.5,awake
2026-04-03 15:25:24.935800,71.05,97.9,0.075,35.5,awake
...
```

## 🔬 Technical Implementation

### Heart Rate Model
- **AR(1) Process**: Natural HRV with mean-reverting behavior
- **Realistic Baselines**: 70-80 bpm awake, 52-60 bpm deep sleep
- **Continuity Constraints**: Maximum 8 bpm change per minute
- **Age Effects**: Older patients show slightly reduced HRV

### Movement Model
- **Sleep Suppression**: 98% reduction during deep sleep
- **Parkinson's Tremor**: Persistent 4.5 Hz oscillation (never fully disappears)
- **Natural Variability**: Realistic movement patterns throughout day

### SpO2 Model
- **Healthy Range**: 96-99% with normal variance
- **Parkinson's Range**: 94-97% with 30% reduced variance (autonomic rigidity)
- **Strict Capping**: Parkinson's patients capped at 97% maximum

### Temperature Model
- **Circadian Rhythm**: 35.5-37.6°C natural variation
- **Smooth Transitions**: No abrupt changes

## 🧪 Testing & Validation

Run the comprehensive test suite:
```bash
python test_final.py
```

Or generate sample datasets:
```bash
python generate_dataset.py
```

Both scripts validate:
- ✅ Realistic heart rate ranges (50-150 bpm)
- ✅ Natural HRV patterns
- ✅ Proper sleep state transitions
- ✅ Parkinson's-specific characteristics
- ✅ Signal continuity and smoothness

## 📈 Performance Characteristics

| Metric | Healthy Patient | Parkinson's Patient |
|--------|----------------|-------------------|
| Heart Rate Range | 50-90 bpm | 50-95 bpm |
| HRV (std) | ~13 bpm | ~14 bpm (stiffer) |
| SpO2 Range | 96-99% | 94-97% |
| Movement Avg | 0.241 | 0.271 |
| Tremor Indicator | 0.130 | 0.141 (persistent) |

## 🔄 Version History

### MVP v1.0 (Current)
- ✅ Core physiological signal generation
- ✅ AR(1) HRV implementation
- ✅ Parkinson's disease modeling
- ✅ Perfect initialization (no startup glitch)
- ✅ Comprehensive validation suite
- ✅ Production-ready codebase

## 🤝 Contributing

This is Member 1 of the Mini ICU project. For contributions:
1. Maintain the modular structure
2. Preserve physiological accuracy
3. Update tests for new features
4. Follow the existing code style

## 📞 Support

For questions about the physiology engine implementation or integration with disease models (Member 2), please refer to the handoff notes above or review the comprehensive test suite for usage examples.

***

### Pro-Tip for your Terminal:
Since you are editing the README directly on the GitHub website, your *local* folder on your computer won't have this file yet. 

Before you start coding anything new locally in Windsurf, run this command in your terminal to download the README you just made online back to your computer:
```bash
git pull origin main
```

That keeps everything perfectly in sync. Let me know when you've got it committed!
