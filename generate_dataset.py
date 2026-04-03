"""
Test script for Mini ICU physiology engine.
Generates sample datasets and demonstrates the simulator functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from physiology_engine.physiology_simulator import PhysiologySimulator


def test_healthy_patient():
    """Test simulation for a healthy patient."""
    print("=" * 60)
    print("Testing Healthy Patient Simulation")
    print("=" * 60)
    
    # Create simulator for healthy 65-year-old
    simulator = PhysiologySimulator(age=65, patient_type='healthy', seed=42)
    
    # Generate 24 hours of data
    df = simulator.generate_vitals(duration_hours=24.0)
    
    # Display basic info
    print(f"Generated {len(df)} samples over 24 hours")
    print(f"Sampling rate: {simulator.sampling_rate} Hz ({1/simulator.sampling_rate:.1f} second intervals)")
    print()
    
    # Show first 10 rows
    print("First 10 rows of generated data:")
    print(df.head(10))
    print()
    
    # Get summary statistics
    stats = simulator.get_summary_stats(df)
    print("Summary Statistics:")
    for vital, values in stats.items():
        print(f"\n{vital.upper()}:")
        for key, value in values.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    print()
    
    # Validate signals
    validation = simulator.validate_signals(df)
    if validation['errors']:
        print("ERRORS FOUND:")
        for error in validation['errors']:
            print(f"  - {error}")
    if validation['warnings']:
        print("WARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    if not validation['errors'] and not validation['warnings']:
        print("✓ All signals within physiological ranges")
    
    return df


def test_parkinsons_patient():
    """Test simulation for a Parkinson's patient."""
    print("\n" + "=" * 60)
    print("Testing Parkinson's Patient Simulation")
    print("=" * 60)
    
    # Create simulator for Parkinson's patient
    simulator = PhysiologySimulator(age=72, patient_type='parkinsons', seed=123)
    
    # Generate 24 hours of data
    df = simulator.generate_vitals(duration_hours=24.0)
    
    # Display basic info
    print(f"Generated {len(df)} samples over 24 hours")
    print()
    
    # Show first 10 rows
    print("First 10 rows of generated data:")
    print(df.head(10))
    print()
    
    # Get summary statistics
    stats = simulator.get_summary_stats(df)
    print("Summary Statistics:")
    for vital, values in stats.items():
        print(f"\n{vital.upper()}:")
        for key, value in values.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    print()
    
    # Validate signals
    validation = simulator.validate_signals(df)
    if validation['errors']:
        print("ERRORS FOUND:")
        for error in validation['errors']:
            print(f"  - {error}")
    if validation['warnings']:
        print("WARNINGS:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    if not validation['errors'] and not validation['warnings']:
        print("✓ All signals within physiological ranges")
    
    return df


def compare_patient_types():
    """Compare healthy vs Parkinson's patients."""
    print("\n" + "=" * 60)
    print("Comparing Healthy vs Parkinson's Patients")
    print("=" * 60)
    
    # Generate data for both patient types
    healthy_sim = PhysiologySimulator(age=70, patient_type='healthy', seed=42)
    parkinsons_sim = PhysiologySimulator(age=70, patient_type='parkinsons', seed=42)
    
    healthy_df = healthy_sim.generate_vitals(duration_hours=24.0)
    parkinsons_df = parkinsons_sim.generate_vitals(duration_hours=24.0)
    
    # Compare key metrics
    print("Comparison of Key Metrics (same age, different conditions):")
    print()
    
    metrics = {
        'Heart Rate (bpm)': ('heart_rate', 'mean'),
        'HRV (std)': ('heart_rate', 'std'),
        'SpO2 (%)': ('spo2', 'mean'),
        'Movement (0-1)': ('movement', 'mean'),
        'Temperature (°C)': ('temperature', 'mean'),
        'Sleep Hours': ('sleep', 'total_sleep_hours')
    }
    
    for metric_name, (col, stat) in metrics.items():
        if col == 'sleep':
            healthy_val = healthy_sim.get_summary_stats(healthy_df)[col][stat]
            parkinsons_val = parkinsons_sim.get_summary_stats(parkinsons_df)[col][stat]
        else:
            healthy_val = healthy_df[col].mean() if stat == 'mean' else healthy_df[col].std()
            parkinsons_val = parkinsons_df[col].mean() if stat == 'mean' else parkinsons_df[col].std()
        
        print(f"{metric_name:20} | Healthy: {healthy_val:7.2f} | Parkinson's: {parkinsons_val:7.2f} | Diff: {parkinsons_val - healthy_val:+6.2f}")
    
    return healthy_df, parkinsons_df


def test_different_ages():
    """Test simulation across different age groups."""
    print("\n" + "=" * 60)
    print("Testing Different Age Groups")
    print("=" * 60)
    
    ages = [25, 45, 65, 85]
    
    print("Age-related variations in vital signs:")
    print()
    print(f"{'Age':<5} {'HR (bpm)':<10} {'SpO2 (%)':<10} {'Temp (°C)':<10} {'Sleep (hrs)':<10}")
    print("-" * 55)
    
    for age in ages:
        simulator = PhysiologySimulator(age=age, patient_type='healthy', seed=age)
        df = simulator.generate_vitals(duration_hours=24.0)
        stats = simulator.get_summary_stats(df)
        
        hr = stats['heart_rate']['mean']
        spo2 = stats['spo2']['mean']
        temp = stats['temperature']['mean']
        sleep = stats['sleep']['total_sleep_hours']
        
        print(f"{age:<5} {hr:<10.1f} {spo2:<10.1f} {temp:<10.2f} {sleep:<10.1f}")


def analyze_signal_realism(df: pd.DataFrame, patient_type: str):
    """Analyze how realistic the generated signals are."""
    print(f"\n" + "=" * 60)
    print(f"Signal Realism Analysis - {patient_type.title()} Patient")
    print("=" * 60)
    
    # Heart rate analysis
    hr = df['heart_rate']
    print("\nHeart Rate Analysis:")
    print(f"  Range: {hr.min():.1f} - {hr.max():.1f} bpm")
    print(f"  Circadian variation: {hr.max() - hr.min():.1f} bpm")
    print(f"  HRV (std): {hr.std():.2f} bpm")
    
    # Sleep-related HR changes
    sleep_periods = df[df['sleep_state'] == 'deep_sleep']
    awake_periods = df[df['sleep_state'] == 'awake']
    if len(sleep_periods) > 0 and len(awake_periods) > 0:
        sleep_hr = sleep_periods['heart_rate'].mean()
        awake_hr = awake_periods['heart_rate'].mean()
        print(f"  Sleep HR: {sleep_hr:.1f} bpm")
        print(f"  Awake HR: {awake_hr:.1f} bpm")
        print(f"  Sleep-Awake difference: {awake_hr - sleep_hr:.1f} bpm")
    
    # Movement analysis
    movement = df['movement']
    print(f"\nMovement Analysis:")
    print(f"  Average: {movement.mean():.3f}")
    print(f"  Peak: {movement.max():.3f}")
    
    if patient_type == 'parkinsons':
        # Check for tremor characteristics
        movement_diff = np.diff(movement)
        high_freq_content = np.std(movement_diff)
        print(f"  Tremor indicator (high-freq content): {high_freq_content:.4f}")
    
    # Sleep analysis
    sleep_state = df['sleep_state']
    total_sleep = ((sleep_state == 'deep_sleep') | (sleep_state == 'light_sleep')).sum() / (0.1 * 3600)  # Convert to hours
    deep_sleep = (sleep_state == 'deep_sleep').sum() / (0.1 * 3600)
    print(f"\nSleep Analysis:")
    print(f"  Total sleep: {total_sleep:.1f} hours")
    print(f"  Deep sleep: {deep_sleep:.1f} hours")
    print(f"  Sleep efficiency: {((sleep_state == 'deep_sleep') | (sleep_state == 'light_sleep')).mean():.2%}")
    
    # Correlation analysis
    print(f"\nSignal Correlations:")
    hr_movement_corr = df['heart_rate'].corr(df['movement'])
    hr_temp_corr = df['heart_rate'].corr(df['temperature'])
    movement_spo2_corr = df['movement'].corr(df['spo2'])
    
    print(f"  HR-Movement: {hr_movement_corr:.3f}")
    print(f"  HR-Temperature: {hr_temp_corr:.3f}")
    print(f"  Movement-SpO2: {movement_spo2_corr:.3f}")
    
    # Realism assessment
    print(f"\nRealism Assessment:")
    realism_score = 0
    
    # Heart rate realism
    if 40 <= hr.min() and hr.max() <= (220 - 65):  # Assuming age 65
        realism_score += 1
        print("  [OK] Heart rate within physiological range")
    else:
        print("  [FAIL] Heart rate out of physiological range")
    
    # Circadian rhythm
    if hr.max() - hr.min() >= 5:  # At least 5 bpm variation
        realism_score += 1
        print("  [OK] Shows circadian rhythm")
    else:
        print("  [FAIL] Limited circadian variation")
    
    # Sleep-movement coupling
    deep_sleep_movement = df[df['sleep_state'] == 'deep_sleep']['movement']
    if len(deep_sleep_movement) > 0 and deep_sleep_movement.mean() < 0.1:
        realism_score += 1
        print("  [OK] Movement suppressed during sleep")
    else:
        print("  [FAIL] Excessive movement during sleep")
    
    # SpO2 stability
    spo2 = df['spo2']
    if 95 <= spo2.mean() <= 99 and spo2.std() < 2:
        realism_score += 1
        print("  [OK] SpO2 stable and normal")
    else:
        print("  [FAIL] SpO2 unrealistic")
    
    # Temperature variation
    temp = df['temperature']
    if 36.0 <= temp.mean() <= 37.5 and temp.max() - temp.min() >= 0.3:
        realism_score += 1
        print("  [OK] Temperature shows normal variation")
    else:
        print("  [FAIL] Temperature unrealistic")
    
    print(f"\nOverall Realism Score: {realism_score}/5")


def main():
    """Main test function."""
    print("Mini ICU Physiology Engine - Test Suite")
    print("=" * 60)
    
    try:
        # Test healthy patient
        healthy_df = test_healthy_patient()
        
        # Test Parkinson's patient
        parkinsons_df = test_parkinsons_patient()
        
        # Compare patient types
        compare_patient_types()
        
        # Test different ages
        test_different_ages()
        
        # Analyze signal realism
        analyze_signal_realism(healthy_df, 'healthy')
        analyze_signal_realism(parkinsons_df, 'parkinsons')
        
        print("\n" + "=" * 60)
        print("[OK] All tests completed successfully!")
        print("=" * 60)
        
        # Save sample datasets
        healthy_df.to_csv('healthy_patient_sample.csv', index=False)
        parkinsons_df.to_csv('parkinsons_patient_sample.csv', index=False)
        print("\nSample datasets saved:")
        print("  - healthy_patient_sample.csv")
        print("  - parkinsons_patient_sample.csv")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
