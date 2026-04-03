"""
Main physiology simulator class for Mini ICU project.
Coordinates all physiological models to generate realistic vital signs data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from .models import HeartRateModel, MovementModel, TemperatureModel, SpO2Model


class PhysiologySimulator:
    """
    Main simulator class that coordinates all physiological models.
    
    Generates realistic vital signs data including:
    - Heart rate with circadian rhythm and HRV
    - Movement with sleep patterns and Parkinson's tremor
    - Body temperature with circadian variation
    - SpO2 with correlations to other signals
    
    Parameters:
    -----------
    age : int
        Patient age in years
    patient_type : str
        'healthy' or 'parkinsons'
    seed : int
        Random seed for reproducible results
    sampling_rate : float
        Sampling rate in Hz (default: 0.1 Hz = 10 second intervals)
    """
    
    def __init__(self, age: int = 65, patient_type: str = 'healthy', 
                 seed: int = 42, sampling_rate: float = 0.1):
        self.age = age
        self.patient_type = patient_type
        self.seed = seed
        self.sampling_rate = sampling_rate
        
        # Validate inputs
        if patient_type not in ['healthy', 'parkinsons']:
            raise ValueError("patient_type must be 'healthy' or 'parkinsons'")
        
        if age < 0 or age > 120:
            raise ValueError("age must be between 0 and 120")
        
        # Initialize individual models
        self.hr_model = HeartRateModel(age, patient_type, seed)
        self.movement_model = MovementModel(patient_type, seed + 1)
        self.temp_model = TemperatureModel(age, seed + 2)
        self.spo2_model = SpO2Model(age, patient_type, seed + 3)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
    
    def generate_vitals(self, duration_hours: float = 24.0, 
                       start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate vital signs data for specified duration.
        
        Parameters:
        -----------
        duration_hours : float
            Duration of simulation in hours
        start_time : datetime, optional
            Start time for simulation (default: current time)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: timestamp, heart_rate, spo2, movement, temperature, sleep_state
        """
        # Set start time
        if start_time is None:
            start_time = datetime.now()
        
        # Generate timestamps
        n_samples = int(duration_hours * 3600 * self.sampling_rate)
        timestamps = np.arange(n_samples) / self.sampling_rate
        datetime_timestamps = [start_time + timedelta(seconds=t) for t in timestamps]
        
        # Generate movement and sleep state first (they influence other signals)
        movement, sleep_state = self.movement_model.generate(timestamps)
        
        # Generate heart rate (influenced by movement and sleep)
        heart_rate = self.hr_model.generate(timestamps, movement, sleep_state)
        
        # Generate temperature (influenced by movement)
        temperature = self.temp_model.generate(timestamps, movement)
        
        # Generate SpO2 (influenced by movement and heart rate)
        spo2 = self.spo2_model.generate(timestamps, movement, heart_rate)
        
        # Create DataFrame
        data = {
            'timestamp': datetime_timestamps,
            'heart_rate': heart_rate,
            'spo2': spo2,
            'movement': movement,
            'temperature': temperature,
            'sleep_state': sleep_state
        }
        
        df = pd.DataFrame(data)
        
        # Add metadata
        df.attrs['age'] = self.age
        df.attrs['patient_type'] = self.patient_type
        df.attrs['sampling_rate'] = self.sampling_rate
        df.attrs['duration_hours'] = duration_hours
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for the vital signs data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame from generate_vitals()
        
        Returns:
        --------
        dict
            Summary statistics for each vital sign
        """
        stats = {}
        
        # Heart rate statistics
        stats['heart_rate'] = {
            'mean': df['heart_rate'].mean(),
            'std': df['heart_rate'].std(),
            'min': df['heart_rate'].min(),
            'max': df['heart_rate'].max(),
            'resting_hr': df[df['sleep_state'] == 'deep_sleep']['heart_rate'].mean() if (df['sleep_state'] == 'deep_sleep').any() else None
        }
        
        # SpO2 statistics
        stats['spo2'] = {
            'mean': df['spo2'].mean(),
            'std': df['spo2'].std(),
            'min': df['spo2'].min(),
            'max': df['spo2'].max()
        }
        
        # Movement analysis
        stats['movement'] = {
            'mean': df['movement'].mean(),
            'std': df['movement'].std(),
            'max': df['movement'].max(),
            'sleep_movement': df[df['sleep_state'] == 'deep_sleep']['movement'].mean() if (df['sleep_state'] == 'deep_sleep').any() else None,
            'awake_movement': df[df['sleep_state'] == 'awake']['movement'].mean() if (df['sleep_state'] == 'awake').any() else None
        }
        
        # Temperature statistics
        stats['temperature'] = {
            'mean': df['temperature'].mean(),
            'std': df['temperature'].std(),
            'min': df['temperature'].min(),
            'max': df['temperature'].max()
        }
        
        # Sleep statistics
        sleep_state = df['sleep_state']
        total_sleep = ((sleep_state == 'deep_sleep') | (sleep_state == 'light_sleep')).sum() / (self.sampling_rate * 3600)
        deep_sleep = (sleep_state == 'deep_sleep').sum() / (self.sampling_rate * 3600)
        stats['sleep'] = {
            'total_sleep_hours': total_sleep,
            'deep_sleep_hours': deep_sleep,
            'light_sleep_hours': total_sleep - deep_sleep,
            'sleep_efficiency': ((sleep_state == 'deep_sleep') | (sleep_state == 'light_sleep')).mean()
        }
        
        return stats
    
    def validate_signals(self, df: pd.DataFrame) -> dict:
        """
        Validate that generated signals are within physiological ranges.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame from generate_vitals()
        
        Returns:
        --------
        dict
            Validation results with any warnings
        """
        validation = {'warnings': [], 'errors': []}
        
        # Heart rate validation
        hr_min, hr_max = df['heart_rate'].min(), df['heart_rate'].max()
        if hr_min < 40:
            validation['warnings'].append(f"Heart rate too low: {hr_min:.1f} bpm")
        if hr_max > (220 - self.age):
            validation['errors'].append(f"Heart rate exceeds maximum: {hr_max:.1f} bpm")
        
        # SpO2 validation
        spo2_min = df['spo2'].min()
        if spo2_min < 90:
            validation['warnings'].append(f"SpO2 critically low: {spo2_min:.1f}%")
        elif spo2_min < 95:
            validation['warnings'].append(f"SpO2 low: {spo2_min:.1f}%")
        
        # Temperature validation
        temp_min, temp_max = df['temperature'].min(), df['temperature'].max()
        if temp_min < 36.0:
            validation['warnings'].append(f"Temperature too low: {temp_min:.1f}°C")
        if temp_max > 38.0:
            validation['warnings'].append(f"Temperature elevated: {temp_max:.1f}°C")
        
        # Movement validation
        if df['movement'].max() > 1.0:
            validation['errors'].append("Movement exceeds 0-1 scale")
        
        # Sleep state validation
        sleep_state = df['sleep_state']
        if not ((sleep_state == 'deep_sleep') | (sleep_state == 'light_sleep') | (sleep_state == 'awake')).all():
            validation['errors'].append("Sleep state not in 'deep_sleep', 'light_sleep', or 'awake' range")
        
        # Correlation checks
        hr_movement_corr = df['heart_rate'].corr(df['movement'])
        if hr_movement_corr < 0.1:
            validation['warnings'].append(f"Low heart rate-movement correlation: {hr_movement_corr:.2f}")
        
        return validation
    
    def __repr__(self) -> str:
        """String representation of the simulator."""
        return (f"PhysiologySimulator(age={self.age}, patient_type='{self.patient_type}', "
                f"seed={self.seed}, sampling_rate={self.sampling_rate} Hz)")


# Convenience function for quick simulation
def quick_simulate(age: int = 65, patient_type: str = 'healthy', 
                  duration_hours: float = 24.0, seed: int = 42) -> pd.DataFrame:
    """
    Quick simulation function for testing and demonstrations.
    
    Parameters:
    -----------
    age : int
        Patient age in years
    patient_type : str
        'healthy' or 'parkinsons'
    duration_hours : float
        Duration of simulation in hours
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Generated vital signs data
    """
    simulator = PhysiologySimulator(age=age, patient_type=patient_type, seed=seed)
    return simulator.generate_vitals(duration_hours=duration_hours)
