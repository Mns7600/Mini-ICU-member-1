"""
Physiology signal models for Mini ICU simulation.
Implements realistic biomedical signals with proper correlations and physiological constraints.
"""

import numpy as np
from typing import Tuple, Optional


class HeartRateModel:
    """Realistic heart rate simulation with circadian rhythm, activity, and HRV."""
    
    def __init__(self, age: int, patient_type: str = 'healthy', seed: int = 42):
        self.age = age
        self.patient_type = patient_type
        self.rng = np.random.default_rng(seed)
        
        # Realistic heart rate baselines for 65-year-old
        if patient_type == 'healthy':
            self.awake_baseline = 75  # 70-80 bpm awake baseline
            self.deep_sleep_baseline = 55  # 52-60 bpm deep sleep
            self.hrv_amplitude = 5  # ±5 bpm natural variability
            self.ar_coefficient = 0.7  # AR(1) coefficient for healthy (moderate memory)
            self.hrv_mean_reversion = 0.1  # Mean reversion strength
        else:  # parkinsons
            self.awake_baseline = 80  # Slightly higher resting HR
            self.deep_sleep_baseline = 58  # Slightly higher sleep HR
            self.hrv_amplitude = 3  # Reduced HRV in Parkinson's
            self.ar_coefficient = 0.9  # Higher AR coefficient (stiffer, less variable)
            self.hrv_mean_reversion = 0.05  # Slower mean reversion (reduced variability)
        
        # Maximum heart rate (220 - age formula)
        self.max_hr = 220 - age
        
        # Minimum heart rate (never below 50 bpm)
        self.min_hr = 50
        
        # Circadian rhythm parameters
        self.circadian_amplitude = 5  # ±5 bpm variation
        self.circadian_phase = -np.pi/2  # Peak at 14:00 (2 PM)
        
        # Continuity constraints (max change per minute)
        self.max_change_per_minute = 8  # bpm (more conservative)
        self.sampling_rate = 0.1  # 10 second intervals
        self.max_change_per_sample = self.max_change_per_minute / 6
        
        # Age effects on HRV
        self.age_hrv_reduction = max(0, (age - 50) * 0.02)
        
    def generate(self, timestamps: np.ndarray, movement: np.ndarray, 
                sleep_state: np.ndarray) -> np.ndarray:
        """Generate heart rate signal with AR(1) HRV and perfect initialization."""
        n_samples = len(timestamps)
        hours = (timestamps - timestamps[0]) / 3600
        
        # Initialize with realistic baseline based on initial sleep state
        hr_signal = np.zeros(n_samples)
        
        # AR(1) process for HRV
        hrv_process = np.zeros(n_samples)
        
        # Set initial value based on first sleep state
        if isinstance(sleep_state[0], str):
            if sleep_state[0] == 'deep_sleep':
                baseline = self.deep_sleep_baseline
            elif sleep_state[0] == 'light_sleep':
                baseline = (self.awake_baseline + self.deep_sleep_baseline) / 2
            else:  # awake
                baseline = self.awake_baseline
        else:
            # Fallback for numeric sleep state
            baseline = self.awake_baseline if sleep_state[0] < 0.5 else self.deep_sleep_baseline
        
        # Calculate exact target for t=0 (including circadian offset and initial state)
        circadian_0 = self.circadian_amplitude * np.sin(2 * np.pi * hours[0] / 24 + self.circadian_phase)
        movement_response_0 = 15 * movement[0] ** 1.5
        hrv_0 = self.rng.normal(0, self.hrv_amplitude)
        
        # Set initial HR to exact target (perfect initialization)
        hr_signal[0] = baseline + circadian_0 + movement_response_0 + hrv_0
        hrv_process[0] = hrv_0
        
        # Generate signal sample by sample for realistic continuity
        for i in range(1, n_samples):
            # Determine baseline based on current sleep state
            if isinstance(sleep_state[i], str):
                if sleep_state[i] == 'deep_sleep':
                    baseline = self.deep_sleep_baseline
                elif sleep_state[i] == 'light_sleep':
                    baseline = (self.awake_baseline + self.deep_sleep_baseline) / 2
                else:  # awake
                    baseline = self.awake_baseline
            else:
                baseline = self.awake_baseline if sleep_state[i] < 0.5 else self.deep_sleep_baseline
            
            # Add circadian rhythm
            circadian = self.circadian_amplitude * np.sin(2 * np.pi * hours[i] / 24 + self.circadian_phase)
            
            # Add movement response
            movement_response = 15 * movement[i] ** 1.5
            
            # AR(1) process for HRV (mean-reverting)
            hrv_noise = self.rng.normal(0, self.hrv_amplitude * 0.3)  # Smaller noise component
            hrv_process[i] = (self.ar_coefficient * hrv_process[i-1] + 
                             self.hrv_mean_reversion * (0 - hrv_process[i-1]) + 
                             hrv_noise)
            
            # Calculate target HR
            target_hr = baseline + circadian + movement_response + hrv_process[i]
            
            # Enforce continuity constraints
            max_change = self.max_change_per_sample
            hr_change = np.clip(target_hr - hr_signal[i-1], -max_change, max_change)
            hr_signal[i] = hr_signal[i-1] + hr_change
        
        # Final constraints
        hr_signal = np.clip(hr_signal, self.min_hr, self.max_hr)
        
        return hr_signal
    
    def _activity_response(self, movement: np.ndarray, sleep_state: np.ndarray) -> np.ndarray:
        """Generate heart rate response to movement and sleep state with stronger correlation."""
        # This method is now integrated into the main generate() method
        # for better continuity and immediate realistic values
        pass
    
    def _generate_hrv(self, n_samples: int, movement: np.ndarray) -> np.ndarray:
        """Generate realistic heart rate variability with natural fluctuations."""
        # This method is now integrated into the main generate() method
        # for more natural time-varying HRV
        pass
    
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window <= 1:
            return signal
        
        # Use convolution for efficient moving average
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')
    
    def _enforce_continuity(self, hr_signal: np.ndarray) -> np.ndarray:
        """Enforce maximum rate of change for heart rate continuity."""
        # Calculate differences between consecutive samples
        diffs = np.diff(hr_signal)
        
        # Clip differences to maximum allowed change
        clipped_diffs = np.clip(diffs, -self.max_change_per_sample, self.max_change_per_sample)
        
        # Reconstruct signal with clipped differences
        corrected_signal = np.zeros_like(hr_signal)
        corrected_signal[0] = hr_signal[0]
        
        for i in range(1, len(hr_signal)):
            corrected_signal[i] = corrected_signal[i-1] + clipped_diffs[i-1]
        
        return corrected_signal


class MovementModel:
    """Realistic movement simulation with sleep patterns and Parkinson's tremor."""
    
    def __init__(self, patient_type: str = 'healthy', seed: int = 42):
        self.patient_type = patient_type
        self.rng = np.random.default_rng(seed)
        
        # Sleep pattern parameters
        self.sleep_start_hour = 23  # 11 PM
        self.sleep_end_hour = 7     # 7 AM
        self.sleep_transition_duration = 1  # 1 hour transition
        
        # Parkinson's tremor parameters
        if patient_type == 'parkinsons':
            self.tremor_amplitude = 0.15  # 15% of movement scale
            self.tremor_frequency = 4.5    # 4.5 Hz typical Parkinson's tremor
        else:
            self.tremor_amplitude = 0
            self.tremor_frequency = 0
    
    def generate(self, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate movement signal and categorical sleep state."""
        n_samples = len(timestamps)
        hours = (timestamps - timestamps[0]) / 3600
        
        # Generate categorical sleep state
        sleep_state = self._generate_sleep_state(hours)
        
        # Base movement (strongly reduced during sleep)
        base_movement = self._generate_base_movement(sleep_state, n_samples)
        
        # Add Parkinson's tremor if applicable
        if self.patient_type == 'parkinsons':
            tremor = self._generate_tremor(n_samples)
            movement = base_movement + tremor
        else:
            movement = base_movement
        
        # Normalize to 0-1 scale
        movement = np.clip(movement, 0, 1)
        
        return movement, sleep_state
    
    def _generate_sleep_state(self, hours: np.ndarray) -> np.ndarray:
        """Generate realistic sleep-wake cycle starting in awake/light_sleep state."""
        sleep_state = []
        
        # Start in awake state for realistic simulation
        for i, hour in enumerate(hours):
            # Normalize hour to 0-24 range
            hour_mod = hour % 24
            
            # Check if we're in sleep period (11 PM - 7 AM)
            if self.sleep_start_hour <= hour_mod or hour_mod < self.sleep_end_hour:
                # Sleep period - determine depth with smooth transitions
                if self.sleep_start_hour <= hour_mod:
                    # Evening transition
                    progress = (hour_mod - self.sleep_start_hour) / self.sleep_transition_duration
                else:
                    # Morning transition
                    progress = 1 - (hour_mod - self.sleep_end_hour) / self.sleep_transition_duration
                
                # For first hour of simulation, start in awake or light_sleep
                if i < 36:  # First 6 minutes (36 samples at 10 sec intervals)
                    if i < 18:  # First 3 minutes - awake
                        state = 'awake'
                    else:  # Next 3 minutes - light_sleep transition
                        state = 'light_sleep'
                else:
                    # Normal sleep state determination
                    if progress < 0.3:
                        state = 'awake'
                    elif progress < 0.7:
                        state = 'light_sleep'
                    else:
                        state = 'deep_sleep'
            else:
                # Awake period
                state = 'awake'
            
            sleep_state.append(state)
        
        return np.array(sleep_state)
    
    def _generate_base_movement(self, sleep_state: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate base movement patterns with sleep suppression."""
        # Random movement during wake periods
        wake_movement = self.rng.beta(2, 5, n_samples) * 0.7  # Mostly low movement
        
        # Add periodic movement (walking, shifting position)
        periodic_movement = 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 600) ** 2
        periodic_movement = np.maximum(0, periodic_movement)
        
        # Combine and suppress during sleep with categorical states
        base_movement = wake_movement + periodic_movement
        
        for i, state in enumerate(sleep_state):
            if state == 'deep_sleep':
                base_movement[i] *= 0.02  # Almost no movement
            elif state == 'light_sleep':
                base_movement[i] *= 0.05  # Very little movement
            # 'awake' - no suppression
        
        return base_movement
    
    def _generate_tremor(self, n_samples: int) -> np.ndarray:
        """Generate persistent Parkinson's tremor that never fully goes away."""
        # Persistent tremor for Parkinson's patients
        self.tremor_amplitude = 0.10  # Constant low-amplitude tremor
        self.tremor_frequency = 4.5    # 4.5 Hz typical Parkinson's tremor
        
        # High-frequency tremor with minimal amplitude modulation
        tremor_time = np.arange(n_samples) / 10  # Assume 10 Hz sampling
        
        # Base tremor (persistent high-frequency oscillation)
        tremor = self.tremor_amplitude * np.sin(2 * np.pi * self.tremor_frequency * tremor_time)
        
        # Minimal amplitude modulation (tremor never fully goes away)
        modulation = 0.7 + 0.3 * np.sin(2 * np.pi * tremor_time / 300)  # Small variation
        tremor *= modulation
        
        # Add higher harmonics for more realistic tremor
        tremor += 0.4 * self.tremor_amplitude * np.sin(2 * np.pi * self.tremor_frequency * 2 * tremor_time)
        tremor += 0.2 * self.tremor_amplitude * np.sin(2 * np.pi * self.tremor_frequency * 3 * tremor_time)
        
        # Add some randomness for natural variation
        tremor += self.rng.normal(0, self.tremor_amplitude * 0.05, n_samples)
        
        # Ensure tremor is always positive (adds to movement) and never zero
        tremor = np.maximum(self.tremor_amplitude * 0.1, tremor)  # Minimum 10% of amplitude
        
        return tremor
    
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window <= 1:
            return signal
        
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')


class TemperatureModel:
    """Realistic body temperature simulation with circadian variation."""
    
    def __init__(self, age: int, seed: int = 42):
        self.age = age
        self.rng = np.random.default_rng(seed)
        
        # Base temperature (36.5-37.5°C for adults)
        self.base_temp = 37.0
        
        # Circadian variation (peak in evening, trough in early morning)
        self.circadian_amplitude = 0.5  # ±0.5°C variation
        self.circadian_phase = np.pi     # Peak around 18:00 (6 PM)
        
        # Age effect (older adults have slightly lower temps)
        self.age_adjustment = -0.01 * max(0, age - 65)
    
    def generate(self, timestamps: np.ndarray, movement: np.ndarray) -> np.ndarray:
        """Generate body temperature signal."""
        n_samples = len(timestamps)
        hours = (timestamps - timestamps[0]) / 3600
        
        # Circadian rhythm
        circadian = self.circadian_amplitude * np.sin(2 * np.pi * hours / 24 + self.circadian_phase)
        
        # Activity effect (slight increase with movement)
        activity_effect = 0.1 * movement
        
        # Base temperature
        base_temp = self.base_temp + self.age_adjustment + circadian + activity_effect
        
        # Add small noise
        noise = self.rng.normal(0, 0.05, n_samples)
        
        # Combine and smooth
        temp_signal = base_temp + noise
        temp_signal = self._smooth_signal(temp_signal, window=10)
        
        # Constrain to realistic range
        temp_signal = np.clip(temp_signal, 35.5, 38.5)
        
        return temp_signal
    
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window <= 1:
            return signal
        
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')


class SpO2Model:
    """Realistic SpO2 (blood oxygen saturation) simulation."""
    
    def __init__(self, age: int, patient_type: str = 'healthy', seed: int = 42):
        self.age = age
        self.patient_type = patient_type
        self.rng = np.random.default_rng(seed)
        
        # Realistic SpO2 baselines
        if patient_type == 'healthy':
            self.base_spo2 = 98  # 96-99% for healthy individuals
            self.spo2_range = (96, 99)
        else:  # parkinsons
            self.base_spo2 = 96  # 94-97% for Parkinson's patients
            self.spo2_range = (94, 97)
        
        # Age effect (slight decrease with age)
        self.age_adjustment = -0.01 * max(0, age - 70)
    
    def generate(self, timestamps: np.ndarray, movement: np.ndarray, 
                heart_rate: np.ndarray) -> np.ndarray:
        """Generate SpO2 signal with Parkinson's-specific constraints."""
        n_samples = len(timestamps)
        
        # Base SpO2 with realistic baseline
        base_spo2 = self.base_spo2 + self.age_adjustment
        
        # Movement effect (slight decrease during high activity)
        movement_effect = -0.3 * movement ** 1.5  # Non-linear effect, reduced magnitude
        
        # Heart rate correlation (high HR can slightly reduce SpO2)
        hr_normalized = (heart_rate - 60) / 80  # Normalize around resting HR
        hr_effect = -0.15 * np.maximum(0, hr_normalized)  # Reduced effect
        
        # Combine signals
        spo2_signal = base_spo2 + movement_effect + hr_effect
        
        # Add noise with Parkinson's-specific reduction
        if self.patient_type == 'parkinsons':
            # Reduce natural variance/noise by 30% for autonomic rigidity
            noise_amplitude = 0.2 * 0.7  # 30% reduction
            max_spo2 = 97.0  # Strict cap for Parkinson's
        else:
            noise_amplitude = 0.2
            max_spo2 = 99.0  # Normal cap for healthy
        
        noise = self.rng.normal(0, noise_amplitude, n_samples)
        spo2_signal += noise
        
        # Smooth slightly
        spo2_signal = self._smooth_signal(spo2_signal, window=5)
        
        # Constrain to realistic range for patient type
        spo2_signal = np.clip(spo2_signal, self.spo2_range[0], max_spo2)
        
        return spo2_signal
    
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Apply moving average smoothing."""
        if window <= 1:
            return signal
        
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')
