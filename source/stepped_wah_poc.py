import soundfile as sf
import numpy as np
from scipy import signal
import os

CHUNK_SIZE = 1024
SAMPLE_RATE = 48000

STEP_DURATION = 5
CROSSFADE_SAMPLES = 256
FREQ_LOW = 200
FREQ_HIGH = 2000
FREQ_STEPS = 5

class SteppedWahPedal:
    def __init__(self, center_frequencies, q=5, crossfade_samples=CROSSFADE_SAMPLES):
        self.center_frequencies = center_frequencies
        self.q = q
        self.current_step = 0
        self.direction = 1
        self.crossfade_samples = crossfade_samples
        self.crossfade_remaining = 0
        
        # Current filter state
        self.current_filter = None
        self.zi_left = None
        self.zi_right = None
        
        # Previous filter state (for crossfading)
        self.prev_filter = None
        self.prev_zi_left = None
        self.prev_zi_right = None
        
        # Crossfade weights (shared between channels)
        self.fade_in_weights = None
        self.fade_out_weights = None
        
        # Initialize the first filter
        self._update_filter(SAMPLE_RATE)
        
    def _update_filter(self, samplerate):
        center_freq = self.center_frequencies[self.current_step]
        
        # Store previous filter and states before updating
        if self.current_filter is not None:
            self.prev_filter = self.current_filter
            self.prev_zi_left = self.zi_left.copy()  # Make copies to prevent state mixing
            self.prev_zi_right = self.zi_right.copy()
        
        # Create new filter
        self.current_filter = signal.iirpeak(center_freq, self.q, samplerate)
        
        # Initialize or reinitialize filter states
        if self.zi_left is None:
            self.zi_left = signal.lfilter_zi(*self.current_filter)
            self.zi_right = signal.lfilter_zi(*self.current_filter)
        else:
            # Reinitialize states when filter changes
            self.zi_left = signal.lfilter_zi(*self.current_filter)
            self.zi_right = signal.lfilter_zi(*self.current_filter)
            
        # Start crossfade
        if self.prev_filter is not None:
            self.crossfade_remaining = self.crossfade_samples
            self.fade_in_weights = None  # Will be calculated in process()
            self.fade_out_weights = None
    
    def _calculate_crossfade_weights(self, chunk_size):
        # Calculate crossfade weights for the current chunk
        if self.crossfade_remaining <= 0:
            return None, None
            
        fade_out = np.linspace(1, 0, min(chunk_size, self.crossfade_remaining), endpoint=True)
        if chunk_size > self.crossfade_remaining:
            fade_out = np.pad(fade_out, (0, chunk_size - self.crossfade_remaining))
        fade_in = 1 - fade_out
        
        return fade_in, fade_out
        
    def process_stereo(self, left_data, right_data, samplerate):
        #Process both channels together to ensure synchronized crossfading
        chunk_size = len(left_data)
        
        # Process current filter
        filtered_left, self.zi_left = signal.lfilter(
            self.current_filter[0],
            self.current_filter[1],
            left_data,
            zi=self.zi_left
        )
        
        filtered_right, self.zi_right = signal.lfilter(
            self.current_filter[0],
            self.current_filter[1],
            right_data,
            zi=self.zi_right
        )
        
        # Handle crossfading if needed
        if self.crossfade_remaining > 0 and self.prev_filter is not None:
            # Process previous filter
            prev_filtered_left, self.prev_zi_left = signal.lfilter(
                self.prev_filter[0],
                self.prev_filter[1],
                left_data,
                zi=self.prev_zi_left
            )
            
            prev_filtered_right, self.prev_zi_right = signal.lfilter(
                self.prev_filter[0],
                self.prev_filter[1],
                right_data,
                zi=self.prev_zi_right
            )
            
            # Calculate crossfade weights if not already done for this chunk
            if self.fade_in_weights is None or len(self.fade_in_weights) != chunk_size:
                self.fade_in_weights, self.fade_out_weights = self._calculate_crossfade_weights(chunk_size)
            
            # Apply crossfade
            filtered_left = (prev_filtered_left * self.fade_out_weights) + (filtered_left * self.fade_in_weights)
            filtered_right = (prev_filtered_right * self.fade_out_weights) + (filtered_right * self.fade_in_weights)
            
            # Update crossfade counter
            self.crossfade_remaining = max(0, self.crossfade_remaining - chunk_size)
            
        return filtered_left, filtered_right
        
    def step(self, samplerate):
        self.current_step += self.direction
        if self.current_step == len(self.center_frequencies) - 1 or self.current_step == 0:
            self.direction *= -1
        self._update_filter(samplerate)

def process_file(input_file, output_file):
    audio_data, samplerate = sf.read(input_file)
    
    # Create our pedal with crossfading
    center_freqs = np.logspace(np.log10(FREQ_LOW), np.log10(FREQ_HIGH), FREQ_STEPS)
    pedal = SteppedWahPedal(center_freqs, crossfade_samples=CROSSFADE_SAMPLES)
    
    print("Processing audio file...")
    print("Frequency steps:", ", ".join(f"{freq:.2f}" for freq in center_freqs))
    
    chunk_size = CHUNK_SIZE
    num_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
    processed_data = np.zeros_like(audio_data)
    
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        
        if num_channels == 2:
            # Process both channels together
            processed_left, processed_right = pedal.process_stereo(
                chunk[:, 0],
                chunk[:, 1],
                samplerate
            )
            processed_chunk = np.column_stack((processed_left, processed_right))
        else:
            # For mono, just process as left channel
            processed_chunk = pedal.process_stereo(chunk, chunk, samplerate)[0]
            
        processed_data[i:i+chunk_size] = processed_chunk
        
        if i // chunk_size % STEP_DURATION == 0:
            pedal.step(samplerate)
            print(f"Stepped to {pedal.center_frequencies[pedal.current_step]:.2f} Hz, i = {i}")
    
    # Normalize the output to prevent clipping
    processed_data = np.tanh(processed_data)
    processed_data = processed_data / np.max(np.abs(processed_data))
    
    # Adjust volume
    processed_data *= 0.8
    
    sf.write(output_file, processed_data, samplerate)
    print("Finished processing audio file.")
    print(f"Output saved as '{output_file}'")

if __name__ == "__main__":
    input_file = "source/input.wav"
    output_file = "source/output_stepped_wah.wav"
    process_file(input_file, output_file)