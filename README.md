# Stepped Wah POC Implementation

Python implementation of a "wah" guitar effect, based on filter steps instead of sweeps.
The "pedal" is stepped up and down at regular intervals to demonstrate the effect.

## Usage

`CHUNK_SIZE` - Size of each audio processing block
`SAMPLE_RATE` - Sample rate of .wav file input
`FREQ_LOW` - Lowest frequency of filter
`FREQ_HIGH` - Highest frequency of filter
`FREQ_STEPS` - Filter steps between FREQ_LOW and FREQ_HIGH
`CROSSFADE_SAMPLES` - Samples crossfaded during filter step transition
`STEP_SIZE` - Duration of each filter step (effect rate)
