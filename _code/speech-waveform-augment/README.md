# speech-waveform-augment

Lightweight waveform augmentation utilities used across our speech pipelines:
speed perturbation, additive noise and SpecAugment-style masking. It is a
shared preprocessing helper and does not implement any paper's model or
training recipe.

## Layout

- `src/augment.py` - augmentation transforms
- `src/io_utils.py` - audio loading helpers
- `tests/test_augment.py` - smoke tests
