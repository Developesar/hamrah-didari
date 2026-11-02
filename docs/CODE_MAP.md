# Selected code and release scope

This release contains ten processing modules from the application, their direct support code and focused tests. It is a cohesive engineering selection, not an export of the full working directory.

## Where to start

1. [Image preparation](../accessible_assistant/vision_preprocess.py): quality, enhancement, deskewing and OCR region proposals.
2. [Grounding](../accessible_assistant/grounding.py): parse untrusted output, validate boxes and normalize supported labels.
3. [Audio preparation](../accessible_assistant/audio_preprocess.py): NumPy signal processing and MFCC analysis.
4. [Segmentation](../accessible_assistant/segmentation.py) and [overlays](../accessible_assistant/annotate.py): masks, text components and annotations.
5. [Tracking](../accessible_assistant/tracking.py) and [Persian descriptions](../accessible_assistant/scene_text.py): association, frame-relative trends and traceable sentences.
6. [Sequence model](../accessible_assistant/sequence_model.py): optional compact GRU. Its [training helper](../scripts/train_approach_gru.py) uses synthetic trajectories; it does not establish real-world motion accuracy.
7. [Contracts](../accessible_assistant/contracts.py) and [normalization](../accessible_assistant/persian_text.py): component results and Persian text.

## Reproduce the checks

Use Python 3.11 and install requirements-dev.txt. Run python -m pytest -q tests from the repository root. The published selection passed 21 tests and 6 subtests locally on Windows. The included workflow checks Windows and Linux; see its actual run status for remote results.

The small example uses synthetic data and requires no camera, network or model files. It demonstrates processing utilities, not model recognition.

## Release boundary

The Streamlit application, phone API, model-serving orchestration, private runtime records, source recordings, evaluation images, model weights and complete training/evaluation workspace are not distributed here. Screenshots and architecture describe the larger application; available source files are the selected modules above.

Public tests cover those modules only. Integration tests that require the complete pipeline are not included. Publication edits to the selected implementation are limited to packaging and module descriptions.
