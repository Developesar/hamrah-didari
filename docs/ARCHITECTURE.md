# Architecture

Hamrah Didari separates interaction, orchestration and model execution so that a model response is not confused with application state or a confirmed fact.

## Interaction surfaces

The Streamlit application provides the local desktop interface. A Persian, phone-oriented PWA exposes camera/gallery input, voice controls, a large primary action and specialized modes. The web API validates requests and returns a shared structured response.

## Processing path

1. Validate the image or audio and track the current interaction.
2. Transcribe a spoken question when needed and allow review of the text.
3. Route the request to scene description, OCR, object listing, finding or question answering.
4. Collect component results with status, elapsed time, data and user-facing messages.
5. Prepare a Persian response, retain uncertainty and optionally synthesize speech.

Model adapters include vision-language inference, detection, OCR, ASR and TTS. Availability depends on the selected backend and local assets. An interface control alone does not prove that every requested object category works with every backend.

## Operational boundaries

The API includes token-based access, daily quotas, input bounds and controlled inference concurrency. Operational user records and runtime state remain outside this public showcase. The application is designed to run model inference on owner-controlled infrastructure; a new hosted inference service is not part of this release.

Direction descriptions are relative to the image frame. The system does not infer a safe route from a single image. Numbers extracted by OCR require appropriate confirmation rather than being spoken as established facts.

## Current limits

Physical-phone camera/microphone behavior, screen-reader acceptance and field quality require separate evaluation. The future local-interface/remote-GPU transport is not presented as completed. This repository contains no executable application distribution.
