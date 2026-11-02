# Capability map

| Area | Implemented work | Evaluation boundary |
| --- | --- | --- |
| Scene understanding | Vision-language adapters and concise Persian response path | Real-photo errors and hallucinations remain possible. |
| Object assistance | Detection adapters, object listing and frame-relative positions | Supported categories depend on the configured detector. |
| General text | OCR adapters and Persian/English text handling | Uncertain numbers are not treated as confirmed. |
| Spoken input | Audio validation, transcription and transcript review | Conversational and noisy Persian speech need separate quality assessment. |
| Spoken output | Configurable speech synthesis and Persian text preparation | A generated WAV is not proof of naturalness or accessibility acceptance. |
| User experience | Streamlit, phone-oriented PWA and specialized modes | Device and screen-reader field checks remain open. |
| Application API | Input validation, access control, quotas, request state and feedback path | Automated tests use isolated state and controlled model substitutes. |

## Capstone contribution

The project brings multiple AI tasks into one Persian assistive workflow, with explicit uncertainty and application-level safeguards. Existing model families supply inference capabilities; the capstone contribution is their integration, interaction design, Persian processing and engineering around them.

## Public media

The interface images in this repository use the development preview server. Its response and quota fixtures are simulated. The screenshots demonstrate interface structure only; there are no participant images, voice recordings or private model artifacts in this showcase.
