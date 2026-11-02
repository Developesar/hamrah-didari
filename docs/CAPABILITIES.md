# Capability map

| Area | Implemented work | Evaluation boundary |
| --- | --- | --- |
| Scene understanding | Vision-language adapters and concise Persian response path | Real-photo errors and hallucinations remain possible. |
| Object assistance | Detection adapters, object listing and frame-relative positions | Supported categories depend on the configured detector. |
| General text | OCR adapters and Persian/English text handling | Uncertain numbers are not treated as confirmed. |
| Spoken input | Audio validation, transcription and transcript review | Conversational and noisy Persian speech need separate quality assessment. |
| Spoken output | Configurable speech synthesis and Persian text preparation | A generated WAV is not proof of naturalness or accessibility acceptance. |
| Laptop camera | Direct capture in the Windows Streamlit app, browser-camera input and an experimental live object-box panel | Current desktop screenshots show controls, not a new capture or inference result. |
| User experience | Streamlit, phone-oriented PWA and specialized modes | Device and screen-reader field checks remain open. |
| Application API | Input validation, access control, quotas, request state and feedback path | Automated tests use isolated state and controlled model substitutes. |

## Capstone contribution

The project brings multiple AI tasks into one Persian assistive workflow, with explicit uncertainty and application-level safeguards. Existing model families supply inference capabilities; the capstone contribution is their integration, interaction design, Persian processing and engineering around them.

## Public media

The phone-interface screenshots use the development preview server, whose quota and response fixtures are simulated. Desktop screenshots show the actual local Streamlit interface. The separately labelled recorded example contains two frames from the owner-provided Windows recording: an uploaded image and a later Persian answer with its audio player. It does not demonstrate webcam capture or establish independent model accuracy. The original video, audio recordings, runtime data and model artifacts are not included.
