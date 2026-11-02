# Verification snapshot

Date: 2 November 2026.

## Published processing selection

The included tests completed locally with **21 passed and 6 subtests passed**, with no skipped tests. They cover parser validation, image/audio preparation, synthetic masks, IoU tracking, Persian descriptions and geometry behavior. They do not load the application's model backends.

~~~sh
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m examples.quick_demo
~~~

The [Windows/Linux workflow](https://github.com/Developesar/hamrah-didari/actions/workflows/components.yml) checks this public selection. Its actual run status is separate from the local result.

## Complete application context

The separate implementation workspace completed 266 software tests and 29 subtests on the same date. That broader suite is not included here. This is a reported local result for the complete application, with one dependency deprecation warning.

Neither test count establishes model accuracy, reliable navigation, noisy Persian speech quality, physical-phone performance or accessibility acceptance. Field evaluation with blind and low-vision users remains open.

The recorded Windows example shows an uploaded-image workflow, not webcam capture or a latency benchmark. Desktop control screenshots do not represent a new inference run. Phone preview readiness and quota indicators are simulated.
