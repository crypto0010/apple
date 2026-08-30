# apple-bench

Hardware-grounded latency-energy-accuracy benchmark of apple leaf disease
classifiers on Jetson Orin Nano and ESP32-S3-CAM.

## Setup

    uv sync --extra dev

## Reproduce trained models (Phase A.1)

    bash scripts/01_download_datasets.sh
    python scripts/02_train_all.py --output runs/

## Repository scope

This repository contains the benchmark source only: the `apple_bench`
package, its test suite, the driver scripts, and the ESP32-S3 firmware.
Datasets, trained weights, exported engines and run artefacts are not
tracked (see `.gitignore`); they are reproduced with the scripts above.
`esp32_firmware/src/model_data.h` is generated from a `.tflite` model by
`scripts/embed_tflite_model.py` and is likewise not tracked.

## License

Released under [CC BY 4.0](LICENSE).
