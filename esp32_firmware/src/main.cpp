// Inference firmware (Plan A.2 Task 17).
//
// Loads the embedded INT8 .tflite model (model_data.h, produced by
// scripts/embed_tflite_model.py) into a PSRAM-backed tensor arena, then
// loops calling tflite::MicroInterpreter::Invoke() on a deterministic
// input pattern.  Per-inference latency, free heap, and free PSRAM are
// logged over USB CDC at 115200 baud in the format the host-side
// scripts/07_bench_esp32.py driver parses:
//
//     INF<n>: <latency_us>us heap=<bytes> psram=<bytes>
//
// The input is a fixed pattern (not camera frames) because we want
// reproducible latency measurements; the OV2640 camera path adds
// per-frame jitter that would confound the benchmark.

#include <Arduino.h>
#include <esp_heap_caps.h>

// MicroTensorFlowLite.h is intentionally empty - including it just forces
// the Arduino toolchain to compile the library's .cpp files.  All real
// TFLM symbols come from the tensorflow/lite/micro/* headers below.
#include <MicroTensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

// EfficientNet-Lite0 INT8 needs ~2.4 MB of arena for its largest
// intermediate activation; budget 3 MB.  Allocated from PSRAM so this
// doesn't touch the 320 KB internal SRAM (which TFLM and Arduino both
// share for stack and small allocations).
constexpr size_t kTensorArenaSize = 3 * 1024 * 1024;
uint8_t* tensor_arena = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

static void halt(const char* msg) {
  Serial.println(msg);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  delay(800);

  if (!psramFound()) halt("FATAL: PSRAM not detected (check qio_opi config)");

  tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));
  if (!tensor_arena) halt("FATAL: tensor_arena PSRAM alloc failed");

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("FATAL: schema mismatch model=%lu lib=%d\n",
                  static_cast<unsigned long>(model->version()),
                  TFLITE_SCHEMA_VERSION);
    while (true) delay(1000);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interp(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interp;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    halt("FATAL: AllocateTensors failed (kTensorArenaSize too small?)");
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Deterministic test pattern - reproducible latencies across runs.
  for (size_t i = 0; i < input->bytes; ++i) {
    input->data.int8[i] = static_cast<int8_t>((i * 7 + 3) & 0xff);
  }

  Serial.printf(
      "model_kb=%u arena_used_kb=%u input_bytes=%u output_bytes=%u\n",
      static_cast<unsigned>(g_model_data_len / 1024),
      static_cast<unsigned>(interpreter->arena_used_bytes() / 1024),
      static_cast<unsigned>(input->bytes),
      static_cast<unsigned>(output->bytes));
  Serial.println("=== inference loop ===");
}

void loop() {
  static uint32_t i = 0;
  uint32_t t0 = micros();
  TfLiteStatus s = interpreter->Invoke();
  uint32_t dt = micros() - t0;
  if (s != kTfLiteOk) {
    Serial.printf("INF%lu: ERROR\n", static_cast<unsigned long>(i++));
  } else {
    Serial.printf("INF%lu: %luus heap=%u psram=%u\n",
                  static_cast<unsigned long>(i++),
                  static_cast<unsigned long>(dt),
                  static_cast<unsigned>(ESP.getFreeHeap()),
                  static_cast<unsigned>(ESP.getFreePsram()));
  }
}
