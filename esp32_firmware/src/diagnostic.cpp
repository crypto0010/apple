// One-shot board probe: prints chip model, flash size, PSRAM size, and free
// heap over the native USB CDC port. Used to confirm the board can support
// the planned INT8 model budget before building the inference firmware.

#include <Arduino.h>
#include <esp_heap_caps.h>

void setup() {
  Serial.begin(115200);
  delay(800);  // let the host enumerate the USB CDC device

  Serial.println();
  Serial.println(F("=== ESP32-S3-CAM diagnostic ==="));
  Serial.printf("chip_model=%s\n", ESP.getChipModel());
  Serial.printf("chip_revision=%d\n", ESP.getChipRevision());
  Serial.printf("chip_cores=%d\n", ESP.getChipCores());
  Serial.printf("cpu_freq_mhz=%d\n", ESP.getCpuFreqMHz());
  Serial.printf("flash_size_bytes=%u\n", ESP.getFlashChipSize());
  Serial.printf("flash_speed_hz=%u\n", ESP.getFlashChipSpeed());
  Serial.printf("psram_found=%d\n", psramFound() ? 1 : 0);
  Serial.printf("psram_size_bytes=%u\n", ESP.getPsramSize());
  Serial.printf("psram_free_bytes=%u\n", ESP.getFreePsram());
  Serial.printf("heap_internal_total=%u\n",
                static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_INTERNAL)));
  Serial.printf("heap_internal_free=%u\n",
                static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
  Serial.printf("heap_spiram_total=%u\n",
                static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)));
  Serial.printf("heap_spiram_free=%u\n",
                static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  Serial.printf("sketch_size_bytes=%u\n", ESP.getSketchSize());
  Serial.printf("sketch_free_bytes=%u\n", ESP.getFreeSketchSpace());
  Serial.println(F("=== ready ==="));
}

void loop() {
  // Heartbeat so the host driver can confirm the device is alive and the
  // serial pipe is not blocked. Printed every 2 s to keep the log readable.
  static uint32_t tick = 0;
  Serial.printf("heartbeat=%lu free_heap=%u free_psram=%u\n",
                static_cast<unsigned long>(tick++),
                ESP.getFreeHeap(),
                ESP.getFreePsram());
  delay(2000);
}
