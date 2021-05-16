#include <Arduino.h>
#include <esp_spi_flash.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "common.h"

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::AllOpsResolver resolver;

#define TENSOR_ARENA_SIZE   3 * 1024 * 1024
uint8_t *tensor_arena;

const void *model_buf;
spi_flash_mmap_handle_t model_handle;
const tflite::Model *model;
tflite::MicroInterpreter *interpreter;

void printMemoryUsage() {
    serialPrint("  Total heap: %d\n", ESP.getHeapSize());
    serialPrint("  Free heap: %d\n", ESP.getFreeHeap());
    serialPrint("  Total PSRAM: %d\n", ESP.getPsramSize());
    serialPrint("  Free PSRAM: %d\n", ESP.getFreePsram());
}

void printTensor(TfLiteTensor *tensor) {
    Serial.print(tensor->name);
    Serial.print(" (");
    for (size_t j = 0; j < tensor->dims->size; j++) {
        if (j > 0) {
            Serial.print(", ");
        }

        Serial.print(tensor->dims->data[j]);
    }

    Serial.println(")");
}

void modelInit() {
    printMemoryUsage();

    esp_err_t err = spi_flash_mmap(0x210000, 0x1F0000, SPI_FLASH_MMAP_DATA, &model_buf, &model_handle);
    if (err != ESP_OK) {
        Serial.println("Failed to mmap ML model from SPI flash");
        while (true);
    }

    const tflite::Model *model = ::tflite::GetModel(model_buf);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
            "Model provided is schema version %d not equal "
            "to supported version %d.\n",
            model->version(), TFLITE_SCHEMA_VERSION);
    }
    
    Serial.println("Loaded ML model from SPI flash");
    printMemoryUsage();

    tensor_arena = (uint8_t *)malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        TF_LITE_REPORT_ERROR(error_reporter,
            "Failed to allocate PSRAM for model arena");
    }

    interpreter = new ::tflite::MicroInterpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter->AllocateTensors();
    
    for (size_t i = 0; i < interpreter->inputs_size(); i++) {
        Serial.print("  input: ");
        printTensor(interpreter->input_tensor(i));
    }

    for (size_t i = 0; i < interpreter->outputs_size(); i++) {
        Serial.print("  output: ");
        printTensor(interpreter->output_tensor(i));
    }

    Serial.println("Initialized TFLite interpreter");
    printMemoryUsage();
}
