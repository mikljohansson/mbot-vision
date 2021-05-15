#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
tflite::AllOpsResolver resolver;

#define TENSOR_ARENA_SIZE   2 * 1024 * 1024
uint8_t *tensor_arena;

const tflite::Model *model;
tflite::MicroInterpreter *interpreter;

void modelInit() {
    /*const tflite::Model *model = ::tflite::GetModel(mv_mobilenet_model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
            "Model provided is schema version %d not equal "
            "to supported version %d.\n",
            model->version(), TFLITE_SCHEMA_VERSION);
    }

    tensor_arena = (uint8_t *)malloc(TENSOR_ARENA_SIZE);
    if (!tensor_arena) {
        TF_LITE_REPORT_ERROR(error_reporter,
            "Failed to allocate PSRAM for model arena");
    }

    interpreter = new ::tflite::MicroInterpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter->AllocateTensors();
    */
}
