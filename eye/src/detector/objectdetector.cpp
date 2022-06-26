#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_utils.h"

#include "detection/person_detect_model_data.h"
#include "detection/person_detect_model_settings.h"
#include "detection/objectdetector.h"
#include "image/camera.h"
#include "image/jpeg.h"
#include "common.h"

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t *tensor_arena;

static void cropAndQuantizeImage(const uint8_t *pixels, size_t image_width, size_t image_height, int8_t *target) {
    const size_t left = (image_width - kNumCols) / 2;
    const size_t right = image_width - left - kNumCols;
    const size_t top = (image_height - kNumRows) / 2;

    const uint8_t* source = pixels + (top * image_width + left) * 3;

    for (size_t y = 0; y < kNumRows; y++) {
        for (size_t x = 0; x < kNumCols; x++) {
            /**
             * Gamma corected rgb to greyscale formula: Y = 0.299R + 0.587G + 0.114B
             * for effiency we use some tricks on this + quantize to [-128, 127]
             */
            *target = ((305 * (int)source[0] + 600 * (int)source[1] + 119 * (int)source[2]) >> 10) - 128;

            target++;
            source += 3;
        }

        source += (right + left) * 3;
    }
}

ObjectDetector::ObjectDetector()
 : _framerate("Object detector framerate: %02f\n"), _detected({0, 0, false}) {
    _signal = xSemaphoreCreateBinary();
}

ObjectDetector::~ObjectDetector() {
    vSemaphoreDelete(_signal);
}

void ObjectDetector::start() {
    tensor_arena = new uint8_t[kTensorArenaSize];

    tflite::InitializeTarget();

    // Set up logging. Google style is to avoid globals or statics because of
    // lifetime uncertainty, but since this has a trivial destructor it's okay.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(g_person_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                            "Model provided is schema version %d not equal "
                            "to supported version %d.",
                            model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // An easier approach is to just use the AllOpsResolver, but this will
    // incur some penalty in code space for op implementations that are not
    // needed by this graph.
    //
    // tflite::AllOpsResolver resolver;
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();

    // Build an interpreter to run the model with.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    // Get information about the memory area to use for the model's input.
    _input = interpreter->input(0);

    if ((_input->dims->size != 4) || 
        (_input->dims->data[0] != 1) ||
        (_input->dims->data[1] != kNumRows) ||
        (_input->dims->data[2] != kNumCols) ||
        (_input->dims->data[3] != kNumChannels) || 
        (_input->type != kTfLiteInt8)) {
        TF_LITE_REPORT_ERROR(error_reporter,
                            "Bad input tensor parameters in model");
        return;
    }

    xTaskCreatePinnedToCore(runStatic, "objectDetector", 10000, this, 1, &_task, 1);
}

DetectedObject ObjectDetector::wait() {
    xSemaphoreTake(_signal, portMAX_DELAY);
    return get();
}

DetectedObject ObjectDetector::get() {
    return _detected;
}

void ObjectDetector::run() {
    _framerate.init();

    Serial.print("Starting face detector");

    while (true) {
        camera_fb_t *fb = fbqueue->take();

        if (fb) {
            bool result = _decoder.decompress(fb->buf, fb->len);
            fbqueue->release(fb);

            if (!result) {
                continue;
            }

            uint8_t *pixels = _decoder.getOutputFrame();
            size_t width = _decoder.getOutputWidth(), height = _decoder.getOutputHeight();

            // Copy frame to input tensor
            cropAndQuantizeImage(pixels, width, height, _input->data.int8);

            // Run the model on this input and make sure it succeeds.
            if (kTfLiteOk != interpreter->Invoke()) {
                TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
            }

            TfLiteTensor* output = interpreter->output(0);

            // Process the inference results.
            // Process the inference results.
            int8_t person_score = output->data.uint8[kPersonIndex];
            int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

            float person_score_f = 
                (person_score - output->params.zero_point) * output->params.scale;
            float no_person_score_f =
                (no_person_score - output->params.zero_point) * output->params.scale;

            serialPrint("Person detection scores: %f yes, %f no\n", person_score_f, no_person_score_f);
            delay(1000);

            _framerate.tick();
        }
    }
}

static int8_t *buffer = 0;

void ObjectDetector::draw(uint8_t *pixels, size_t width, size_t height) {
    if (!buffer) {
        buffer = new int8_t[kNumCols * kNumRows * kNumChannels];
        memset(buffer, 0, kNumCols * kNumRows * kNumChannels);
    }
    
    cropAndQuantizeImage(pixels, width, height, buffer);

    const size_t left = (width - kNumCols) / 2;
    const size_t right = width - left - kNumCols;
    const size_t top = (height - kNumRows) / 2;

    uint8_t* target = pixels + (top * width + left) * 3;
    const int8_t* source = buffer;

    for (size_t y = 0; y < kNumRows; y++) {
        for (size_t x = 0; x < kNumCols; x++) {
            target[0] = target[1] = target[2] = (uint8_t)((int)*source + 128);
            target += 3;
            source++;
        }

        target += (right + left) * 3;
    }
}

void ObjectDetector::runStatic(void *p) {
    ObjectDetector *detector = (ObjectDetector *)p;
    detector->run();
    vTaskDelete(NULL);
}
