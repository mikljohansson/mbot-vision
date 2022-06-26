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
TfLiteTensor* input = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t *tensor_arena;

TfLiteStatus CropAndQuantizeImage(tflite::ErrorReporter* error_reporter, const uint8_t *image_buffer,
                                  size_t image_width, size_t image_height,
                                  const TfLiteTensor* tensor) {
    TF_LITE_REPORT_ERROR(error_reporter, "Cropping image and quantizing");

    const size_t left = (image_width - kNumCols) / 2;
    const size_t right = image_width - left - kNumCols;
    const size_t top = (image_height - kNumRows) / 2;
    int8_t* image_data = tensor->data.int8;

    int8_t* target = image_data;
    const uint8_t* source = image_buffer + (top * image_width + left) * 3;

    for (size_t y = 0; y < kNumRows; y++) {
        for (size_t x = 0; x <= kNumCols; x++) {
            /**
             * Gamma corected rgb to greyscale formula: Y = 0.299R + 0.587G + 0.114B
             * for effiency we use some tricks on this + quantize to [-128, 127]
             */
            int8_t grey_pixel = ((305 * source[0] + 600 * source[1] + 119 * source[2]) >> 10) - 128;

            *target = grey_pixel;
            target++;
            source += 3;
        }

        source += (left + right) * 3;
    }

    TF_LITE_REPORT_ERROR(error_reporter, "Image cropped and quantized");
    return kTfLiteOk;
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
    input = interpreter->input(0);

    if ((input->dims->size != 4) || (input->dims->data[0] != 1) ||
        (input->dims->data[1] != kNumRows) ||
        (input->dims->data[2] != kNumCols) ||
        (input->dims->data[3] != kNumChannels) || (input->type != kTfLiteInt8)) {
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
    JpegDecoder decoder;
    _framerate.init();

    Serial.print("Starting face detector");

    while (true) {
        camera_fb_t *fb = fbqueue->take();

        if (fb) {
            bool result = decoder.decompress(fb->buf, fb->len);
            fbqueue->release(fb);

            if (!result) {
                continue;
            }

            uint8_t *pixels = decoder.getOutputFrame();
            size_t width = decoder.getOutputWidth(), height = decoder.getOutputHeight();

            // Copy frame to input tensor
            CropAndQuantizeImage(error_reporter, pixels, width, height, input);

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

            serialPrint("Person detection scores: %f yes, %f no\n", person_score, no_person_score);
            delay(1000);

            _framerate.tick();
        }
    }
}

void ObjectDetector::runStatic(void *p) {
    ObjectDetector *detector = (ObjectDetector *)p;
    detector->run();
    vTaskDelete(NULL);
}
