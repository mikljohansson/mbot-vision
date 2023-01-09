#include <esp_nn.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_utils.h"

#include <mbot-vision-model.h>
#include "detection/objectdetector.h"
#include "image/camera.h"
#include "image/jpeg.h"
#include "common.h"

static tflite::ErrorReporter* error_reporter = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

constexpr int kTensorArenaSize = 512 * 1024;
static uint8_t *tensor_arena;

static TfLiteTensor *_input;

static void cropAndQuantizeImage(const uint8_t *pixels, size_t image_width, size_t image_height, int8_t *target) {
    const size_t left = (image_width - MBOT_VISION_MODEL_INPUT_WIDTH) / 2;
    const size_t right = image_width - left - MBOT_VISION_MODEL_INPUT_WIDTH;
    const size_t top = (image_height - MBOT_VISION_MODEL_INPUT_HEIGHT) / 2;

    const uint8_t* source = pixels + (top * image_width + left) * 3;
    const size_t pixel_count = MBOT_VISION_MODEL_INPUT_WIDTH * MBOT_VISION_MODEL_INPUT_HEIGHT;

    // Input and target is stored in HWC format

    for (size_t y = 0; y < MBOT_VISION_MODEL_INPUT_HEIGHT; y++) {
        for (size_t x = 0; x < MBOT_VISION_MODEL_INPUT_WIDTH; x++) {
            target[0] = ((int)source[0]) - 127;
            target[1] = ((int)source[1]) - 127;
            target[2] = ((int)source[2]) - 127;
            
            target += 3;
            source += 3;
        }

        source += (right + left) * 3;
    }
}

ObjectDetector::ObjectDetector()
 : _framerate("Object detector framerate: %02f\n"), _detected({0, 0, false}), _lastoutputbuf(0) {
    _signal = xSemaphoreCreateBinary();
}

ObjectDetector::~ObjectDetector() {
    vSemaphoreDelete(_signal);
}

void ObjectDetector::begin() {
    Serial.println("Initializing tflite");
    tensor_arena = new uint8_t[kTensorArenaSize];

    tflite::InitializeTarget();

    // Set up logging. Google style is to avoid globals or statics because of
    // lifetime uncertainty, but since this has a trivial destructor it's okay.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(mbot_vision_model);
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


    // NOTE: Don't forget to change the max number of ops in the template
    static tflite::MicroMutableOpResolver<22> micro_op_resolver;
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddConcatenation();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddLog();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddMinimum();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddPad();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddResizeNearestNeighbor();
    micro_op_resolver.AddRsqrt();
    micro_op_resolver.AddSplit();
    micro_op_resolver.AddSquaredDifference();
    micro_op_resolver.AddSub();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddTransposeConv();

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
        (_input->dims->data[1] != MBOT_VISION_MODEL_INPUT_HEIGHT) ||
        (_input->dims->data[2] != MBOT_VISION_MODEL_INPUT_WIDTH) ||
        (_input->dims->data[3] != 3) || 
        (_input->type != kTfLiteInt8)) {
        TF_LITE_REPORT_ERROR(error_reporter,
                            "The models input tensor shape and type doesn't match what's expected by objectdetector.cc");
        return;
    }

    xTaskCreatePinnedToCore(runStatic, "objectDetector", 10000, this, 1, &_task, 1);
    Serial.println("Tflite successfully initialized");
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

    Serial.println("Starting object detector");

    while (true) {
        camera_fb_t *fb = fbqueue->take();

        if (fb) {
            BenchmarkTimer frame, decompress;

            bool result = _decoder.decompress(fb->buf, fb->len);
            decompress.stop();

            fbqueue->release(fb);

            if (!result) {
                continue;
            }

            uint8_t *pixels = _decoder.getOutputFrame();
            size_t width = _decoder.getOutputWidth(), height = _decoder.getOutputHeight();

            // Copy frame to input tensor
            cropAndQuantizeImage(pixels, width, height, _input->data.int8);

            // Run the model on this input and make sure it succeeds.
            BenchmarkTimer inference;
            if (kTfLiteOk != interpreter->Invoke()) {
                TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
            }
            inference.stop();

            TfLiteTensor* output = interpreter->output(0);

            // Process the inference results
            int maxx = 0, maxy = 0;
            int maxv = -1.0;

            for (int y = 0; y < MBOT_VISION_MODEL_OUTPUT_HEIGHT; y++) {
                for (int x = 0; x < MBOT_VISION_MODEL_OUTPUT_WIDTH; x++) {
                    int val = output->data.int8[y * MBOT_VISION_MODEL_OUTPUT_WIDTH + x];

                    if (val > maxv) {
                        maxv = val;
                        maxx = x;
                        maxy = y;
                    }
                }
            }

            if (!_lastoutputbuf) {
                _lastoutputbuf = new int8_t[MBOT_VISION_MODEL_OUTPUT_HEIGHT * MBOT_VISION_MODEL_OUTPUT_WIDTH];
            }
            memcpy(_lastoutputbuf, output->data.int8, MBOT_VISION_MODEL_OUTPUT_HEIGHT * MBOT_VISION_MODEL_OUTPUT_WIDTH);

            float probability = (float)maxv / 255;
            if (probability >= 0.1) {
                _detected = {(float)maxx / MBOT_VISION_MODEL_OUTPUT_WIDTH, (float)maxy / MBOT_VISION_MODEL_OUTPUT_HEIGHT, true};
                xSemaphoreGive(_signal);
                serialPrint("Object detected at coordinate %.02f x %.02f with probability %.02f (decompress %dms, inference %dms, total %dms)\n", 
                    _detected.x, _detected.y, probability, decompress.took(), inference.took(), frame.took());
            }
            else {
                _detected = {0, 0, false};
            }

            _framerate.tick();
        }
    }
}

static int8_t *buffer = 0;

void ObjectDetector::draw(uint8_t *pixels, size_t width, size_t height) {
    if (!_lastoutputbuf) {
        return;
    }

    TfLiteTensor* output = interpreter->output(0);
    size_t offset = width / 2 - MBOT_VISION_MODEL_OUTPUT_WIDTH / 2 + (height / 2 - MBOT_VISION_MODEL_OUTPUT_HEIGHT / 2) * width;

    for (int y = 0; y < MBOT_VISION_MODEL_OUTPUT_HEIGHT; y++) {
        for (int x = 0; x < MBOT_VISION_MODEL_OUTPUT_WIDTH; x++) {
            int8_t val = _lastoutputbuf[y * MBOT_VISION_MODEL_OUTPUT_WIDTH + x];
            size_t pos = (offset + y * width + x) * 3;
            pixels[pos] = pixels[pos + 1] = pixels[pos + 2] = (val + 127);
        }
    }
}

void ObjectDetector::runStatic(void *p) {
    ObjectDetector *detector = (ObjectDetector *)p;
    detector->run();
    vTaskDelete(NULL);
}