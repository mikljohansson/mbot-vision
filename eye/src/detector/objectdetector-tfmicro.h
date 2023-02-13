#include <esp_nn.h>
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

static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

// https://github.com/espressif/tflite-micro-esp-examples/blob/master/examples/person_detection/main/main_functions.cc#L53
#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 39 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
//constexpr int kTensorArenaSize = 512 * 1024 + scratchBufSize;
static uint8_t *tensor_arena;

static TfLiteTensor *_input;

#ifdef MBOT_VISION_CHANNELS_LAST
static const bool channels_last = true;
#else
static const bool channels_last = false;
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

static void printModelStats() {
#if defined(COLLECT_CPU_STATS)
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif
}

static void cropAndQuantizeImage(const uint8_t *pixels, size_t image_width, size_t image_height, int8_t *target) {
    /*
    for (const uint8_t *plast = pixels + image_width * image_height; pixels < plast; pixels++, target++) {
        *target = ((int)*pixels) + 127;
    }
    */

    const size_t left = (image_width - MBOT_VISION_MODEL_INPUT_WIDTH) / 2;
    const size_t right = image_width - left - MBOT_VISION_MODEL_INPUT_WIDTH;
    const size_t top = (image_height - MBOT_VISION_MODEL_INPUT_HEIGHT) / 2;

    //serialPrint("frame %dx%d, left %d right %d top %d\n", image_width, image_height, left, right, top);

    const uint8_t* source = pixels + (top * image_width + left) * 3;
    const size_t pixel_count = MBOT_VISION_MODEL_INPUT_WIDTH * MBOT_VISION_MODEL_INPUT_HEIGHT;

    if (channels_last) {
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
    else {
        // Input is stored in HWC format, target needs to be CHW format
        for (size_t y = 0; y < MBOT_VISION_MODEL_INPUT_HEIGHT; y++) {
            for (size_t x = 0; x < MBOT_VISION_MODEL_INPUT_WIDTH; x++) {
                target[0] = ((int)source[0]) - 127;
                target[pixel_count] = ((int)source[1]) - 127;
                target[pixel_count * 2] = ((int)source[2]) - 127;
                
                target++;
                source += 3;
            }

            source += (right + left) * 3;
        }
    }
}

ObjectDetector::ObjectDetector()
 : _framerate("Object detector framerate: %02f\n"), _detected({0, 0, false}), _lastoutputbuf(0) {
    // Allocate the tensor memory block on internal memory which is much smaller but a bit faster than external SPI RAM
    // https://github.com/espressif/tflite-micro-esp-examples/blob/master/examples/person_detection/main/main_functions.cc#L70
    tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // Allocates on SPI RAM (slower but bigger memory)
    //tensor_arena = new uint8_t[kTensorArenaSize];

    if (!tensor_arena) {
        serialPrint("Failed to allocate %d bytez for tfmicro tensor arena\n", kTensorArenaSize);
    }

    _signal = xSemaphoreCreateBinary();
}

ObjectDetector::~ObjectDetector() {
    vSemaphoreDelete(_signal);
}

void ObjectDetector::begin() {
    Serial.print("Initializing tflite");
    Serial.printf(" on core %d, clock %d MHz", xPortGetCoreID(), getCpuFrequencyMhz());

    if (channels_last) {
        Serial.println(" with a channels-last model (best inference speed)");
    }
    else {
        Serial.println(" with a channels-first model (slower inference speed)");
    }

    try {
        // Map the model into a usable data structure. This doesn't involve any
        // copying or parsing, it's a very lightweight operation.
        model = tflite::GetModel(mbot_vision_model);
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            MicroPrintf(
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
        static tflite::MicroMutableOpResolver<27> micro_op_resolver;
        micro_op_resolver.AddAdd();
        //micro_op_resolver.AddBatchMatMul();
        micro_op_resolver.AddConcatenation();
        micro_op_resolver.AddConv2D();
        micro_op_resolver.AddDepthwiseConv2D();
        micro_op_resolver.AddDequantize();
        micro_op_resolver.AddDiv();
        micro_op_resolver.AddLog();
        micro_op_resolver.AddLogistic();
        micro_op_resolver.AddMaxPool2D();
        micro_op_resolver.AddMean();
        micro_op_resolver.AddMinimum();
        micro_op_resolver.AddMul();
        micro_op_resolver.AddPad();
        micro_op_resolver.AddQuantize();
        micro_op_resolver.AddRelu();
        micro_op_resolver.AddReduceMax();
        micro_op_resolver.AddReshape();
        micro_op_resolver.AddResizeNearestNeighbor();
        micro_op_resolver.AddRsqrt();
        micro_op_resolver.AddSoftmax();
        micro_op_resolver.AddSplit();
        micro_op_resolver.AddSquaredDifference();
        micro_op_resolver.AddStridedSlice();
        micro_op_resolver.AddSub();
        micro_op_resolver.AddSum();
        micro_op_resolver.AddTranspose();
        micro_op_resolver.AddTransposeConv();

        // Build an interpreter to run the model with.
        // NOLINTNEXTLINE(runtime-global-variables)
        static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
        interpreter = &static_interpreter;

        // Allocate memory from the tensor_arena for the model's tensors.
        TfLiteStatus allocate_status = interpreter->AllocateTensors();
        if (allocate_status != kTfLiteOk) {
            MicroPrintf( "AllocateTensors() failed");
            return;
        }

        // Get information about the memory area to use for the model's input.
        _input = interpreter->input(0);

        if (channels_last) {
            if ((_input->dims->size != 4) || 
                (_input->dims->data[0] != 1) ||
                (_input->dims->data[1] != MBOT_VISION_MODEL_INPUT_HEIGHT) ||
                (_input->dims->data[2] != MBOT_VISION_MODEL_INPUT_WIDTH) ||
                (_input->dims->data[3] != 3) || 
                (_input->type != kTfLiteInt8)) {
                MicroPrintf("The models input tensor shape and type doesn't match what's expected by objectdetector-tflite.h");
                return;
            }
        }
        else {
            if ((_input->dims->size != 4) || 
                (_input->dims->data[0] != 1) ||
                (_input->dims->data[1] != 3) ||
                (_input->dims->data[2] != MBOT_VISION_MODEL_INPUT_HEIGHT) ||
                (_input->dims->data[3] != MBOT_VISION_MODEL_INPUT_WIDTH) || 
                (_input->type != kTfLiteInt8)) {
                MicroPrintf("The models input tensor shape and type doesn't match what's expected by objectdetector-tflite.h");
                return;
            }
        }
    }
    catch (std::exception &e) {
        serialPrint("Caught tflite exception when initializing model: %s\n", e.what());
        return;
    }
    catch (...) {
        serialPrint("Caught unknown tflite exception when initializing model\n");
        return;
    }

    xTaskCreatePinnedToCore(runStatic, "objectDetector", 10000, this, 2, &_task, 1);
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
    FrameBufferItem fb;

    while (true) {
        fb = fbqueue->take(fb);

        BenchmarkTimer frame, decompress;
        bool result = _decoder.decompress(fb.frame->buf, fb.frame->len);
        decompress.stop();

        fbqueue->release(fb);

        // Failed to decompress, probably some bit error so wait a bit and retry
        if (!result) {
            backoff();
            continue;
        }

        uint8_t *pixels = _decoder.getOutputFrame();
        size_t width = _decoder.getOutputWidth(), height = _decoder.getOutputHeight();
        BenchmarkTimer inference;

        // Copy frame to input tensor
        cropAndQuantizeImage(pixels, width, height, _input->data.int8);
        TfLiteTensor* output;

        try {
            // Run the model on this input and make sure it succeeds.
            if (kTfLiteOk != interpreter->Invoke()) {
                MicroPrintf( "Invoke failed.");
            }
            
            output = interpreter->output(0);
            inference.stop();
        }
        catch (std::exception &e) {
            serialPrint("Caught tflite exception when executing model: %s\n", e.what());
            return;
        }
        catch (...) {
            serialPrint("Caught unknown tflite exception when executing model\n");
            return;
        }

        // Process the inference results
        int maxx = 0, maxy = 0;
        int maxv = -256;

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

        if (_lastoutputbuf) {
            memcpy(_lastoutputbuf, output->data.int8, MBOT_VISION_MODEL_OUTPUT_HEIGHT * MBOT_VISION_MODEL_OUTPUT_WIDTH);
        }

        float probability = ((float)maxv + 127) / 255;
        if (probability >= 0.5) {
            _detected = {
                ((float)maxx + 0.5f) / MBOT_VISION_MODEL_OUTPUT_WIDTH, 
                ((float)maxy + 0.5f) / MBOT_VISION_MODEL_OUTPUT_HEIGHT,
                true};
            serialPrint("Object detected at coordinate %.02f x %.02f with probability %.02f (decompress %dms, inference %dms, total %dms)\n", 
                _detected.x, _detected.y, probability, decompress.took(), inference.took(), frame.took());
            printModelStats();
        }
        else {
            _detected = {0, 0, false};
        }

        xSemaphoreGive(_signal);
        _framerate.tick();
    
        // Let other tasks run too
        delay(1);
    }
}

static int8_t *buffer = 0;

void ObjectDetector::draw(uint8_t *pixels, size_t width, size_t height) {
    if (!_lastoutputbuf) {
        _lastoutputbuf = new int8_t[MBOT_VISION_MODEL_OUTPUT_HEIGHT * MBOT_VISION_MODEL_OUTPUT_WIDTH];
        memset(_lastoutputbuf, 0, MBOT_VISION_MODEL_OUTPUT_HEIGHT * MBOT_VISION_MODEL_OUTPUT_WIDTH);
    }

    size_t offset = (height - MBOT_VISION_MODEL_OUTPUT_HEIGHT) * width;
    for (int y = 0; y < MBOT_VISION_MODEL_OUTPUT_HEIGHT; y++) {
        for (int x = 0; x < MBOT_VISION_MODEL_OUTPUT_WIDTH; x++) {
            int val = _lastoutputbuf[y * MBOT_VISION_MODEL_OUTPUT_WIDTH + x];
            size_t pos = (offset + y * width + x) * 3;
            pixels[pos] = pixels[pos + 1] = pixels[pos + 2] = std::min(std::max(0, val + 127), 255);
        }
    }
}

void ObjectDetector::runStatic(void *p) {
    ObjectDetector *detector = (ObjectDetector *)p;
    detector->run();
    vTaskDelete(NULL);
}
