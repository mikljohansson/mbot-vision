extern "C" {
    #include <tinymaix.h>
}

#include <mbot-vision-model.h>
#include "detection/objectdetector.h"
#include "image/camera.h"
#include "image/jpeg.h"
#include "common.h"

static tm_mdl_t mdl;
static tm_mat_t *_in;
static int8_t *_input;
static int8_t *_output;

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

static uint32_t _t0,_t1;
static tm_err_t layer_cb(tm_mdl_t* mdl, tml_head_t* lh)
{   //dump middle result
    int h = lh->out_dims[1];
    int w = lh->out_dims[2];
    int ch= lh->out_dims[3];
    mtype_t* output = TML_GET_OUTPUT(mdl, lh);
    //TM_PRINTF("Layer %d callback ========\n", mdl->layer_i);
    #if 0
    for(int y=0; y<h; y++){
        TM_PRINTF("[");
        for(int x=0; x<w; x++){
            TM_PRINTF("[");
            for(int c=0; c<ch; c++){
            #if TM_MDL_TYPE == TM_MDL_FP32
                TM_PRINTF("%.3f,", output[(y*w+x)*ch+c]);
            #else
                TM_PRINTF("%.4f,", TML_DEQUANT(lh,output[(y*w+x)*ch+c]));
            #endif
            }
            TM_PRINTF("],");
        }
        TM_PRINTF("],\n");
    }
    TM_PRINTF("\n");
    #endif
    _t1 = TM_GET_US();
    //TM_PRINTF("===L%d use %.3f ms\n", mdl->layer_i, (float)(_t1-_t0)/1000.0);
    _t0 = TM_GET_US();

    return TM_OK;
}

void ObjectDetector::begin() {
    _input = new int8_t[MBOT_VISION_MODEL_INPUT_HEIGHT * MBOT_VISION_MODEL_INPUT_WIDTH * 3];
    _output = new int8_t[MBOT_VISION_MODEL_OUTPUT_HEIGHT * MBOT_VISION_MODEL_OUTPUT_WIDTH * 1];

    _in = (tm_mat_t *)malloc(sizeof(tm_mat_t));
    *_in = {3, MBOT_VISION_MODEL_INPUT_HEIGHT, MBOT_VISION_MODEL_INPUT_WIDTH, 3, {(mtype_t*)_input}};

    TM_DBGT_INIT();

    tm_stat((tm_mdlbin_t*)mbot_vision_model); 
    tm_err_t res = tm_load(&mdl, mbot_vision_model, NULL, layer_cb, _in);

    if (res != TM_OK) {
        serialPrint("tinymaix tm_load() failed with error code %d", res);
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

    Serial.println("Starting object detector");
    FrameBufferItem fb;

    while (true) {
        fb = fbqueue->take(fb);

        BenchmarkTimer frame, decompress;
        bool result = _decoder.decompress(fb.fb->buf, fb.fb->len);
        decompress.stop();

        fbqueue->release(fb);

        // Failed to decompress, probabily some bit error so wait a bit and retry
        if (!result) {
            backoff();
            continue;
        }

        uint8_t *pixels = _decoder.getOutputFrame();
        size_t width = _decoder.getOutputWidth(), height = _decoder.getOutputHeight();
        BenchmarkTimer inference;

        // Copy frame to input tensor
        cropAndQuantizeImage(pixels, width, height, _input);

        // Run the model on this input and make sure it succeeds.
        tm_mat_t outs = {3, MBOT_VISION_MODEL_OUTPUT_HEIGHT, MBOT_VISION_MODEL_OUTPUT_WIDTH, 1, {(mtype_t*)_output}};
        tm_err_t res = tm_run(&mdl, _in, &outs);

        if(res != TM_OK) {
            serialPrint("tinymaix tm_run() failed with error code %d", res);
        }
        
        inference.stop();

        // Process the inference results
        int maxx = 0, maxy = 0;
        int maxv = -256;

        for (int y = 0; y < MBOT_VISION_MODEL_OUTPUT_HEIGHT; y++) {
            for (int x = 0; x < MBOT_VISION_MODEL_OUTPUT_WIDTH; x++) {
                int val = _output[y * MBOT_VISION_MODEL_OUTPUT_WIDTH + x];

                if (val > maxv) {
                    maxv = val;
                    maxx = x;
                    maxy = y;
                }
            }
        }

        if (_lastoutputbuf) {
            memcpy(_lastoutputbuf, _output, MBOT_VISION_MODEL_OUTPUT_WIDTH * MBOT_VISION_MODEL_OUTPUT_HEIGHT);
        }

        float probability = ((float)maxv + 127) / 255;
        if (probability >= 0.5) {
            _detected = {(float)maxx / MBOT_VISION_MODEL_OUTPUT_WIDTH, (float)maxy / MBOT_VISION_MODEL_OUTPUT_HEIGHT, true};
            serialPrint("Object detected at coordinate %.02f x %.02f with probability %.02f (decompress %dms, inference %dms, total %dms)\n", 
                _detected.x, _detected.y, probability, decompress.took(), inference.took(), frame.took());
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
