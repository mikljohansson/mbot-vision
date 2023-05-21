#include "mbot-vision/http/httpd.h"

#include <Arduino.h>
#include <esp_http_server.h>
#include "mbot-vision/image/camera.h"
#include "mbot-vision/framerate.h"
#include "mbot-vision/detection/detector.h"
#include "mbot-vision/datalog.h"
#include "mbot-vision/wiring.h"
#include "mbot-vision/common.h"
#include "mbot-vision/http/index.h"
#include "mbot-vision/http/jpeg-stream.h"
#include "mbot-vision/http/event-stream.h"

static Detector *detector;
static DataLogger *logger;

esp_err_t handleIndex(httpd_req_t *req) {
    Serial.println("Sending index.html");
    httpd_resp_set_hdr(req, "Connection", "close");
    return httpd_resp_sendstr(req, indexDocument.c_str());
}

httpd_uri_t uri_index = {
    .uri      = "/",
    .method   = HTTP_GET,
    .handler  = handleIndex,
    .user_ctx = NULL
};



esp_err_t handleFlash(httpd_req_t *req) {
    httpd_resp_set_hdr(req, "Connection", "close");
    
    size_t buf_len = httpd_req_get_url_query_len(req) + 1;
    char variable[32] = {0,};
    
    if (buf_len > 1) {
        char *buf = (char*)malloc(buf_len);
        if (!buf){
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
        
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK && 
            httpd_query_key_value(buf, "v", variable, sizeof(variable)) == ESP_OK) {
            int duty = atoi(variable);
            duty = std::max(0, std::min(duty, MV_FLASH_MAX));
            
            if (duty < MV_FLASH_MAX / 15) {
                duty = 0;
            }
            
            if (!BOARD_OVERLOADED_SDCARD_PINS || !logger->isEnabled()) {
                Serial.println(String("Turning flash to value ") + duty);
                ledcWrite(MV_FLASH_CHAN, duty);
            }

            free(buf);
            return httpd_resp_sendstr(req, "OK");
        }

        free(buf);
    }

    httpd_resp_send_404(req);
    return ESP_FAIL;
}

httpd_uri_t uri_flash = {
    .uri      = "/flash",
    .method   = HTTP_POST,
    .handler  = handleFlash,
    .user_ctx = NULL
};



esp_err_t handleToggleDetectorStream(httpd_req_t *req) {
    Serial.println("Toggling model output stream");
    JpegStream::toggleDetectorStream();
    httpd_resp_set_hdr(req, "Connection", "close");
    return httpd_resp_sendstr(req, "OK");
}

httpd_uri_t uri_toggle_detector_stream = {
    .uri      = "/toggleDetectorStream",
    .method   = HTTP_POST,
    .handler  = handleToggleDetectorStream,
    .user_ctx = NULL
};



esp_err_t handleToggleImageLogging(httpd_req_t *req) {
    Serial.println("Toggling image logging");
    logger->setActive(!logger->isActive());
    httpd_resp_set_hdr(req, "Connection", "close");
    return httpd_resp_sendstr(req, logger->isActive() ? "active" : "inactive");
}

httpd_uri_t uri_toggle_image_logging = {
    .uri      = "/toggleImageLogging",
    .method   = HTTP_POST,
    .handler  = handleToggleImageLogging,
    .user_ctx = NULL
};



void runJpegStream(void *p) {
    JpegStream *stream = (JpegStream *)p;
    stream->run();
    delete stream;
    vTaskDelete(NULL);
}

esp_err_t handleJpegStream(httpd_req_t *req) {
    Serial.println("MJPEG stream connected");
    JpegStream *stream = new JpegStream(req, *detector);

    TaskHandle_t task;
    xTaskCreatePinnedToCore(runJpegStream, "jpegStream", 4096, stream, 1, &task, 0);

    return ESP_OK;
}

httpd_uri_t uri_stream = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = handleJpegStream,
    .user_ctx = NULL
};



void runEventStream(void *p) {
    EventStream *stream = (EventStream *)p;
    stream->run();
    delete stream;
    vTaskDelete(NULL);
}

esp_err_t handleEventStream(httpd_req_t *req) {
    Serial.println("Event stream connected");
    EventStream *stream = new EventStream(req, *detector);

    TaskHandle_t task;
    xTaskCreatePinnedToCore(runEventStream, "eventStream", 4096, stream, 1, &task, 0);
    
    return ESP_OK;
}

httpd_uri_t uri_events = {
    .uri      = "/events",
    .method   = HTTP_GET,
    .handler  = handleEventStream,
    .user_ctx = NULL
};



void httpdRun(Detector &d, DataLogger &l) {
    detector = &d;
    logger = &l;

    // https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/protocols/esp_http_server.html
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.core_id = 0;
    config.lru_purge_enable = true;

    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &uri_index);
        httpd_register_uri_handler(server, &uri_flash);
        httpd_register_uri_handler(server, &uri_toggle_detector_stream);
        httpd_register_uri_handler(server, &uri_toggle_image_logging);
        httpd_register_uri_handler(server, &uri_stream);
        httpd_register_uri_handler(server, &uri_events);
    }
}
