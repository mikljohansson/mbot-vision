#include <Arduino.h>
#include <esp_http_server.h>
#include "http/httpd.h"
#include "image/camera.h"
#include "framerate.h"
#include "detection/detector.h"
#include "wiring.h"
#include "common.h"

#include "http/index.h"
#include "http/jpeg-stream.h"
#include "http/event-stream.h"

static Detector *detector;

esp_err_t handleIndex(httpd_req_t *req) {
    Serial.println("Sending index.html");
    return httpd_resp_send(req, indexDocument.c_str(), HTTPD_RESP_USE_STRLEN);
}

httpd_uri_t uri_index = {
    .uri      = "/",
    .method   = HTTP_GET,
    .handler  = handleIndex,
    .user_ctx = NULL
};



esp_err_t handleFlash(httpd_req_t *req) {
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
            
            if (!LOG_TO_SDCARD) {
                Serial.println(String("Turning flash to value ") + duty);
                ledcWrite(MV_FLASH_CHAN, duty);
            }

            free(buf);
            return httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
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



esp_err_t handleJpegStream(httpd_req_t *req) {
    Serial.println("MJPEG stream connected");
    JpegStream *stream = new JpegStream(req, false, *detector);
    stream->start();
    return ESP_OK;
}

httpd_uri_t uri_stream = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = handleJpegStream,
    .user_ctx = NULL
};



esp_err_t handleDetectorStream(httpd_req_t *req) {
    Serial.println("MJPEG model output stream connected");
    JpegStream *stream = new JpegStream(req, true, *detector);
    stream->start();
    return ESP_OK;
}

httpd_uri_t uri_detector = {
    .uri      = "/detector",
    .method   = HTTP_GET,
    .handler  = handleDetectorStream,
    .user_ctx = NULL
};



esp_err_t handleEventStream(httpd_req_t *req) {
    Serial.println("Event stream connected");
    EventStream *stream = new EventStream(req, *detector);
    stream->start();
    return ESP_OK;
}

httpd_uri_t uri_events = {
    .uri      = "/events",
    .method   = HTTP_GET,
    .handler  = handleEventStream,
    .user_ctx = NULL
};



void httpdRun(Detector &d) {
    detector = &d;

    // https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/protocols/esp_http_server.html
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &uri_index);
        httpd_register_uri_handler(server, &uri_flash);
        httpd_register_uri_handler(server, &uri_stream);
        httpd_register_uri_handler(server, &uri_detector);
        httpd_register_uri_handler(server, &uri_events);
    }
}
