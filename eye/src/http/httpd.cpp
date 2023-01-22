#include <Arduino.h>
#include <WebServer.h>
#include "http/httpd.h"
#include "image/camera.h"
#include "framerate.h"
#include "detection/detector.h"
#include "wiring.h"
#include "common.h"

#include "http/index.h"
#include "http/jpeg-stream.h"
#include "http/event-stream.h"

static WebServer server(80);
static TaskHandle_t httpdTask;
static Detector *detector;

void handleIndex() {
    Serial.print("Sending index.html to: ");
    Serial.println(server.client().remoteIP());
    server.send(200, "text/html", indexDocument);
}

void handleFlash() {
    String value = server.arg(0);
    int duty = value.toInt();
    duty = std::max(0, std::min(duty, MV_FLASH_MAX));
    
    if (duty < MV_FLASH_MAX / 15) {
        duty = 0;
    }
    
    if (!LOG_TO_SDCARD) {
        Serial.println("Turning flash to value " + duty);
        ledcWrite(MV_FLASH_CHAN, duty);
    }

    server.send(200, "text/plain", "OK");
}

void handleJpegStream() {
    JpegStream *stream = new JpegStream(server.client(), false, *detector);
    stream->start();
}

void handleDetectorStream() {
    JpegStream *stream = new JpegStream(server.client(), true, *detector);
    stream->start();
}

void handleEventStream() {
    EventStream *stream = new EventStream(server.client(), *detector);
    stream->start();
}

void handleNotFound() {
    String message = "Server is running!\n\n";
    message += "URI: ";
    message += server.uri();
    message += "\nMethod: ";
    message += (server.method() == HTTP_GET) ? "GET" : "POST";
    message += "\nArguments: ";
    message += server.args();
    message += "\n";
    server.send(200, "text/plain", message);
}

void httpdServiceRequests(void *p) {
    server.on("/", HTTP_GET, handleIndex);
    server.on("/flash", HTTP_POST, handleFlash);
    server.on("/stream", HTTP_GET, handleJpegStream);
    server.on("/detector", HTTP_GET, handleDetectorStream);
    server.on("/events", HTTP_GET, handleEventStream);
    server.onNotFound(handleNotFound);
    server.begin();

    while (true) {
        // Service HTTP requests
        server.handleClient();
        
        // Prevents watchdog from triggering
        delay(1);
    }

    vTaskDelete(NULL);
}

void httpdRun(Detector &d) {
    detector = &d;
    xTaskCreatePinnedToCore(httpdServiceRequests, "httpd", 10000, NULL, 2, &httpdTask, 0);
}
