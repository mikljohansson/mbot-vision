#include <Arduino.h>
#include <ESPAsyncWebServer.h>
#include "http/httpd.h"
#include "image/camera.h"
#include "framerate.h"
#include "detection/detector.h"
#include "wiring.h"
#include "common.h"

#include "http/index.h"
#include "http/jpeg-stream.h"
#include "http/event-stream.h"

static AsyncWebServer server(80);
static AsyncEventSource events("/events");
static Detector *detector;

void handleIndex(AsyncWebServerRequest *request) {
    Serial.print("Sending index.html to: ");
    Serial.println(request->client()->remoteIP());
    request->send(200, "text/html", indexDocument);
}

void handleFlash(AsyncWebServerRequest *request) {
    String value = request->arg((size_t)0);
    int duty = value.toInt();
    duty = std::max(0, std::min(duty, MV_FLASH_MAX));
    
    if (duty < MV_FLASH_MAX / 15) {
        duty = 0;
    }
    
    if (!LOG_TO_SDCARD) {
        Serial.println("Turning flash to value " + duty);
        ledcWrite(MV_FLASH_CHAN, duty);
    }

    request->send(200, "text/plain", "OK");
}

void handleJpegStream(AsyncWebServerRequest *request) {
    Serial.print("MJPEG stream connected: ");
    Serial.println(request->client()->remoteIP());
    request->send(new JpegStream(false, *detector));
}

void handleDetectorStream(AsyncWebServerRequest *request) {
    Serial.print("MJPEG model output stream connected: ");
    Serial.println(request->client()->remoteIP());
    request->send(new JpegStream(true, *detector));
}

void handleEventStream(AsyncWebServerRequest *request) {
    Serial.print("Event stream connected: ");
    Serial.println(request->client()->remoteIP());
    request->send(new EventStream(*detector));
}

void handleNotFound(AsyncWebServerRequest *request) {
    String message = "Server is running!\n\n";
    message += "URI: ";
    message += request->url();
    message += "\nMethod: ";
    message += (request->method() == HTTP_GET) ? "GET" : "POST";
    message += "\nArguments: ";
    message += request->args();
    message += "\n";
    request->send(200, "text/plain", message);
}

void httpdRun(Detector &d) {
    detector = &d;

    server.on("/", HTTP_GET, handleIndex);
    server.on("/flash", HTTP_POST, handleFlash);
    server.on("/stream", HTTP_GET, handleJpegStream);
    server.on("/detector", HTTP_GET, handleDetectorStream);
    server.on("/events", HTTP_GET, handleEventStream);
    server.onNotFound(handleNotFound);
    server.begin();
}
