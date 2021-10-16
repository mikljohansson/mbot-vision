#include <Arduino.h>
#include <WebServer.h>
#include "httpd.h"
#include "camera.h"
#include "framerate.h"
#include "blobdetector.h"
#include "wiring.h"

static WebServer server(80);
static TaskHandle_t httpdTask;
static BlobDetector *detector;

class JpegStream {
    private:
        WiFiClient _client;
        TaskHandle_t _task;
        Framerate _framerate;

    public:
        JpegStream(const WiFiClient &client)
         : _client(client), _framerate("Stream framerate: %02f\n") {}

        void start() {
            Serial.print("Stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "jpegStream", 10000, this, 3, &_task, 1);
        }

    private:
        void send(const void *data, size_t len) {
            for (size_t written = 0; data && _client.connected() && written < len; yield()) {
                written += _client.write(((const uint8_t *)data) + written, len - written);
            }
        }

        void send(const char *data) {
            send(data, strlen(data));
        }

        void run() {
            send("HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=gc0p4Jq0M2Yt08jU534c0p\r\n");
            _framerate.init();

            while (true) {
                camera_fb_t *fb = fbqueue->take();

                if (fb) {
                    String headers = "\r\n--gc0p4Jq0M2Yt08jU534c0p\r\nContent-Type: image/jpeg\r\n";

                    if (fb->format == PIXFORMAT_JPEG) {
                        headers += "Content-Length: ";
                        headers += fb->len;
                        headers += "\r\n";
                    }

                    headers += "\r\n";
                    send(headers.c_str(), headers.length());

                    if (fb->format == PIXFORMAT_JPEG) {
                        send(fb->buf, fb->len);
                    }
                    else if (!frame2jpg_cb(fb, 40, sendStatic, this)) {
                        Serial.println("Failed to convert framebuffer to jpeg");
                    }

                    fbqueue->release(fb);
                    _framerate.tick();
                }

                if (!_client.connected()) {
                    Serial.print("HTTP stream disconnected: ");
                    Serial.println(_client.remoteIP());
                    break;
                }
            }
        }

        static void runStatic(void *p) {
            JpegStream *stream = (JpegStream *)p;
            stream->run();
            delete stream;
            vTaskDelete(NULL);
        }

        static size_t sendStatic(void *p, size_t index, const void *data, size_t len) {
            JpegStream *stream = (JpegStream *)p;
            stream->send(data, len);
            return len;
        }
};

class EventStream {
    private:
        WiFiClient _client;
        TaskHandle_t _task;
        Framerate _framerate;

    public:
        EventStream(const WiFiClient &client)
         : _client(client), _framerate("Event stream framerate: %02f\n") {}

        void start() {
            Serial.print("Event stream connected: ");
            Serial.println(_client.remoteIP());
            xTaskCreatePinnedToCore(runStatic, "eventStream", 10000, this, 2, &_task, 1);
        }

    private:
        void send(const void *data, size_t len) {
            for (size_t written = 0; data && _client.connected() && written < len; yield()) {
                written += _client.write(((const uint8_t *)data) + written, len - written);
            }
        }

        void send(const char *data) {
            send(data, strlen(data));
        }

        void run() {
            send("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
            _framerate.init();

            while (true) {
                DetectedBlob blob = detector->get();
                
                String data = "data: ";
                blob.serialize(data);
                data += "\r\n\r\n";
                send(data.c_str(), data.length());

                _framerate.tick();

                if (!_client.connected()) {
                    Serial.print("HTTP event stream disconnected: ");
                    Serial.println(_client.remoteIP());
                    break;
                }
            }
        }

        static void runStatic(void *p) {
            EventStream *stream = (EventStream *)p;
            stream->run();
            delete stream;
            vTaskDelete(NULL);
        }
};

void handleIndex();
void handleFlash();
void handleJpegStream();
void handleEventStream();
void handleNotFound();

void httpdServiceRequests(void *p) {
    server.on("/", HTTP_GET, handleIndex);
    server.on("/flash", HTTP_POST, handleFlash);
    server.on("/stream", HTTP_GET, handleJpegStream);
    server.on("/events", HTTP_GET, handleEventStream);
    server.onNotFound(handleNotFound);
    server.begin();

    while (true) {
        // Service HTTP requests
        server.handleClient();
        yield();
    }

    vTaskDelete(NULL);
}

void httpdRun(BlobDetector &d) {
    detector = &d;
    xTaskCreatePinnedToCore(httpdServiceRequests, "httpd", 10000, NULL, 0, &httpdTask, 1);
}

String indexDocument = R"doc(<html>
  <head>
    <link rel="icon" href="data:;base64,=">
    <script type="text/javascript">
        const post = (url) => {
            var xmlHttp = new XMLHttpRequest();
            xmlHttp.open("POST", url, true);
            xmlHttp.send(null);
        };
        
        const debounce = (callback, wait = 250) => {
            let timeout = null;
            let useargs = null;

            return (...args) => {
                if (!timeout) {
                    useargs = args;
                    timeout = setTimeout(() => {
                        timeout = null;
                        callback(...useargs);
                    }, wait);
                }
                else {
                    useargs = args;
                }
            };
        };

        const handleFlash = debounce((value) => {
            post(`/flash?v=${value}`);
        });

        const source = new EventSource('/events');
        source.onmessage = (e) => {
            const crosshair = document.getElementById("crosshair");
            const blob = JSON.parse(e.data);
            const rect = crosshair.parentElement.getBoundingClientRect();
            crosshair.style.left = Math.round(rect.width * blob.x) + "px";
            crosshair.style.top = Math.round(rect.height * blob.y) + "px";

        };
    </script>
    <style>
      .body {
        background-color: #000;
        display: flex;
        justify-content: center;
      }

      .container {
        display: inline-block;
        position: relative;
        width: 100%;
        height: 100%;
        max-width: 320px;
        max-height: 240px;
        padding: 5px;
        background-image: url("/stream");
        background-repeat: no-repeat;
        background-size: contain;
        background-position: top center;
      }

      #crosshair {
        position: absolute;
        color: white;
        font-size: 200%;
        transform: translate(-50%, -50%);
      }

      .container td {
        vertical-align: top;
      }

      .inputcell {
        width: 99%;
      }

      .container input {
        width: 100%;
      }
    </style>
  </head>
  <body class="body">
    <div class="container">
      <div id="crosshair">&#x2316</div>
      <table>
        <td>&#x1F4A1;</td>
        <td class="inputcell"><input type="range" min="0" max="175" value="0" id="flash" oninput="handleFlash(this.value);" onchange="handleFlash(this.value);"></td>
      </table>
    </div>
  <body>
<html>)doc";

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

    Serial.println("Turning flash to value " + value);
    ledcWrite(MV_FLASH_CHAN, duty);

    server.send(200, "text/plain", "OK");
}

void handleJpegStream() {
    JpegStream *stream = new JpegStream(server.client());
    stream->start();
}

void handleEventStream() {
    EventStream *stream = new EventStream(server.client());
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
