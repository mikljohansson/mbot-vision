#include <Arduino.h>
#include <WebServer.h>
#include "httpd.h"
#include "wiring.h"

WebServer server(80);
std::vector<WiFiClient *> streams;

void handleIndex();
void handleFlash();
void handleJpegStream();
void handleNotFound();

void httpdInit() {
    server.on("/", HTTP_GET, handleIndex);
    server.on("/flash", HTTP_POST, handleFlash);
    server.on("/stream", HTTP_GET, handleJpegStream);
    server.onNotFound(handleNotFound);
    server.begin();
}

void httpdLoop() {
    server.handleClient();
}

size_t httpdWriteStream(void * arg, size_t index, const void* data, size_t len) {
    size_t written = 0;
    
    for (int i = 0; i < streams.size(); ) {
        WiFiClient *client = streams[i];
        if (client->connected()) {
            written = max(client->write((const char *)data, len), written);
        }
    }

    return written;
}

void httpdSendStream(camera_fb_t *fb) {
    for (int i = 0; i < streams.size(); ) {
        WiFiClient *client = streams[i];
        if (client->connected()) {
            client->write("--gc0p4Jq0M2Yt08jU534c0p\r\n");
            client->write("Content-Type: image/jpeg\r\n\r\n");
        }
    }

    if (!frame2jpg_cb(fb, 25, httpdWriteStream, 0)) {
        Serial.println("Failed to convert framebuffer to jpeg");
    }

    for (int i = 0; i < streams.size(); ) {
        WiFiClient *client = streams[i];
        if (client->connected()) {
            client->write("\r\n");
        }
        
        if (client->connected()) {
            i++;
        }
        else {
            delete client;
            streams.erase(streams.begin() + i);
        }
    }
}

void handleIndex() {
    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: text/html\r\n\r\n";
    response += R"doc(<html>
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
    </script>
    <style>
      .body {
        background-color: #000;
        display: flex;
        justify-content: center;
      }

      .container {
        width: 100%;
        height: 100%;
        max-width: 800px;
        max-height: 600px;
        padding: 5px;
        background-image: url("/stream");
        background-repeat: no-repeat;
        background-size: contain;
        background-position: top center;
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
    <table class="container">
      <td>&#x1F4A1;</td>
      <td class="inputcell"><input type="range" min="0" max="175" value="0" id="flash" oninput="handleFlash(this.value);" onchange="handleFlash(this.value);"></td>
    </table>
  <body>
<html>)doc";

    server.sendContent(response);
    server.sendContent("\r\n");
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
    
    WiFiClient client = server.client();
    client.write("HTTP/1.1 200 OK\r\n");
}

void handleJpegStream() {
    WiFiClient *client = new WiFiClient();
    *client = server.client();
    client->write("HTTP/1.1 200 OK\r\n");
    client->write("Content-Type: multipart/x-mixed-replace; boundary=gc0p4Jq0M2Yt08jU534c0p\r\n\r\n");
    streams.push_back(client);
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
