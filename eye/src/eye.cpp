#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiClient.h>
#include <WebServer.h>
//#include <AsyncTCP.h>
//#include <ESPAsyncWebServer.h>
#include <OV2640.h>
#include <OV2640Streamer.h>
#include <CRtspSession.h>

typedef struct _WifiNetwork {
    const char *ssid;
    const char *password;
} WifiNetwork;

WifiNetwork wifiNetworks[] = {
    {"krokodil-zyxel", "mistress"},
    {"krokodil", "mistress"},
    {"dlink-BF60", "ptfmm78341"},
};

const char *hostname = "mbot";

#define MV_LED_PIN      33
#define MV_FLASH_PIN    12
#define MV_FLASH_CHAN   0
#define MV_FLASH_MAX    150

// I2C bus for display
#define MV_SDA_PIN      13
#define MV_SCL_PIN      15

// UART comms with MBot
#define MV_UTX_PIN      14
#define MV_URX_PIN      2

Adafruit_SSD1306 oled(128, 32);
OV2640 cam;
WiFiMulti wifiMulti;
WebServer server(80);
WiFiServer rtspServer(8554);
CStreamer *streamer;

std::vector<WiFiClient *> streams;

template <typename... T>
void oledPrint(const char *message, T... args);

void handleIndex();
void handleFlash();
void handleJpegStream();
void handleJpeg();
void handleNotFound();
void loopRTSP();

void setup() {
    pinMode(MV_LED_PIN, OUTPUT);
    digitalWrite(MV_LED_PIN, LOW);

    ledcAttachPin(MV_FLASH_PIN, MV_FLASH_CHAN);
    ledcSetup(MV_FLASH_CHAN, 151379, 8);
    ledcWrite(MV_FLASH_CHAN, 0);

    Serial.begin(115200);
    while (!Serial);
    Serial.println("Starting up");

    // Initialize display
    Wire.setPins(MV_SDA_PIN, MV_SCL_PIN);
    oled.begin();
    oled.setTextColor(1);
    oled.setTextSize(1);
    oledPrint("Starting up");
    
    // Connect to Wifi
    oledPrint("WiFi connecting");
    for (auto network : wifiNetworks) {
        wifiMulti.addAP(network.ssid, network.password);
        Serial.print("Added WiFi AP: ");
        Serial.print(network.ssid);
        Serial.print(" ");
        Serial.println(network.password);
    }

    WiFi.mode(WIFI_STA);
    WiFi.setHostname(hostname);
    WiFi.setAutoReconnect(true);
    
    while (wifiMulti.run() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    // Initialize camera
    oledPrint("Init camera");
    if (cam.init(esp32cam_aithinker_config) != ESP_OK) {
        oledPrint("Camera failed");
        while (true);
    }

    // Start webserver
    server.on("/", HTTP_GET, handleIndex);
    server.on("/flash", HTTP_POST, handleFlash);
    server.on("/stream", HTTP_GET, handleJpegStream);
    server.on("/jpg", HTTP_GET, handleJpeg);
    server.onNotFound(handleNotFound);
    server.begin();

    // Start RTSP server
    rtspServer.begin();
    streamer = new OV2640Streamer(cam);

    // Connect to Mbot
    //oledPrint("Connecting to MBot");
    //Serial1.setDebugOutput(true);
    //Serial1.begin(115200, SERIAL_8N1, MV_URX_PIN, MV_UTX_PIN);
    //while (!Serial1);

    IPAddress ip = WiFi.localIP();
    Serial.print("\nWiFi connected. IP: ");
    Serial.print(ip);
    Serial.print(", hostname: ");
    Serial.println(WiFi.getHostname());
    oledPrint("Host: %s\nIP: %s", WiFi.getHostname(), ip.toString().c_str());

    digitalWrite(MV_LED_PIN, HIGH);
    Serial.println("Setup complete");
}

void loop() {
    // Service HTTP requests
    server.handleClient();

    // Service RTSP clients
    loopRTSP();

    // Take a snapshot
    cam.run();

    // Send snapshot to each stream client
    for (int i = 0; i < streams.size(); ) {
        WiFiClient *client = streams[i];
        if (client->connected()) {
            client->write("--gc0p4Jq0M2Yt08jU534c0p\r\n");
            client->write("Content-Type: image/jpeg\r\n\r\n");
            client->write((char *)cam.getfb(), cam.getSize());
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

template <typename... T>
void oledPrint(const char *message, T... args) {
    oled.clearDisplay();
    oled.setCursor(0, 0);
    
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        oled.println(buf);
    }
    
    oled.display();
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
        
        // https://gist.github.com/nmsdvid/8807205#gistcomment-3325286
        const debounce = (callback, wait = 250) => {
            let timer;
            return (...args) => {
                clearTimeout(timer);
                timer = setTimeout(() => callback(...args), wait);
            };
        };

        const handleFlash = debounce((value) => {
            post(`/flash?v=${value}`);
        });
    </script>
  </head>
  <body style="background-color: #000;">
    <div>
        <input style="width:100%; max-width:800px" type="range" min="0" max="150" value="0" id="flash" oninput="handleFlash(this.value);" onchange="handleFlash(this.value);">
    </div>
    <img src="/stream" width="800" height="600" />
  <body>
<html>)doc";

    server.sendContent(response);
    server.sendContent("\r\n");
}

void handleFlash() {
    String value = server.arg(0);
    int duty = value.toInt();
    duty = std::max(0, std::min(duty, MV_FLASH_MAX));
    
    if (duty < MV_FLASH_MAX / 10) {
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

void handleJpeg() {
    WiFiClient client = server.client();

    cam.run();
    if (!client.connected()) {
        return;
    }

    String response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Disposition: inline; filename=capture.jpg\r\n";
    response += "Content-Type: image/jpeg\r\n\r\n";
    server.sendContent(response);
    client.write((char *)cam.getfb(), cam.getSize());
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

void loopRTSP() {
    uint32_t msecPerFrame = 100;
    static uint32_t lastimage = millis();

    // If we have an active client connection, just service that until gone
    streamer->handleRequests(0); // we don't use a timeout here,
    // instead we send only if we have new enough frames
    uint32_t now = millis();
    if(streamer->anySessions()) {
        if(now > lastimage + msecPerFrame || now < lastimage) { // handle clock rollover
            streamer->streamImage(now);
            lastimage = now;

            // check if we are overrunning our max frame rate
            now = millis();
            if(now > lastimage + msecPerFrame) {
                printf("warning exceeding max frame rate of %d ms\n", now - lastimage);
            }
        }
    }
    
    WiFiClient rtspClient = rtspServer.accept();
    if(rtspClient) {
        Serial.print("client: ");
        Serial.print(rtspClient.remoteIP());
        Serial.println();
        streamer->addSession(rtspClient);
    }
}
