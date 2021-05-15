#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiClient.h>
#include <esp_camera.h>
#include "camera.h"
#include "httpd.h"
#include "model.h"
#include "wiring.h"

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

Adafruit_SSD1306 oled(128, 32);
WiFiMulti wifiMulti;

// Framerate calculation
double windowTime;
double windowFrames;
unsigned long lastUpdatedWindow;
unsigned long lastShowedFramerate = 0;

template <typename... T>
void oledPrint(const char *message, T... args);

template <typename... T>
void serialPrint(const char *message, T... args);

void setup() {
    pinMode(MV_LED_PIN, OUTPUT);
    digitalWrite(MV_LED_PIN, LOW);

    ledcAttachPin(MV_FLASH_PIN, MV_FLASH_CHAN);
    ledcSetup(MV_FLASH_CHAN, 151379, 8);
    ledcWrite(MV_FLASH_CHAN, 0);

    Serial.begin(115200);
    while (!Serial);
    Serial.println("Starting up");

    serialPrint("Total heap: %d\n", ESP.getHeapSize());
    serialPrint("Free heap: %d\n", ESP.getFreeHeap());
    serialPrint("Total PSRAM: %d\n", ESP.getPsramSize());
    serialPrint("Free PSRAM: %d\n", ESP.getFreePsram());

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
        serialPrint("Added WiFi AP: %s %s\n", network.ssid, network.password);
    }

    while (true) {
        WiFi.mode(WIFI_STA);
        WiFi.setHostname(hostname);
        WiFi.setAutoReconnect(true);
    
        if (wifiMulti.run() == WL_CONNECTED) {
            break;
        }

        WiFi.disconnect(true);
        WiFi.mode(WIFI_OFF);
        delay(500);        
        Serial.print(".");
    }

    // Initialize camera
    oledPrint("Init camera");
    while (true) {
        esp_err_t err = esp_camera_init(&mv_camera_aithinker_config);
        if (err == ESP_OK) {
            break;
        }

        oledPrint("Camera probe failed with error 0x%x", err);
        delay(250);
    }

    // Start webserver
    httpdInit();

    // Connect to Mbot
    //oledPrint("Connecting to MBot");
    //Serial1.setDebugOutput(true);
    //Serial1.begin(115200, SERIAL_8N1, MV_URX_PIN, MV_UTX_PIN);
    //while (!Serial1);

    IPAddress ip = WiFi.localIP();
    serialPrint("Hostname: %s IP: %s\n", WiFi.getHostname(), ip.toString().c_str());
    oledPrint("%s %s", WiFi.getHostname(), ip.toString().c_str());

    // Load the ML model
    modelInit();

    digitalWrite(MV_LED_PIN, HIGH);
    Serial.println("Setup complete");
}

void detect(camera_fb_t *fb) {

}

void loop() {
    // Service HTTP requests
    httpdLoop();

    if (httpdStreamCount() > 0) {
        // Send a frame from the camera
        camera_fb_t *fb = esp_camera_fb_get();
        //detect(fb);        
        httpdSendStream(fb);
        esp_camera_fb_return(fb);

        // Calculate the framerate
        while (windowFrames >= 25.0) {
            windowTime -= (windowTime / windowFrames);
            windowFrames--;
        }

        unsigned long ts = millis();
        windowTime += (ts - lastUpdatedWindow);
        windowFrames++;
        lastUpdatedWindow = ts;

        // Display the framerate
        if (ts - lastShowedFramerate > 5000) {
            serialPrint("Framerate: %02f\n", windowFrames / (windowTime / 1000.0));
            lastShowedFramerate = ts;
        }
    }
    else {
        windowTime = 1;
        windowFrames = 0;
        lastUpdatedWindow = millis();
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
        oled.print(buf);
    }
    
    oled.display();
}

template <typename... T>
void serialPrint(const char *message, T... args) {
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        Serial.print(buf);
    }
}
