#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiClient.h>
#include <esp_camera.h>
#include <esp_wifi.h>
#include <ESP32Ping.h>
#include <TimeLib.h>
#include "http/httpd.h"
#include "image/camera.h"
#include "detection/objectdetector.h"
#include "datalog.h"
#include "wiring.h"
#include "mbot-pwm.h"
#include "common.h"

typedef struct _WifiNetwork {
    const char *ssid;
    const char *password;
} WifiNetwork;

static WifiNetwork wifiNetworks[] = {
    {"krokodil-ap2", "mistress"},
    {"dlink-BF60", "ptfmm78341"},
};

static const char *hostname = "mbot";

Adafruit_SSD1306 oled(128, 32);
static WiFiMulti wifiMulti;

static DataLogger logger;
static Camera camera(logger);

//static BlobDetector detector({175, 60, 75});
static ObjectDetector detector;

static MBotPWM mbot(detector);

void setup() {
    if (!LOG_TO_SDCARD) {
        pinMode(MV_LED_PIN, OUTPUT);
        digitalWrite(MV_LED_PIN, LOW);

        ledcAttachPin(MV_FLASH_PIN, MV_FLASH_CHAN);
        ledcSetup(MV_FLASH_CHAN, 151379, 8);
        ledcWrite(MV_FLASH_CHAN, 0);
    }

    Serial.begin(115200);
    while (!Serial);
    Serial.println("Starting up");
    Serial.printf("Core %d, clock %d MHz\n", xPortGetCoreID(), getCpuFrequencyMhz());
    serialPrint("Total heap: %d\n", ESP.getHeapSize());
    serialPrint("Total PSRAM: %d\n", ESP.getPsramSize());
    
    // Initialize display
    if (!LOG_TO_SDCARD) {
        Wire.setPins(MV_SDA_PIN, MV_SCL_PIN);
        oled.begin();
        oled.setTextColor(1);
        oled.setTextSize(1);
    }
    oledPrint("Starting up");

    // Start the camera frame capture task
    camera.begin();

    // Start object detector
    detector.begin();

    // Connect to Mbot
    if (!LOG_TO_SDCARD) {
        mbot.begin();
    }

    // Connect to Wifi
    oledPrint("WiFi connecting");
    for (auto network : wifiNetworks) {
        wifiMulti.addAP(network.ssid, network.password);
        serialPrint("Added WiFi AP: %s %s\n", network.ssid, network.password);
    }
        
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(hostname);
    WiFi.setAutoReconnect(true);

    // https://github.com/espressif/esp-idf/issues/1366#issuecomment-569377207
    WiFi.persistent(false);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    cfg.nvs_enable = 0;

    while (true) {
        if (wifiMulti.run() == WL_CONNECTED) {
            break;
        }

        Serial.print(".");
        delay(1000);
    }
    delay(500);

    IPAddress ip = WiFi.localIP();
    serialPrint("\nHostname: %s\n", WiFi.getHostname());
    serialPrint("IP: %s\n", ip.toString().c_str());
    serialPrint("Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
    serialPrint("DNS: %s\n\n", WiFi.dnsIP(0).toString().c_str());

    // Send a ping to the router
    bool ret = Ping.ping(WiFi.gatewayIP(), 1);
    delay(500);
    Serial.println(ret ? "Internet gateway was reachable" : "Not able to reach internet gateway");
    oledPrint("%s %s", WiFi.getHostname(), ip.toString().c_str());

    // Use NTP to configure local time
    Serial.print("Retrieving time: ");
    configTime(0, 0, "pool.ntp.org");
    time_t now = time(nullptr);
    while (now < 24 * 3600) {
        Serial.print(".");
        delay(100);
        now = time(nullptr);
    }
    Serial.println(now);
    setTime(time(nullptr));

    // Initialize SD card
    if (LOG_TO_SDCARD) {
        Serial.println("Initializing memory card");
        logger.begin();
    }

    // Start webserver
    httpdRun(detector);

    if (!LOG_TO_SDCARD) {
        digitalWrite(MV_LED_PIN, HIGH);
        ledcWrite(MV_FLASH_CHAN, 0);
    }

    Serial.println("Setup complete");

    serialPrint("Startup completed\n");
    serialPrint("Free heap: %d\n", ESP.getFreeHeap());
    serialPrint("Free PSRAM: %d\n", ESP.getFreePsram());
}

void loop() {
    delay(1000);
}
