#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WiFiClient.h>
#include <esp_camera.h>
#include <esp_wifi.h>
#include <ESP32Ping.h>
#include "camera.h"
#include "httpd.h"
#include "wiring.h"
#include "common.h"

typedef struct _WifiNetwork {
    const char *ssid;
    const char *password;
} WifiNetwork;

WifiNetwork wifiNetworks[] = {
    {"krokodil", "mistress"},
    {"dlink-BF60", "ptfmm78341"},
};

const char *hostname = "mbot";

Adafruit_SSD1306 oled(128, 32);
WiFiMulti wifiMulti;

template <typename... T>
void oledPrint(const char *message, T... args);

void setup() {
    pinMode(MV_LED_PIN, OUTPUT);
    digitalWrite(MV_LED_PIN, LOW);

    ledcAttachPin(MV_FLASH_PIN, MV_FLASH_CHAN);
    ledcSetup(MV_FLASH_CHAN, 151379, 8);
    ledcWrite(MV_FLASH_CHAN, 0);

    Serial.begin(115200);
    while (!Serial);
    Serial.println("Starting up");
    Serial.printf("Core %d, clock %d MHz\n", xPortGetCoreID(), getCpuFrequencyMhz());
    Serial.printf("  XTAL %d MHz, APB %d MHz\n\n", getXtalFrequencyMhz(), getApbFrequency() / 1000000);

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

    // Start the camera frame capture task
    cameraRun();

    // Start webserver
    httpdInit();

    // Connect to Mbot
    //oledPrint("Connecting to MBot");
    //Serial1.setDebugOutput(true);
    //Serial1.begin(115200, SERIAL_8N1, MV_URX_PIN, MV_UTX_PIN);
    //while (!Serial1);

    oledPrint("%s %s", WiFi.getHostname(), ip.toString().c_str());

    digitalWrite(MV_LED_PIN, HIGH);
    Serial.println("Setup complete");

    ledcWrite(MV_FLASH_CHAN, 0);
}

void loop() {
    // Service HTTP requests
    httpdLoop();
    yield();
}
