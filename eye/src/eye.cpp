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
#include <Adafruit_SSD1306.h>
#include "mbot-vision/http/httpd.h"
#include "mbot-vision/image/camera.h"
#include "mbot-vision/detection/objectdetector.h"
#include "mbot-vision/datalog.h"
#include "mbot-vision/wiring.h"
#include "mbot-vision/mbot-pwm.h"
#include "mbot-vision/common.h"
#include "mbot-vision-config.h"

static Adafruit_SSD1306 *oled;
static WiFiMulti *wifiMulti;
static DataLogger *logger;
static Camera *camera;
static ObjectDetector *detector;
static MBotPWM *mbot;

template <typename... T>
void oledPrint(const char *message, T... args) {
    if (LOG_TO_SDCARD) {
        return;
    }
    
    oled->clearDisplay();
    oled->setCursor(0, 0);
    
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        oled->print(buf);
    }
    
    oled->display();
}

static void oledDisplayImage(const uint8_t *image, size_t width, size_t height) {
    oled->drawGrayscaleBitmap(0, 0, image, width, height);
}

void setupWifi() {
    // Connect to Wifi
    oledPrint("WiFi connecting");
    for (auto network : wifiNetworks) {
        wifiMulti->addAP(network.ssid, network.password);
        serialPrint("Added WiFi AP: %s %s\n", network.ssid, network.password);
    }
        
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(MV_HOSTNAME);
    WiFi.setAutoReconnect(true);

    // https://github.com/espressif/esp-idf/issues/1366#issuecomment-569377207
    WiFi.persistent(false);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    cfg.nvs_enable = 0;

    while (true) {
        if (wifiMulti->run() == WL_CONNECTED) {
            break;
        }

        Serial.print(".");
        delay(1000);
    }

    IPAddress ip = WiFi.localIP();
    serialPrint("\nHostname: %s\n", WiFi.getHostname());
    serialPrint("IP: %s\n", ip.toString().c_str());
    serialPrint("Gateway: %s\n", WiFi.gatewayIP().toString().c_str());
    serialPrint("DNS: %s\n\n", WiFi.dnsIP(0).toString().c_str());

    // Send a ping to the router
    bool ret = Ping.ping(WiFi.gatewayIP(), 1);
    Serial.println(ret ? "Internet gateway was reachable" : "Not able to reach internet gateway");
    oledPrint("%s %s", WiFi.getHostname(), ip.toString().c_str());

    // Use NTP to configure local time
    if (LOG_TO_SDCARD) {
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
    }
}

void setup() {
    randomSeed(1);

    // Allocate the ObjectDectector first, it needs to allocate the tensor arena on the internal memory 
    // system heap. If the tensors got allocated in external SPI RAM the model latency is much higher
    detector = new ObjectDetector();

    oled = new Adafruit_SSD1306(128, 32);
    wifiMulti = new WiFiMulti();
    logger = new DataLogger();
    camera = new Camera(*logger);
    mbot = new MBotPWM(*detector);

    if (!LOG_TO_SDCARD) {
        pinMode(MV_LED_PIN, OUTPUT);
        digitalWrite(MV_LED_PIN, LOW);

        ledcAttachPin(MV_FLASH_PIN, MV_FLASH_CHAN);
        ledcSetup(MV_FLASH_CHAN, 151379, 8);
        ledcWrite(MV_FLASH_CHAN, 0);
    }

    Serial.begin(115200);
    Serial.println("Starting up");
    Serial.printf("Core %d, clock %d MHz\n", xPortGetCoreID(), getCpuFrequencyMhz());
    serialPrint("Total heap: %d\n", ESP.getHeapSize());
    serialPrint("Free heap: %d\n", ESP.getFreeHeap());
    serialPrint("Total PSRAM: %d\n", ESP.getPsramSize());
    serialPrint("Free PSRAM: %d\n", ESP.getFreePsram());
    
    // Initialize display
    if (!LOG_TO_SDCARD) {
        Wire.setPins(MV_SDA_PIN, MV_SCL_PIN);
        oled->begin();
        oled->setTextColor(1);
        oled->setTextSize(1);
    }
    oledPrint("Starting up");

    // Start the camera frame capture task. Start this before initializing the wifi and wait for the first 
    // frame to be captured, otherwise sporadic brownouts and wifi connections issues arise.
    camera->begin();
    fbqueue->release(fbqueue->take(FrameBufferItem()));

    // Connect to the Wifi
    setupWifi();

    // Initialize SD card
    if (LOG_TO_SDCARD) {
        Serial.println("Initializing memory card");
        logger->begin();
    }

    // Start object detector
    detector->begin();
    detector->wait();

    // Connect to Mbot
    if (!LOG_TO_SDCARD) {
        mbot->begin();
    }

    // Start webserver
    httpdRun(*detector);

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
    delay(10000);
}
