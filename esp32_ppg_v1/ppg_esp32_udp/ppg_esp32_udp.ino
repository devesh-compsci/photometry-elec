/*
  PPG Wireless Monitor — ESP32-S3
  ================================
  Reads analog PPG signal from GPIO1 (ADC1_CH0) and streams samples
  over UDP to a Python dashboard on the same WiFi network.

  Packet format (CSV per line):
    <sample_index>,<adc_raw>,<timestamp_ms>\n

  Wiring:
    PPG sensor OUT  →  GPIO1  (ADC1 channel 0, 12-bit, 0–4095)
    PPG sensor VCC  →  3.3V
    PPG sensor GND  →  GND

  Dependencies: none beyond Arduino ESP32 core
*/

#include <WiFi.h>
#include <WiFiUdp.h>

// ── CONFIG ────────────────────────────────────────────────────────────────────
const char* SSID         = "CHCI";
const char* PASSWORD     = "CHCI@54321#";

// IP of the PC running the Python dashboard.
// Run ipconfig (Windows) / ip addr (Linux) and paste your IPv4 here.
const char* DEST_IP      = "192.168.1.161";
const uint16_t DEST_PORT = 5005;

// Sampling
const int   PPG_PIN      = 6;          // GPIO1 = ADC1_CH0 on most ESP32-S3 boards
const int   SAMPLE_RATE  = 100;        // Hz — keep ≤200 for stable WiFi + ADC
const int   BATCH_SIZE   = 10;         // samples per UDP packet (reduces overhead)
// ─────────────────────────────────────────────────────────────────────────────

WiFiUDP udp;
uint32_t sampleIndex = 0;
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE;

char packetBuf[512];
int  packetLen = 0;
int  batchCount = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n=== PPG Wireless Monitor ===");
  Serial.printf("Target: %s:%d  |  Fs=%d Hz  |  Batch=%d\n",
                DEST_IP, DEST_PORT, SAMPLE_RATE, BATCH_SIZE);

  // ADC config — 12-bit, attenuation for 0–3.3 V range
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);   // full 0–3.3 V range

  // WiFi
  Serial.printf("Connecting to %s", SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(SSID, PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());

  udp.begin(4999);   // local port (any unused port)
  packetLen = 0;
  batchCount = 0;
  lastSampleTime = micros();
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime < SAMPLE_INTERVAL_US) return;
  lastSampleTime += SAMPLE_INTERVAL_US;

  int adcRaw = analogRead(PPG_PIN);
  unsigned long ts = millis();

  // Append sample to batch buffer
  packetLen += snprintf(packetBuf + packetLen,
                        sizeof(packetBuf) - packetLen,
                        "%lu,%d,%lu\n",
                        (unsigned long)sampleIndex, adcRaw, ts);
  sampleIndex++;
  batchCount++;

  if (batchCount >= BATCH_SIZE || packetLen > 400) {
    udp.beginPacket(DEST_IP, DEST_PORT);
    udp.write((uint8_t*)packetBuf, packetLen);
    udp.endPacket();
    packetLen = 0;
    batchCount = 0;
  }
}
