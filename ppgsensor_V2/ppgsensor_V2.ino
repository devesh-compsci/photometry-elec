// PPG_reader.ino
// Reads analog PPG signal and sends via Serial at high baud rate

const int PPG_PIN = A0;
const int SAMPLE_RATE = 500;  // Hz — well above 48Hz hardware cutoff
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE;

unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(500000);
  analogReference(DEFAULT);   // 5V ref on Uno, 3.3V on 3.3V boards
  // Increase ADC speed for cleaner sampling
  // ADCSRA = (ADCSRA & ~0x07) | 0x04; // prescaler 16 → ~77kHz ADC clock
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime = now;
    int raw = analogRead(PPG_PIN);
    // Send as raw integer line — Python parses this
    Serial.println(raw);
  }
}
