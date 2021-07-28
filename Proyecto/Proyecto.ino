// DHT Temperature & Humidity Sensor
// Unified Sensor Library Example
// Written by Tony DiCola for Adafruit Industries
// Released under an MIT license.

// REQUIRES the following Arduino libraries:
// - DHT Sensor Library: https://github.com/adafruit/DHT-sensor-library
// - Adafruit Unified Sensor Lib: https://github.com/adafruit/Adafruit_Sensor

#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>

#define DHTPIN 6     // Digital pin connected to the DHT sensor 
// Feather HUZZAH ESP8266 note: use pins 3, 4, 5, 12, 13 or 14 --
// Pin 15 can work but DHT must be disconnected during program upload.

// Uncomment the type of sensor in use:
//#define DHTTYPE    DHT11     // DHT 11
#define DHTTYPE    DHT22     // DHT 22 (AM2302)
//#define DHTTYPE    DHT21     // DHT 21 (AM2301)

// See guide for details on sensor wiring and usage:
//   https://learn.adafruit.com/dht/overview

DHT_Unified dht(DHTPIN, DHTTYPE);

uint32_t delayMS;

/*
* Demo name  ?: HP20x_dev demo
* Usage      ?: I2C PRECISION BAROMETER AND ALTIMETER [HP206C hopeRF]
* Author     ?: Oliver Wang from Seeed Studio
* Version    ?: V0.1
* Change log ?: Add kalman filter 2014/04/04
*/
 
#include <HP20x_dev.h>
#include <KalmanFilter.h>
#include "Arduino.h"
#include "Wire.h"
 
unsigned char ret = 0;
/* Instance */
KalmanFilter t_filter;    //temperature filter
KalmanFilter p_filter;    //pressure filter
KalmanFilter a_filter;    //altitude filter

/*                          AN-157 Demo of T-66 using Software Serial

 Arduino example for t6613 CO2 sensor 0-2000 PPM   19200 BPS
 2/2017 by Marv kausch @ Co2meter.com
*/
#include "SoftwareSerial.h"
SoftwareSerial T66_Serial(12,13); //Sets up a virtual serial port
 //Using pin 12 for Rx and pin 13 for Tx
byte readCO2[] = {0xFF, 0XFE,2,2,3}; //T66 read CO2 command: 5 bytes

byte response[] = {0,0,0,0,0}; //create an array to store the response


void setup() {
  //DHT22_____________________________________________________________________________________
  Serial.begin(9600);
  // Initialize device.
  dht.begin();
  Serial.println(F("DHTxx Unified Sensor Example"));
  // Print temperature sensor details.
  sensor_t sensor;
  dht.temperature().getSensor(&sensor);
  Serial.println(F("------------------------------------"));
  Serial.println(F("Temperature Sensor"));
  Serial.print  (F("Sensor Type: ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:  ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:   ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:   ")); Serial.print(sensor.max_value); Serial.println(F("°C"));
  Serial.print  (F("Min Value:   ")); Serial.print(sensor.min_value); Serial.println(F("°C"));
  Serial.print  (F("Resolution:  ")); Serial.print(sensor.resolution); Serial.println(F("°C"));
  Serial.println(F("------------------------------------"));
  // Print humidity sensor details.
  dht.humidity().getSensor(&sensor);
  Serial.println(F("Humidity Sensor"));
  Serial.print  (F("Sensor Type: ")); Serial.println(sensor.name);
  Serial.print  (F("Driver Ver:  ")); Serial.println(sensor.version);
  Serial.print  (F("Unique ID:   ")); Serial.println(sensor.sensor_id);
  Serial.print  (F("Max Value:   ")); Serial.print(sensor.max_value); Serial.println(F("%"));
  Serial.print  (F("Min Value:   ")); Serial.print(sensor.min_value); Serial.println(F("%"));
  Serial.print  (F("Resolution:  ")); Serial.print(sensor.resolution); Serial.println(F("%"));
  Serial.println(F("------------------------------------"));
  // Set delay between sensor readings based on sensor details.
  delayMS = sensor.min_delay / 5000;

  //HP20x_____________________________________________________________________________________
  Serial.begin(9600);        // start serial for output
 
  Serial.println("****HP20x_dev demo by seeed studio****\n");
  Serial.println("Calculation formula: H = [8.5(101325-P)]/100 \n");
  /* Power up,delay 150ms,until voltage is stable*/
  delay(150);
  /* Reset HP20x_dev*/
  HP20x.begin();
  delay(100);
 
  /* Determine HP20x_dev is available or not*/
  ret = HP20x.isAvailable();
  if(OK_HP20X_DEV == ret)
  {
      Serial.println("HP20x_dev is available.\n");
  }
  else
  {
      Serial.println("HP20x_dev isn't available.\n");
  }

 //T6615-50KF________________________________________________________________________________
 Serial.begin(19200); //Opens the main serial port to communicate with the computer
 T66_Serial.begin(19200); //Opens the virtual serial port with a baud of 9600
 Serial.println("    Demo of AN-157  Software Serial and T66 sensor");
 Serial.print("\n");
}


void loop() {
  //DHT22____________________________________________________________________________________
  // Delay between measurements.
  delay(delayMS);
  // Get temperature event and print its value.
  sensors_event_t event;
  dht.temperature().getEvent(&event);
  if (isnan(event.temperature)) {
    Serial.println(F("Error reading temperature!"));
  }
  else {
    Serial.print(F("Temperature: "));
    Serial.print(event.temperature);
    Serial.println(F("°C"));
  }
  // Get humidity event and print its value.
  dht.humidity().getEvent(&event);
  if (isnan(event.relative_humidity)) {
    Serial.println(F("Error reading humidity!"));
  }
  else {
    Serial.print(F("Humidity: "));
    Serial.print(event.relative_humidity);
    Serial.println(F("%"));
  }

  //HP20x___________________________________________________________________________________
  char display[40];
  if(OK_HP20X_DEV == ret)
  {
    Serial.println("------------------\n");
    long Temper = HP20x.ReadTemperature();
    Serial.println("Temper:");
    float t = Temper/100.0;
    Serial.print(t);
    Serial.println("C.\n");
    Serial.println("Filter:");
    Serial.print(t_filter.Filter(t));
    Serial.println("C.\n");
 
    long Pressure = HP20x.ReadPressure();
    Serial.println("Pressure:");
    t = Pressure/100.0;
    Serial.print(t);
    Serial.println("hPa.\n");
    Serial.println("Filter:");
    Serial.print(p_filter.Filter(t));
    Serial.println("hPa\n");
 
    long Altitude = HP20x.ReadAltitude();
    Serial.println("Altitude:");
    t = Altitude/100.0;
    Serial.print(t);
    Serial.println("m.\n");
    Serial.println("Filter:");
    Serial.print(a_filter.Filter(t));
    Serial.println("m.\n");
    Serial.println("------------------\n");
    delay(5000);
  }

  //T6615-50KF________________________________________________________________________________
  sendRequest(readCO2);   //Locate the problem of program reset whduring this function call
 
  unsigned long valCO2 = getValue(response);// Request from sensor 5 bytes of data
  Serial.print("Sensor response:   ");
  for(int i=0;i<5;i++)
  {
      Serial.print(response[i],HEX);
      Serial.print(" ");
  }
  Serial.print("    Co2 ppm = ");
  Serial.println(valCO2);
  delay(2000);  //T6613 spec indicates signal update every 4 seconds
}


void sendRequest(byte packet[])
{
 while(!T66_Serial.available()) //keep sending request until we start to get a response
 {
 T66_Serial.write(readCO2,5);// Write to sensor 5 byte command
 delay(50);
 delay(1000);
 }
 int timeout=0; //set a timeoute counter
 while(T66_Serial.available() < 5 ) //Wait to get a 7 byte response
 {
 timeout++;
 if(timeout > 10) //if it takes to long there was probably an error
 Serial.print("Timeout");
 {
 while(T66_Serial.available()) //flush whatever we have
 T66_Serial.read();

 break; //exit and try again
 }
 delay(50);
 }
  for (int i=0; i < 5; i++) response[i] = T66_Serial.read();
}

unsigned long getValue(byte packet[])
{
 int high = packet[3]; //high byte for value is 4th byte in packet in the packet
 int low = packet[4]; //low byte for value is 5th byte in the packet
 unsigned long val = high*256 + low; //Combine high byte and low byte with this formula to get value
 return val;
}
