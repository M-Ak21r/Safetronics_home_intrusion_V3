/*
 * Lockdown Protocol - Door Lock Mechanism Controller
 * ===================================================
 * 
 * This firmware controls a servo motor attached to door hinges for
 * active suspect containment. When a theft is detected by the vision
 * system, it receives a lockdown signal and immediately locks the door.
 * 
 * Hardware:
 * - Arduino (Uno, Nano, or similar)
 * - Servo motor (e.g., SG90, MG996R)
 * - External power supply for servo (recommended for high-torque servos)
 * 
 * Wiring:
 * - Servo signal wire -> Pin 9
 * - Servo power -> 5V or external power
 * - Servo ground -> GND
 * 
 * Serial Protocol:
 * - Baud rate: 9600
 * - Command 'L': Trigger lockdown (close/lock door)
 * - Command 'U': Unlock door (return to open position)
 * 
 * Author: Safetronics
 */

#include <Servo.h>

// Pin configuration
const int SERVO_PIN = 9;

// Servo position constants (degrees)
const int POSITION_OPEN = 0;      // Door open/unlocked position
const int POSITION_LOCKED = 180;  // Door closed/locked position

// State tracking
enum DoorState {
  STATE_OPEN,
  STATE_LOCKED
};

// Global objects
Servo doorServo;
DoorState currentState = STATE_OPEN;

void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);
  
  // Attach servo to pin
  doorServo.attach(SERVO_PIN);
  
  // Initialize to open position (idle state)
  doorServo.write(POSITION_OPEN);
  currentState = STATE_OPEN;
}

void loop() {
  // Check for incoming serial data
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    // Process command immediately - no delay in actuation path
    if (command == 'L') {
      // Lockdown command received - close door immediately
      if (currentState != STATE_LOCKED) {
        doorServo.write(POSITION_LOCKED);
        currentState = STATE_LOCKED;
      }
    }
    else if (command == 'U') {
      // Unlock command received - open door
      if (currentState != STATE_OPEN) {
        doorServo.write(POSITION_OPEN);
        currentState = STATE_OPEN;
      }
    }
  }
}
