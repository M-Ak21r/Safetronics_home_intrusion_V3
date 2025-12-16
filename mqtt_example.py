"""
Example MQTT Client for Level 2 Security Microservice

This script demonstrates how to:
1. Subscribe to theft/suspicious activity events
2. Send LOCKDOWN commands to the microservice

Requirements:
    pip install paho-mqtt

Usage:
    # Listen for events
    python mqtt_example.py --mode listen

    # Send LOCKDOWN command
    python mqtt_example.py --mode lockdown

    # Both (listen and send command after 5 seconds)
    python mqtt_example.py --mode both
"""

import argparse
import json
import time
import paho.mqtt.client as mqtt


MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_EVENTS = "sentinel/level2/events"
TOPIC_COMMANDS = "sentinel/commands"


def on_connect(client, userdata, flags, rc):
    """Callback when client connects to broker."""
    if rc == 0:
        print(f"✓ Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_EVENTS)
        print(f"✓ Subscribed to {TOPIC_EVENTS}")
    else:
        print(f"✗ Connection failed with code {rc}")


def on_message(client, userdata, msg):
    """Callback when message is received."""
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        
        # Pretty print the event
        print("\n" + "=" * 60)
        print(f"🚨 EVENT RECEIVED: {payload.get('type', 'UNKNOWN')}")
        print("=" * 60)
        print(f"Timestamp:   {payload.get('timestamp', 'N/A')}")
        print(f"Suspect ID:  {payload.get('suspect_id', 'N/A')}")
        print(f"Status:      {payload.get('status', 'N/A')}")
        
        metadata = payload.get('metadata', {})
        if metadata:
            print(f"Metadata:")
            for key, value in metadata.items():
                print(f"  - {key}: {value}")
        
        print("=" * 60 + "\n")
        
    except json.JSONDecodeError:
        print(f"Raw message: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"Error processing message: {e}")


def listen_mode():
    """Listen for events from the microservice."""
    print("Starting event listener...")
    print(f"Listening on topic: {TOPIC_EVENTS}")
    print("Press Ctrl+C to exit\n")
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.disconnect()
    except Exception as e:
        print(f"Error: {e}")


def send_lockdown():
    """Send LOCKDOWN command to the microservice."""
    print("Sending LOCKDOWN command...")
    
    client = mqtt.Client()
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # Option 1: JSON format
        command = {"command": "LOCKDOWN"}
        json_payload = json.dumps(command)
        result = client.publish(TOPIC_COMMANDS, json_payload, qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"✓ LOCKDOWN command sent to {TOPIC_COMMANDS}")
        else:
            print(f"✗ Failed to send command: {result.rc}")
        
        time.sleep(1)  # Give time for message to be sent
        client.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")


def both_mode():
    """Listen for events and send LOCKDOWN after 5 seconds."""
    print("Starting both modes...")
    print("Will send LOCKDOWN command in 5 seconds...\n")
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # Wait 5 seconds
        time.sleep(5)
        
        # Send LOCKDOWN
        print("\nSending LOCKDOWN command...")
        command = {"command": "LOCKDOWN"}
        json_payload = json.dumps(command)
        result = client.publish(TOPIC_COMMANDS, json_payload, qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"✓ LOCKDOWN command sent")
        else:
            print(f"✗ Failed to send command")
        
        print("\nContinuing to listen for events...")
        print("Press Ctrl+C to exit\n")
        
        # Continue listening
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="MQTT Client Example for Level 2 Security Microservice"
    )
    parser.add_argument(
        "--mode",
        choices=["listen", "lockdown", "both"],
        default="listen",
        help="Operation mode: listen for events, send lockdown, or both"
    )
    parser.add_argument(
        "--broker",
        default="localhost",
        help="MQTT broker address (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)"
    )
    
    args = parser.parse_args()
    
    # Update global broker settings
    global MQTT_BROKER, MQTT_PORT
    MQTT_BROKER = args.broker
    MQTT_PORT = args.port
    
    print("=" * 60)
    print("Level 2 Security Microservice - MQTT Client Example")
    print("=" * 60)
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Mode: {args.mode}")
    print("=" * 60 + "\n")
    
    if args.mode == "listen":
        listen_mode()
    elif args.mode == "lockdown":
        send_lockdown()
    elif args.mode == "both":
        both_mode()


if __name__ == "__main__":
    main()
