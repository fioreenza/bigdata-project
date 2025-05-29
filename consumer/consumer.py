from kafka import KafkaConsumer
import csv
import os
from datetime import datetime, timedelta

topic = 'price_paid_data'
consumer = KafkaConsumer(
    topic,
    bootstrap_servers='kafka:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: x.decode('utf-8')
)

os.makedirs("data_batches", exist_ok=True)
buffer = []
batch_id = 1
batch_limit = 3
interval = timedelta(minutes=10)
start_time = datetime.now()


for msg in consumer:
    line = msg.value
    print(f"Terima data: {line}")
    buffer.append(line.split(','))

    if datetime.now() - start_time >= interval:
        filename = f"data_batches/batch_{batch_id}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(buffer)
        print(f"Saved {filename} dengan {len(buffer)} record")
        buffer = []
        batch_id += 1
        start_time = datetime.now()

        if batch_id > batch_limit:
            print("Batch limit tercapai. Berhenti consume.")
            break
