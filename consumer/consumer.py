from kafka import KafkaConsumer
import os
import csv

consumer = KafkaConsumer(
    'price_paid_data',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True
)

os.makedirs("data_batches", exist_ok=True)
batch_size = 100000
buffer = []
batch_id = 1

for msg in consumer:
    buffer.append(msg.value.decode('utf-8').split(','))
    if len(buffer) >= batch_size:
        with open(f'data_batches/batch_{batch_id}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(buffer)
        print(f"Saved batch_{batch_id}.csv")
        batch_id += 1
        buffer = []