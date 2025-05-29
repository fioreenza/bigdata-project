from kafka import KafkaProducer
import time
import random

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: v.encode('utf-8') 
)

topic = 'price_paid_data'

with open('price_paid_records.csv', 'r') as f:
    for line in f:
        line = line.strip()
        producer.send(topic, value=line)
        print(f"Kirim data: {line}")
        time.sleep(random.uniform(0.1, 1.0))

producer.flush()
print("Producer selesai mengirim data")
