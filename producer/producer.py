from kafka import KafkaProducer
import csv
import time
import random
import json

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  
)

topic = 'price_paid_data'

with open('price_paid_records.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)  
    for row in reader:
        producer.send(topic, value=row)
        print(f"Kirim data: {row}")
        time.sleep(random.uniform(0.1, 1.0))  

producer.flush()
print("Producer selesai mengirim data")
