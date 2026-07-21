import json

from confluent_kafka import Consumer

conf = {
    "bootstrap.servers": "broker:29092",
    "group.id": "prediction-monitoring",
    "auto.offset.reset": "earliest",
}

consumer = Consumer(conf)

consumer.subscribe(["recsys_predictions"])

while True:

    msg = consumer.poll(1.0)

    if msg is None:
        continue

    if msg.error():
        print(msg.error())
        continue

    data = json.loads(msg.value().decode("utf-8"))

    print(data)
