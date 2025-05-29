# Tugas Big Data

Sistem Big Data yang akan dibuat berfungsi untuk mensimulasikan pemrosesan data stream menggunakan **Kafka** dan **Apache Spark**. Kafka Consumer tidak wajib menggunakan Spark Streaming.

---

## Alur Sistem

1. **Dataset**
   - Terdapat sebuah file dataset yang akan dibaca secara sekuensial oleh Kafka Producer.
   - **TIDAK DIPERBOLEHKAN** menggunakan dataset yang sudah dipakai pada tugas-tugas sebelumnya.

2. **Kafka Producer**
   - Mengirimkan data per baris ke Kafka Server seolah-olah melakukan streaming.
   - Proses pengiriman data dilakukan dengan menambahkan jeda/sleep secara random agar data tidak langsung dikirim secara sekaligus.

3. **Kafka Consumer**
   - Membaca data dari Kafka Server.
   - Menyimpan data yang diterima dalam bentuk batch, dengan batch berdasarkan:
     - Jumlah data yang diterima, atau
     - Rentang waktu proses (window).
   - Hasilnya berupa beberapa file dataset sesuai batch yang ditentukan.

4. **Spark Script**
   - Melakukan training model sesuai data yang masuk.
   - Diharapkan ada beberapa model yang dihasilkan, sesuai batch/data yang diproses.
   - Contoh skema model:
     - Model 1: data 5 menit pertama atau 500.000 data pertama.
     - Model 2: data 5 menit kedua atau 500.000 data kedua.
     - Model 3: data 5 menit ketiga atau 500.000 data ketiga.
   - Alternatif skema model:
     - Model 1: 1/3 data pertama.
     - Model 2: 1/3 data pertama + 1/3 data kedua.
     - Model 3: 1/3 data pertama + 1/3 data kedua + 1/3 data ketiga (semua data).
   - Model yang dihasilkan akan digunakan dalam API.
   - Buat endpoint API sesuai jumlah model yang ada.

5. **API**
   - User akan request ke API.
   - API memberikan respon sesuai request, misalnya:
     - Jika request rekomendasi, input: rating user, output: daftar rekomendasi.
     - Jika model kasus clustering, output: cluster tempat data input user berada.
   - Minimal jumlah endpoint API = jumlah anggota kelompok.
     - Contoh: jika 3 anggota, buat minimal 3 endpoint API dengan fungsi berbeda.

---

## Cara Menjalankan Sistem

1. Membuat topic kafka
    ```
    docker exec -it kafka bash
    
    kafka-topics --create \
        --topic price_paid_data \
        --bootstrap-server localhost:9092 \
        --partitions 1 \
        --replication-factor 1
    ```
2. Menjalankan kafka producer
   ```
   docker exec -it producer bash

   pip install kafka-python
    
   python producer.py
   ```
3. Menjalankan kafka consumer
    ```
    docker exec -it consumer bash

    pip install kafka-python
    
    python consumer.py
    ```
4. Menjalankan Spark Training Script
   ```
    docker exec -it train bash
    
    pip install numpy pandas
    
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 train_lr.py
    ```
5. Menjalankan API
   ```
    docker exec -it backend bash
   
    pip install flask pyspark
    
    python app.py
   ```



   
