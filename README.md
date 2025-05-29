# Tugas Big Data

Sistem Big Data yang akan dibuat berfungsi untuk mensimulasikan pemrosesan data stream menggunakan **Kafka** dan **Apache Spark**. Kafka Consumer tidak wajib menggunakan Spark Streaming.

Dikerjakan Oleh:
- Fiorenza Adelia Nalle 5027231053
- Harwinda 5027231079
- Aryasatya Alaauddin 5027231082

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

## Dataset yang digunakan

### Ringkasan Dataset: UK Housing Prices Paid

Dataset ini berisi **data historis semua transaksi properti individual di Inggris dan Wales sejak tahun 1995** yang tercatat dengan harga pasar penuh (full market value). Data ini dirilis oleh **HM Land Registry** dan tersedia secara publik melalui lisensi **Open Government License 3.0**.

### Informasi yang Disediakan

Setiap baris data mewakili satu transaksi properti, dengan informasi sebagai berikut:

* **Transaction Unique Identifier**: ID unik untuk setiap transaksi.
* **Price**: Harga jual properti sesuai akta jual-beli.
* **Date of Transfer**: Tanggal selesainya transaksi.
* **Property Type**:

  * `D` = Rumah Terpisah (Detached)
  * `S` = Rumah Setengah Terpisah (Semi-Detached)
  * `T` = Rumah Berderet (Terraced)
  * `F` = Apartemen/Maisonette (Flats/Maisonettes)
  * `O` = Lainnya
* **Old/New**: Status usia properti

  * `Y` = Properti baru dibangun
  * `N` = Properti lama/eksisting
* **Duration**: Jenis kepemilikan properti

  * `F` = Freehold
  * `L` = Leasehold
* **Town/City**: Kota atau kota kecil tempat properti berada.
* **District** dan **County**: Wilayah administratif properti.
* **PPD Category Type**:

  * `A` = Transaksi normal (dijual ke individu dengan harga pasar penuh)
  * `B` = Transaksi tambahan (contoh: lelang, buy-to-let, atau ke institusi non-pribadi)
* **Record Status - monthly file only**:

  * `A` = Penambahan data baru
  * `C` = Perubahan data
  * `D` = Penghapusan data

> **Catatan**: Alamat lengkap telah disingkat hanya sampai tingkat kota/kabupaten untuk menjaga privasi.

### Sumber Resmi

Dataset ini dipublikasikan oleh [HM Land Registry](https://www.gov.uk/government/organisations/land-registry) dan juga tersedia di [Kaggle](https://www.kaggle.com/datasets/hm-land-registry/uk-housing-prices-paid).

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
    
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 file_training_model.py
    ```
5. Menjalankan API
   ```
    docker exec -it backend bash
   
    pip install flask pyspark
    
    python app.py
   ```
---

## Dokumentasi

![image](https://github.com/user-attachments/assets/800cd275-f15e-4188-8256-3cb3883a02c5)
![WhatsApp Image 2025-05-30 at 00 46 08_bd5a2cdd](https://github.com/user-attachments/assets/2d4d133d-2bb1-4d1d-ae81-03371f7a70bf)
![Screenshot 2025-05-30 021647](https://github.com/user-attachments/assets/50d32ca9-e99a-494a-bcdc-b3b2872bdd33)





   
