import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, rand
from pyspark.ml.feature import (
    StringIndexer, StringIndexerModel, 
    VectorAssembler, 
    BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel
)

# Setup logging biar gampang lihat apa yang lagi terjadi di program
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyRecommenderEngine:
    """
    Kelas buat bikin sistem rekomendasi properti pake PySpark dan Locality Sensitive Hashing (LSH).
    """
    def __init__(self, spark, dataset_path=None, model_dir="models/train_batch3", load_existing=False):
        """
        Inisialisasi engine rekomendasi properti.

        Args:
            spark (SparkSession): Sesi Spark yang aktif.
            dataset_path (str, optional): Lokasi file CSV dataset. Harus diisi kalau ga load model yang udah ada.
            model_dir (str, optional): Folder buat nyimpen atau load model.
            load_existing (bool, optional): Kalau True, load model dari model_dir. Kalau False, bikin model baru.
        """
        self.spark = spark
        self.model_dir = model_dir
        # Kolom kategorikal yang dianggap penting buat cari kesamaan properti
        self.base_categorical_cols = [
            "Property Type", "Old/New", "Duration", "Town/City", 
            "District", "County", "PPDCategory Type", "Record Status - monthly file only"
        ]
        self.active_categorical_cols = [] # Kolom yang beneran dipake, tergantung data yang ada
        self.indexers = {}  # Nyimpen model StringIndexer
        self.lsh_model = None  # Nyimpen model LSH
        self.indexed_data_for_lsh = None  # DataFrame yang punya semua properti dengan fitur buat query LSH
        
        # Tentuin path buat simpen/load komponen
        self.indexed_data_path = os.path.join(self.model_dir, "indexed_property_data.parquet")
        self.lsh_model_path = os.path.join(self.model_dir, "lsh_model")
        self.active_cols_path = os.path.join(self.model_dir, "active_categorical_cols.txt")

        # Kalau folder model belum ada, bikin dulu
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"Folder model dibuat: {self.model_dir}")

        # Kalau load_existing True, load model yang udah ada
        if load_existing:
            self.__load_model_and_data()
        else:
            # Kalau ga load model, harus ada dataset_path buat bikin model baru
            if dataset_path is None:
                raise ValueError("Harus kasih dataset_path kalau load_existing False")
            self.__train_model_and_features(dataset_path)

    def __train_model_and_features(self, dataset_path):
        """
        Load data, latih StringIndexer, VectorAssembler, dan model LSH, terus simpen.
        """
        logger.info(f"Mulai latih model dan siapin fitur dari dataset: {dataset_path}")
        df = self.spark.read.csv(dataset_path, header=True, inferSchema=True)

        # Cek kolom kategorikal mana yang beneran ada di dataset
        self.active_categorical_cols = [c for c in self.base_categorical_cols if c in df.columns]
        missing_cols = set(self.base_categorical_cols) - set(self.active_categorical_cols)
        if missing_cols:
            logger.warning(f"Kolom kategorikal ini ga ada di dataset, diabaikan: {missing_cols}")
        if not self.active_categorical_cols:
            raise ValueError("Ga ada kolom kategorikal yang relevan di dataset. Cek struktur dataset dan base_categorical_cols.")

        # Preproses kolom kategorikal: ubah ke string dan isi null dengan "Unknown"
        for col_name in self.active_categorical_cols:
            df = df.withColumn(col_name, col(col_name).cast("string"))
            df = df.na.fill({col_name: "Unknown"}) # Null di kolom kategorikal diisi "Unknown"

        # Latih dan simpen StringIndexer
        for col_name in self.active_categorical_cols:
            logger.info(f"Indexing kolom kategorikal: {col_name}")
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="keep")
            model_indexer = indexer.fit(df)
            df = model_indexer.transform(df)
            
            indexer_path = os.path.join(self.model_dir, f"{col_name}_indexer")
            model_indexer.write().overwrite().save(indexer_path)
            self.indexers[col_name] = model_indexer
        logger.info("Selesai indexing semua kolom kategorikal.")

        # Simpen daftar kolom kategorikal yang aktif
        with open(self.active_cols_path, "w") as f:
            for col_name in self.active_categorical_cols:
                f.write(f"{col_name}\n")

        # Gabungin fitur ke vektor
        feature_cols_indexed = [col_name + "_index" for col_name in self.active_categorical_cols]
        assembler = VectorAssembler(inputCols=feature_cols_indexed, outputCol="features", handleInvalid="skip") # Skip baris yang punya null di fitur
        df_assembled = assembler.transform(df)

        if "features" not in df_assembled.columns:
            raise RuntimeError("Gagal bikin kolom 'features'. Cek VectorAssembler.")
        
        # Latih model LSH
        logger.info("Latih model BucketedRandomProjectionLSH...")
        brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=2.0, numHashTables=3)
        self.lsh_model = brp.fit(df_assembled)
        self.lsh_model.write().overwrite().save(self.lsh_model_path)
        logger.info(f"Model LSH disimpan ke: {self.lsh_model_path}")

        # Simpen DataFrame yang udah di-transform (termasuk kolom asli buat konteks) buat query LSH
        self.indexed_data_for_lsh = df_assembled
        self.indexed_data_for_lsh.write.mode("overwrite").parquet(self.indexed_data_path)
        logger.info(f"Data properti yang udah diindex disimpan ke: {self.indexed_data_path}")
        logger.info("Latihan model dan persiapan fitur selesai.")

    def __load_model_and_data(self):
        """
        Load StringIndexer, model LSH, dan data fitur yang udah diindex dari disk.
        """
        logger.info(f"Load model dan data dari folder: {self.model_dir}")

        # Load daftar kolom kategorikal aktif
        if not os.path.exists(self.active_cols_path):
            raise FileNotFoundError(f"Daftar kolom kategorikal aktif ga ditemukan: {self.active_cols_path}. Model mungkin ga lengkap.")
        with open(self.active_cols_path, "r") as f:
            self.active_categorical_cols = [line.strip() for line in f if line.strip()]
        
        if not self.active_categorical_cols:
            logger.warning("Ga ada kolom kategorikal aktif yang diload. Rekomendasi mungkin ga jalan seperti yang diharapkan.")

        # Load StringIndexer
        for col_name in self.active_categorical_cols:
            indexer_path = os.path.join(self.model_dir, f"{col_name}_indexer")
            if os.path.exists(indexer_path):
                self.indexers[col_name] = StringIndexerModel.load(indexer_path)
            else:
                raise FileNotFoundError(f"Model indexer untuk '{col_name}' ga ditemukan di {indexer_path}, padahal terdaftar sebagai aktif.")
        logger.info(f"Berhasil load {len(self.indexers)} model StringIndexer.")

        # Load model LSH
        if not os.path.exists(self.lsh_model_path):
            raise FileNotFoundError(f"Model LSH ga ditemukan di: {self.lsh_model_path}")
        self.lsh_model = BucketedRandomProjectionLSHModel.load(self.lsh_model_path)
        logger.info(f"Model LSH diload dari: {self.lsh_model_path}")

        # Load data yang udah diindex buat query LSH
        if not os.path.exists(self.indexed_data_path):
            raise FileNotFoundError(f"File Parquet data properti yang udah diindex ga ditemukan di: {self.indexed_data_path}")
        self.indexed_data_for_lsh = self.spark.read.parquet(self.indexed_data_path)
        logger.info(f"Data properti yang udah diindex diload dari: {self.indexed_data_path}")
        logger.info("Model dan data berhasil diload.")

    def recommend_similar(self, input_property_dict, num_recommendations=5, input_property_id=None, id_column_name="Transaction unique identifier"):
        """
        Rekomendasi properti yang mirip dengan properti input.

        Args:
            input_property_dict (dict): Dictionary dengan kunci nama kolom kategorikal
                                        (contoh: "Property Type", "Town/City") dan nilai string-nya.
            num_recommendations (int, optional): Jumlah properti yang direkomendasikan.
            input_property_id (str, optional): ID unik properti input. Kalau ada, ID ini dikecualikan dari rekomendasi.
            id_column_name (str, optional): Nama kolom ID unik di dataset.

        Returns:
            DataFrame: DataFrame Spark berisi properti yang direkomendasikan, termasuk kolom 'distCol'
                       yang nunjukin tingkat kemiripan (lebih kecil = lebih mirip).
        """
        if self.lsh_model is None or self.indexed_data_for_lsh is None:
            raise ValueError("Model LSH atau data yang udah diindex belum diload/dilatih. Panggil train atau set load_existing=True dulu.")
        if not self.active_categorical_cols:
             raise ValueError("Ga ada kolom kategorikal aktif. Engine mungkin belum diinisialisasi atau dilatih dengan benar.")

        # Bikin DataFrame dari dictionary input
        # Pastikan semua kolom kategorikal aktif ada, isi dengan "Unknown" kalau ga ada di input_dict
        data_for_df_row = {}
        for col_name in self.active_categorical_cols:
            data_for_df_row[col_name] = str(input_property_dict.get(col_name, "Unknown"))
        
        input_df_raw = self.spark.createDataFrame([data_for_df_row])

        # Terapin StringIndexer
        input_df_indexed = input_df_raw
        for col_name in self.active_categorical_cols:
            if col_name in self.indexers:
                input_df_indexed = self.indexers[col_name].transform(input_df_indexed)
            else:
                logger.error(f"Error kritis: Indexer untuk kolom aktif '{col_name}' ga ditemukan di self.indexers.")
                input_df_indexed = input_df_indexed.withColumn(col_name + "_index", lit(0.0)) 

        # Gabungin fitur buat properti query
        feature_cols_indexed = [col_name + "_index" for col_name in self.active_categorical_cols]
        
        # Pastikan semua feature_cols_indexed ada sebelum digabung
        for idx_col in feature_cols_indexed:
            if idx_col not in input_df_indexed.columns:
                 logger.warning(f"Kolom yang udah diindex '{idx_col}' ga ada di query. Ditambah dengan lit(0.0). Ini bisa ngaruh ke kualitas rekomendasi.")
                 input_df_indexed = input_df_indexed.withColumn(idx_col, lit(0.0))

        assembler = VectorAssembler(inputCols=feature_cols_indexed, outputCol="features_query", handleInvalid="keep")
        input_df_assembled = assembler.transform(input_df_indexed)

        if "features_query" not in input_df_assembled.columns or input_df_assembled.isEmpty():
            logger.error("Gagal gabungin fitur buat dictionary properti input atau input-nya kosong.")
            return self.spark.createDataFrame([], self.indexed_data_for_lsh.schema.add("distCol", "double"))

        query_features_row = input_df_assembled.select("features_query").first()
        if not query_features_row:
            logger.error("Ga bisa ambil 'features_query' dari DataFrame yang udah digabung.")
            return self.spark.createDataFrame([], self.indexed_data_for_lsh.schema.add("distCol", "double"))
        
        query_key_vector = query_features_row.features_query

        logger.info(f"Cari {num_recommendations} tetangga terdekat buat properti input...")
        
        # Kalau input_property_id dikasih, ambil satu tetangga ekstra
        # buat jaga-jaga kalau properti input sendiri termasuk yang paling dekat
        effective_num_recommendations = num_recommendations
        if input_property_id and id_column_name in self.indexed_data_for_lsh.columns:
            effective_num_recommendations += 1

        # Cari tetangga terdekat pake LSH
        recommendations_df = self.lsh_model.approxNearestNeighbors(
            dataset=self.indexed_data_for_lsh, 
            key=query_key_vector,               
            numNearestNeighbors=effective_num_recommendations
        )

        # Filter properti input sendiri kalau ID-nya dikasih
        if input_property_id and id_column_name in recommendations_df.columns:
            recommendations_df = recommendations_df.filter(col(id_column_name) != input_property_id)
        
        # Pilih kolom asli supaya hasilnya gampang dibaca, plus kolom jarak
        original_cols_to_show = [c for c in self.indexed_data_for_lsh.columns 
                                 if c not in ["features", "hashes"] and not c.endswith("_index")]
        
        # Kalau original_cols_to_show kosong, pilih kolom minimal
        if not original_cols_to_show:
            logger.warning("Ga bisa tentuin kolom asli buat ditampilin. Default ke ID dan Price kalau ada.")
            original_cols_to_show = []
            if id_column_name in self.indexed_data_for_lsh.columns:
                original_cols_to_show.append(id_column_name)
            if "Price" in self.indexed_data_for_lsh.columns:
                 original_cols_to_show.append("Price")
            if not original_cols_to_show and self.active_categorical_cols:
                 original_cols_to_show.append(self.active_categorical_cols[0])

        # Pastikan kolom 'distCol' dipilih buat sorting dan cek hasil
        final_selection_cols = list(set(original_cols_to_show + ["distCol"]))
        
        # Filter kolom yang beneran ada di recommendations_df
        final_selection_cols = [c for c in final_selection_cols if c in recommendations_df.columns]
        if "distCol" not in final_selection_cols:
            logger.error("distCol ga ada di hasil LSH, ga bisa sortir berdasarkan jarak.")
            return recommendations_df.limit(num_recommendations)

        return recommendations_df.select(final_selection_cols).orderBy("distCol").limit(num_recommendations)


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("PropertyRecommendationDemo") \
        .getOrCreate() # Buat sesi Spark

    # --- Konfigurasi ---
    DATASET_PATH = "data_batches/batch_3.csv" # Pastiin file ini ada
    MODEL_DIR = "models/train_batch3"
    LOAD_EXISTING_MODEL = False # True kalau mau load model, False buat bikin baru
    
    logger.info(f"Inisialisasi PropertyRecommenderEngine. Load existing: {LOAD_EXISTING_MODEL}")
    try:
        recommender_engine = PropertyRecommenderEngine(
            spark,
            dataset_path=DATASET_PATH,
            model_dir=MODEL_DIR,
            load_existing=LOAD_EXISTING_MODEL
        )
        logger.info("PropertyRecommenderEngine berhasil diinisialisasi.")

        # --- Tes Rekomendasi ---
        # Ambil contoh properti buat tes rekomendasi
        if recommender_engine.indexed_data_for_lsh and not recommender_engine.indexed_data_for_lsh.isEmpty():
            logger.info("Coba ambil contoh properti buat tes rekomendasi...")
            sample_property_row = recommender_engine.indexed_data_for_lsh.orderBy(rand()).first()

            if sample_property_row:
                input_property_as_dict = sample_property_row.asDict()
                
                # Bikin dictionary query cuma pake fitur kategorikal asli
                query_dict = {}
                for cat_col in recommender_engine.active_categorical_cols:
                    if cat_col in input_property_as_dict:
                        query_dict[cat_col] = input_property_as_dict[cat_col]
                
                id_col = "Transaction unique identifier"
                sample_id = input_property_as_dict.get(id_col)

                if query_dict:
                    logger.info(f"Ambil rekomendasi buat properti contoh (ID: {sample_id}): {query_dict}")
                    
                    recommendations = recommender_engine.recommend_similar(
                        input_property_dict=query_dict,
                        num_recommendations=5,
                        input_property_id=sample_id,
                        id_column_name=id_col
                    )
                    
                    logger.info(f"--- Top 5 Properti yang Direkomendasikan (kecuali ID: {sample_id}) ---")
                    recommendations.show(truncate=False)
                else:
                    logger.warning("Ga bisa bikin query_dict valid dari properti contoh. Lewatin rekomendasi.")
            else:
                logger.warning("Ga bisa ambil properti contoh dari data yang udah diindex. Lewatin contoh rekomendasi.")
        else:
            logger.warning("indexed_data_for_lsh ga tersedia atau kosong. Lewatin contoh rekomendasi. Pastiin model dilatih atau diload dengan benar.")

    except Exception as e:
        logger.error(f"Ada error: {e}", exc_info=True)
    finally:
        spark.stop()
        logger.info("Sesi Spark dihentikan.")