# import streamlit as st
# import pandas as pd
# import os

# # Judul aplikasi
# st.set_page_config(page_title="Tabel Dataset Resesi", layout="wide")
# st.title("📊 Tabel Dataset Resesi Ekonomi")

# # Path file CSV
# csv_path = "recession_dataset_1984_2024.csv"

# # Cek apakah file ada
# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path)

#     # Menampilkan dimensi dan preview data
#     st.write(f"Dataset berisi **{df.shape[0]} baris** dan **{df.shape[1]} kolom**.")
#     st.dataframe(df, use_container_width=True)

#     # Optional: Statistik ringkas
#     if st.checkbox("Tampilkan statistik deskriptif"):
#         st.subheader("Statistik Deskriptif")
#         st.write(df.describe())
# else:
#     st.error(f"File `{csv_path}` tidak ditemukan. Pastikan file sudah ada di direktori yang benar.")

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Resesi Ekonomi", layout="wide")

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "1. Dataset Original",
    "2. Dataset Setelah Preprocessing",
    "3. Forecasting ARIMA",
    "4. Klasifikasi (SVM & LogReg)"
])

# Page 1 - Dataset Original
if page == "1. Dataset Original":
    st.title("📂 Dataset Original")
    try:
        df_ori = pd.read_csv("recession_dataset_1984_2024.csv")
        st.write(f"Dataset berisi **{df_ori.shape[0]} baris** dan **{df_ori.shape[1]} kolom**.")
        st.dataframe(df_ori, use_container_width=True)
        if st.checkbox("Tampilkan statistik deskriptif"):
            st.write(df_ori.describe())
    except:
        st.error("File 'dataset_original.csv' tidak ditemukan.")

# Page 2 - Dataset Setelah Preprocessing
elif page == "2. Dataset Setelah Preprocessing":
    st.title("🧪 Dataset Setelah Preprocessing")
    try:
        df_prep = pd.read_csv("df_monthly_transformed.csv")
        st.write(f"Dataset berisi **{df_prep.shape[0]} baris** dan **{df_prep.shape[1]} kolom**.")
        st.dataframe(df_prep, use_container_width=True)
        if st.checkbox("Tampilkan statistik deskriptif"):
            st.write(df_prep.describe())
    except:
        st.error("File 'df_monthly_transformed.csv' tidak ditemukan.")

# Page 3 - Forecasting ARIMA
elif page == "3. Forecasting ARIMA":
    st.title("📈 Hasil Forecasting ARIMA")
    try:
        df_forecast = pd.read_csv("arima_forecast.csv")  # pastikan path benar
        df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
        df_forecast.set_index("Date", inplace=True)

        # st.line_chart(df_forecast)
        st.dataframe(df_forecast, use_container_width=True)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")


# elif page == "4. Klasifikasi (SVM & LogReg)":
#     selected_features = [
#         'Indeks_Kepercayaan_Konsumen', 'Selisih_Obligasi_Korporat_dan_10Y_Treasury',
#         'Izin_Pembangunan_Rumah_Baru', 'Pembangunan_Rumah_Baru_Dimulai',
#         'Selisih_10Y_dan_2Y_Treasury', 'Tingkat_Pengangguran', 'Baa',
#         'Imbal_Hasil_1Y_Treasury', 'Stok_Uang_M2_Real', 'Suku_Bunga_Federal'
#     ]
#     st.title("🔍 Klasifikasi Hasil Forecast (Logistic Regression)")

#     try:
#         import joblib
#         from custom import LogisticRegression


#         with open("logreg_model.pkl", "rb") as f:
#             model = joblib.load(f)

#         df_forecast = pd.read_csv("arima_forecast.csv")
#         df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
#         df_forecast.set_index("Date", inplace=True)

#         st.write("Preview data forecast:")
#         st.write(df_forecast.head())

#         X_forecast = df_forecast[selected_features].values

#         scaler = joblib.load("logres_scaler.pkl")
#         X_forecast_scaled = scaler.transform(X_forecast)

#         pred_target = model.predict(X_forecast_scaled)
#         df_forecast["Prediksi_Resesi"] = pred_target

#         st.subheader("📅 Prediksi Resesi 12 Bulan ke Depan")
#         st.dataframe(df_forecast, use_container_width=True)

#         # Visualisasi fitur dengan line chart
#         st.line_chart(df_forecast[selected_features])

#         # Visualisasi prediksi klasifikasi (0/1) dengan bar chart
#         st.subheader("Distribusi Prediksi Resesi")
#         st.bar_chart(df_forecast["Prediksi_Resesi"].value_counts())

#         df_monthly_transformed = pd.read_csv("df_monthly_transformed.csv")
#         df_historical = df_monthly_transformed.copy()
#         df_historical = df_historical.dropna(subset=['Target'])

#         X = df_historical[selected_features].values
#         y = df_historical['Target'].values

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, test_size=0.2, random_state=42, stratify=y
#         )

#         y_pred = model.predict(X_test)

#         st.subheader("📊 Evaluasi Model Logistic Regression")
#         st.text("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
#         st.text("Confusion Matrix (Visualisasi):")
#         cm = confusion_matrix(y_test, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Recession", "Recession"], yticklabels=["No Recession", "Recession"], ax=ax)
#         ax.set_xlabel("Predicted")
#         ax.set_ylabel("Actual")
#         ax.set_title("Confusion Matrix")
#         st.pyplot(fig)

#         st.text("Classification Report:")
#         st.text(classification_report(y_test, y_pred))

#     except FileNotFoundError as e:
#         st.error(f"File tidak ditemukan: {e.filename}")
#     except Exception as e:
#         st.error(f"Terjadi error saat memuat model atau data: {e}")
#         st.write(e)

elif page == "4. Klasifikasi (SVM & LogReg)":
    selected_features = [
        'Indeks_Kepercayaan_Konsumen', 'Selisih_Obligasi_Korporat_dan_10Y_Treasury',
        'Izin_Pembangunan_Rumah_Baru', 'Pembangunan_Rumah_Baru_Dimulai',
        'Selisih_10Y_dan_2Y_Treasury', 'Tingkat_Pengangguran', 'Baa',
        'Imbal_Hasil_1Y_Treasury', 'Stok_Uang_M2_Real', 'Suku_Bunga_Federal'
    ]
    st.title("🔍 Klasifikasi Hasil Forecast (Logistic Regression)")

    tab1, tab2, tab3 = st.tabs(["🔵 Logistic Regression", "🟣 SVM", "perbandingan"])

    with tab1:
        try:
            import joblib
            from custom import LogisticRegression
            from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split

            # === Load Forecast Data ===
            st.header("📅 Prediksi Resesi 12 Bulan ke Depan")
            df_forecast = pd.read_csv("arima_forecast.csv")
            df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
            df_forecast.set_index("Date", inplace=True)

            st.write("Preview data forecast:")
            st.dataframe(df_forecast.head())

            # === Load Model, Scaler, dan Prediksi ===
            # sesuaikan dengan fiturnya
            X_forecast = df_forecast[selected_features].values

            model = joblib.load("logreg_model1.pkl")
            scaler = joblib.load("logres_scaler1.pkl")
            X_forecast_scaled = scaler.transform(X_forecast)

            pred_target = model.predict(X_forecast_scaled)
            df_forecast["Prediksi_Resesi_Logres"] = pred_target


            # === Visualisasi Forecast ===
            st.subheader("📊 Hasil Prediksi")
            st.dataframe(df_forecast)

            # st.subheader("📈 Grafik Fitur")
            # st.line_chart(df_forecast[selected_features])

            # st.subheader("🔍 Distribusi Prediksi Resesi")
            # st.bar_chart(df_forecast["Prediksi_Resesi_Logres"].value_counts())

            # === Evaluasi Model di Data Historis ===
            st.header("🧪 Evaluasi Model di Data Historis")
            df_historical = pd.read_csv("df_monthly_transformed.csv")
            df_historical = df_historical.dropna(subset=["Target"])

            X = df_historical[selected_features].values
            y = df_historical["Target"].values

            X_scaled = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            y_pred = model.predict(X_test)

            st.subheader("📌 Evaluasi Logistic Regression")
            st.text("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["No Recession", "Recession"],
                        yticklabels=["No Recession", "Recession"],
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

        except FileNotFoundError as e:
            st.error(f"File tidak ditemukan: {e.filename}")
        except Exception as e:
            st.error("Terjadi error saat memuat model atau data.")
            st.exception(e)


    with tab2:

        st.subheader("🟣 Prediksi Resesi dengan SVM Manual")

        try:
            import joblib
            from custom import ManualSVM

            # Load model SVM manual
            svm_model = joblib.load("manual_svm_model.pkl")

            alphas = svm_model["alphas"]
            w = svm_model["w"]
            b = svm_model["b"]
            support_vectors = svm_model["support_vectors"]
            support_labels = svm_model["support_labels"]
            X_train_balanced = svm_model["X_train"]
            y_train_svm = svm_model["y_train"]

            # Load data forecast
            df_forecast = pd.read_csv("arima_forecast.csv")
            df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
            df_forecast.set_index("Date", inplace=True)

            X_forecast = df_forecast[selected_features].values
            scaler = StandardScaler()
            X_forecast_scaled = scaler.fit_transform(X_forecast)

            # Buat ulang objek SVM manual & set atributnya
            model_manual = ManualSVM()
            model_manual.alphas = alphas
            model_manual.w = w
            model_manual.b = b
            model_manual.support_vectors = support_vectors
            model_manual.support_labels = support_labels
            model_manual.X_train = X_train_balanced
            model_manual.y_train = y_train_svm

            # Prediksi
            y_pred_forecast = model_manual.predict(X_forecast_scaled)
            df_forecast["Prediksi_Resesi_SVM"] = y_pred_forecast

            st.subheader("📅 Prediksi Resesi 12 Bulan (SVM)")
            st.dataframe(df_forecast, use_container_width=True)
            # st.bar_chart(df_forecast["Prediksi_Resesi_SVM"].value_counts())

            # Load data historis
            df_monthly_transformed = pd.read_csv("df_monthly_transformed.csv")
            df_historical = df_monthly_transformed.copy()
            df_historical = df_historical.dropna(subset=['Target'])

            X = df_historical[selected_features].values
            y = df_historical['Target'].values
            y_svm = np.where(y == 0, -1, 1)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_svm, test_size=0.2, random_state=42, stratify=y_svm
            )

            y_pred_svm = np.array([model_manual.decision_function(x) for x in X_test])
            y_pred_svm_bin = np.where(y_pred_svm >= 0, 1, 0)
            y_test_bin = np.where(y_test == -1, 0, 1)

            st.subheader("📊 Evaluasi Model SVM Manual")
            st.text("Accuracy: {:.2f}".format(accuracy_score(y_test_bin, y_pred_svm_bin)))

            cm = confusion_matrix(y_test_bin, y_pred_svm_bin)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                        xticklabels=["No Recession", "Recession"],
                        yticklabels=["No Recession", "Recession"],
                        ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix (SVM Manual)")
            st.pyplot(fig)

            st.text("Classification Report:")
            st.text(classification_report(y_test_bin, y_pred_svm_bin))
        


        except FileNotFoundError as e:
            st.error(f"File tidak ditemukan: {e.filename}")
        except Exception as e:
            st.error(f"Terjadi error saat menjalankan SVM Manual: {e}")
            st.write(e)

    with tab3:
        st.subheader("🔁 Perbandingan SVM Manual vs Logistic Regression")

        try:
            import joblib
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            from sklearn.preprocessing import StandardScaler
            from custom import ManualSVM  # pastikan ini sudah didefinisikan sesuai implementasi

            # Load model SVM manual
            svm_model = joblib.load("manual_svm_model.pkl")
            model_manual = ManualSVM()
            model_manual.alphas = svm_model["alphas"]
            model_manual.w = svm_model["w"]
            model_manual.b = svm_model["b"]
            model_manual.support_vectors = svm_model["support_vectors"]
            model_manual.support_labels = svm_model["support_labels"]
            model_manual.X_train = svm_model["X_train"]
            model_manual.y_train = svm_model["y_train"]

            # Load model Logistic Regression
            logreg_model = joblib.load("logreg_model1.pkl")

            # Load historical data
            df_monthly_transformed = pd.read_csv("df_monthly_transformed.csv")
            df_historical = df_monthly_transformed.dropna(subset=['Target'])

            X = df_historical[selected_features].values
            y_logreg = df_historical['Target'].values
            y_svm = np.where(y_logreg == 0, -1, 1)

            # Scaling
            scaler = joblib.load("logres_scaler1.pkl")  # atau scaler SVM kamu sendiri
            X_forecast_scaled = scaler.transform(X_forecast)

            # Split data sekali saja
            X_train, X_test, y_train_logreg, y_test_logreg = train_test_split(
                X_scaled, y_logreg, test_size=0.2, random_state=42, stratify=y_logreg
            )
            y_test_svm = np.where(y_test_logreg == 0, -1, 1)

            # Prediksi SVM Manual
            y_pred_svm = np.array([model_manual.decision_function(x) for x in X_test])
            y_pred_svm_bin = np.where(y_pred_svm >= 0, 1, 0)
            y_test_bin = np.where(y_test_svm == -1, 0, 1)

            # Prediksi Logistic Regression
            y_pred_logreg = logreg_model.predict(X_test)

            # Tampilkan metrik akurasi
            st.markdown("### 📈 Akurasi Model")
            col1, col2 = st.columns(2)
            col1.metric("SVM Manual", f"{accuracy_score(y_test_bin, y_pred_svm_bin):.4f}")
            col2.metric("Logistic Regression", f"{accuracy_score(y_test_logreg, y_pred_logreg):.4f}")

            # Confusion matrix
            st.markdown("### 📊 Confusion Matrix")
            col1, col2 = st.columns(2)

            cm_svm = confusion_matrix(y_test_bin, y_pred_svm_bin)
            fig1, ax1 = plt.subplots()
            sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Purples", cbar=False,
                        xticklabels=["No Recession", "Recession"],
                        yticklabels=["No Recession", "Recession"], ax=ax1)
            ax1.set_title("SVM Manual")
            col1.pyplot(fig1)

            cm_logreg = confusion_matrix(y_test_logreg, y_pred_logreg)
            fig2, ax2 = plt.subplots()
            sns.heatmap(cm_logreg, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["No Recession", "Recession"],
                        yticklabels=["No Recession", "Recession"], ax=ax2)
            ax2.set_title("Logistic Regression")
            col2.pyplot(fig2)


        except FileNotFoundError as e:
            st.error(f"File tidak ditemukan: {e.filename}")
        except Exception as e:
            st.error(f"Terjadi error saat membandingkan model: {e}")
            st.write(e)
