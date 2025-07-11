import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Ukulele by Yousician - Non-Churn User Clustering", layout="wide")
st.title("🪕 Ukulele by Yousician - Non-Churn User Clustering App")

# === Mapping Label Klaster ===
label_mapping = {
    0: "Consistent Engagers",
    1: "Low Commitment Users",
    2: "High Performing Explorers"
}

# === Pilih Mode Input ===
input_mode = st.radio("Pilih Mode Input:", ["👤 Input Manual (1 Pengguna)", "📁 Upload CSV (Batch Pengguna)"])

# === MODE 1: INPUT MANUAL UNTUK 1 USER ===
if input_mode == "👤 Input Manual (1 Pengguna)":
    st.subheader("Masukkan Data Pengguna")

    # Load model dan scaler
    model_1user = joblib.load("kmeans_nonchurned_model_1user.pkl")
    scaler_1user = joblib.load("scaler_clustering_nc_1user.pkl")
    feature_names_1user = joblib.load("clustering_feature_names_nc1user.pkl")

    # Input manual
    user_input = {}
    for feat in feature_names_1user:
        user_input[feat] = st.number_input(f"{feat}", value=0.0)

    predicted_churn = st.selectbox("Apakah Predicted_Churn user ini?", [0, 1])

    if st.button("🧭 Prediksi Cluster"):
        if predicted_churn == 0:
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler_1user.transform(input_df)
            cluster = model_1user.predict(input_scaled)[0]
            label = label_mapping.get(cluster, f"Cluster {cluster}")
            st.success(f"✅ User ini diprediksi **tidak churn** dan berada pada Cluster: **{label}**")
        else:
            st.info("ℹ️ User ini churn. Tidak dilakukan pemetaan klaster.")

# === MODE 2: UPLOAD CSV UNTUK BANYAK USER ===
else:
    st.subheader("Unggah File CSV")

    # Load model dan scaler
    model_batch = joblib.load("kmeans_nonchurned_batch_model.pkl")
    scaler_batch = joblib.load("scaler_nc_clustering_batch.pkl")
    feature_names_batch = joblib.load("nc_clustering_feature_names_batch.pkl")

    uploaded_file = st.file_uploader("Unggah file CSV dengan fitur yang sesuai", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Preview Data (semua user):", df.head())

            # Validasi kolom
            required_columns = set(feature_names_batch + ["Predicted_Churn"])
            if not required_columns.issubset(df.columns):
                st.error(f"❌ File harus memuat kolom: {required_columns}")
            else:
                df_nonchurn = df[df["Predicted_Churn"] == 0].copy()

                if df_nonchurn.empty:
                    st.warning("⚠️ Tidak ada user dengan Predicted_Churn = 0.")
                else:
                    X = df_nonchurn[feature_names_batch]
                    X_scaled = scaler_batch.transform(X)
                    clusters = model_batch.predict(X_scaled)

                    df_nonchurn['Predicted_Cluster'] = clusters
                    df_nonchurn['Cluster_Label'] = df_nonchurn['Predicted_Cluster'].map(label_mapping)

                    st.success(f"📈 {len(df_nonchurn)} user non-churn berhasil diklasterisasi.")
                    st.dataframe(df_nonchurn)

                    # === Visualisasi: PCA 2D ===
                    st.markdown("### 🔍 Visualisasi PCA (2D)")
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(X_scaled)
                    df_nonchurn['PCA1'] = pca_result[:, 0]
                    df_nonchurn['PCA2'] = pca_result[:, 1]

                    fig_pca, ax_pca = plt.subplots()
                    sns.scatterplot(data=df_nonchurn, x="PCA1", y="PCA2", hue="Cluster_Label", palette="Set2", s=60, ax=ax_pca)
                    ax_pca.set_title("PCA - 2D Projection of Clusters")
                    st.pyplot(fig_pca)

                    # === Visualisasi: Heatmap Korelasi ===
                    st.markdown("### 🌡️ Korelasi Antar Fitur")
                    corr_matrix = df_nonchurn[feature_names_batch].corr()
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
                    ax_corr.set_title("Correlation Heatmap")
                    st.pyplot(fig_corr)

                    # === Visualisasi: Profil Segmentasi ===
                    st.markdown("### 📦 Segment Profile (Boxplot per Fitur)")
                    selected_features = st.multiselect("Pilih fitur yang ingin divisualisasikan:", feature_names_batch, default=feature_names_batch[:3])
                    for feat in selected_features:
                        fig_box, ax_box = plt.subplots()
                        sns.boxplot(data=df_nonchurn, x="Cluster_Label", y=feat, palette="Set2", ax=ax_box)
                        ax_box.set_title(f"{feat} by Cluster")
                        st.pyplot(fig_box)

                    # === Tombol Download CSV ===
                    csv = df_nonchurn.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Hasil Non-Churned Users", csv, "nonchurned_clustered_users.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat membaca file: {e}")
