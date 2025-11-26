import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Clustering Kebiasaan Membaca Mahasiswa",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# JUDUL APLIKASI
# =============================================
st.title("📚 Clustering Kebiasaan Membaca Buku pada Mahasiswa")
st.markdown("""
**Sistem ini menggunakan metode Clustering (K-Means) untuk mengelompokkan mahasiswa 
berdasarkan kebiasaan membaca buku mereka.**
""")

# =============================================
# GENERATE DATASET
# =============================================
@st.cache_data
def generate_reading_data():
    np.random.seed(42)
    n_samples = 250
    
    data = {
        'usia': np.random.randint(18, 25, n_samples),
        'semester': np.random.randint(1, 8, n_samples),
        'buku_per_bulan': np.random.poisson(3, n_samples),
        'jam_baca_per_minggu': np.random.exponential(8, n_samples),
        'ebook_vs_fisik': np.random.beta(2, 3, n_samples),
        'genre_fiksi': np.random.beta(2, 2, n_samples),
        'genre_non_fiksi': np.random.beta(2, 2, n_samples),
        'perpustakaan_freq': np.random.poisson(2, n_samples),
        'beli_buku_per_bulan': np.random.poisson(1, n_samples),
        'budaya_baca_skor': np.random.randint(1, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Normalisasi
    df['ebook_vs_fisik'] = df['ebook_vs_fisik'] * 100
    df['genre_fiksi'] = df['genre_fiksi'] * 100
    df['genre_non_fiksi'] = df['genre_non_fiksi'] * 100
    
    # Buat pola realistis
    df.loc[df['buku_per_bulan'] > 5, 'jam_baca_per_minggu'] *= 1.5
    df.loc[df['budaya_baca_skor'] > 7, 'buku_per_bulan'] += 2
    df.loc[df['semester'] > 4, 'genre_non_fiksi'] *= 1.2
    
    return df

# =============================================
# LOAD DATA
# =============================================
df = generate_reading_data()

# Deskripsi fitur
feature_descriptions = {
    'usia': 'Usia Mahasiswa (tahun)',
    'semester': 'Semester Saat Ini',
    'buku_per_bulan': 'Jumlah Buku Dibaca per Bulan',
    'jam_baca_per_minggu': 'Jam Membaca per Minggu',
    'ebook_vs_fisik': 'Persentase Baca E-book vs Fisik (%)',
    'genre_fiksi': 'Persentase Baca Genre Fiksi (%)',
    'genre_non_fiksi': 'Persentase Baca Genre Non-Fiksi (%)',
    'perpustakaan_freq': 'Frekuensi Kunjung Perpustakaan per Bulan',
    'beli_buku_per_bulan': 'Jumlah Buku Dibeli per Bulan',
    'budaya_baca_skor': 'Skor Budaya Baca (1-10)'
}

# =============================================
# SIDEBAR NAVIGASI
# =============================================
st.sidebar.title("📚 Navigasi Sistem")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Pilih Menu:",
    ["🏠 Beranda", "📊 Eksplorasi Data", "🔍 Clustering Analysis", "🎯 Profil Cluster", "📈 Prediksi Cluster", "ℹ️ Tentang"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Sistem Pendukung Keputusan**
Metode: K-Means Clustering
Dataset: Kebiasaan Membaca Mahasiswa
""")

# =============================================
# HALAMAN BERANDA
# =============================================
if menu == "🏠 Beranda":
    st.header("🎯 Selamat Datang di Sistem Clustering Kebiasaan Membaca")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📖 Tentang Sistem Ini
        
        Sistem ini menganalisis kebiasaan membaca buku mahasiswa menggunakan **algoritma K-Means Clustering** 
        untuk mengidentifikasi pola dan kelompok mahasiswa berdasarkan:
        
        ### 📋 Variabel yang Dianalisis:
        - **📊 Demografi**: Usia, Semester
        - **⏱️ Intensitas Membaca**: Buku/bulan, Jam/minggu
        - **📖 Preferensi**: E-book vs Fisik, Genre Fiksi/Non-fiksi
        - **🏛️ Perilaku**: Frekuensi perpustakaan, Pembelian buku
        - **🌟 Budaya Baca**: Skor penilaian diri
        
        ### 🚀 Cara Menggunakan:
        1. **📊 Eksplorasi Data**: Lihat distribusi dan statistik dataset
        2. **🔍 Clustering Analysis**: Temukan pola kelompok mahasiswa
        3. **🎯 Profil Cluster**: Pahami karakteristik setiap kelompok
        4. **📈 Prediksi**: Masukkan data baru untuk prediksi cluster
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400", 
                caption="Kebiasaan Membaca Mahasiswa", use_column_width=True)
    
    # Quick stats
    st.subheader("📈 Statistik Cepat Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Mahasiswa", len(df))
    with col2:
        st.metric("📚 Rata-rata Buku/Bulan", f"{df['buku_per_bulan'].mean():.1f}")
    with col3:
        st.metric("⏰ Rata-rata Jam/Minggu", f"{df['jam_baca_per_minggu'].mean():.1f}")
    with col4:
        st.metric("⭐ Rata-rata Skor Budaya", f"{df['budaya_baca_skor'].mean():.1f}")

# =============================================
# HALAMAN EKSPLORASI DATA
# =============================================
elif menu == "📊 Eksplorasi Data":
    st.header("📊 Eksplorasi Dataset Kebiasaan Membaca")
    
    # Tampilkan dataset
    st.subheader("📋 Data Sample (10 Mahasiswa Pertama)")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Distribusi variabel
    st.subheader("📊 Distribusi Variabel")
    
    variables = st.multiselect(
        "Pilih variabel untuk dilihat distribusinya:",
        options=df.columns,
        default=['buku_per_bulan', 'jam_baca_per_minggu', 'budaya_baca_skor']
    )
    
    if variables:
        fig, axes = plt.subplots(1, len(variables), figsize=(5*len(variables), 4))
        if len(variables) == 1:
            axes = [axes]
        
        for i, var in enumerate(variables):
            axes[i].hist(df[var], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'Distribusi {feature_descriptions[var]}')
            axes[i].set_xlabel(feature_descriptions[var])
            axes[i].set_ylabel('Frekuensi')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("🔥 Heatmap Korelasi Antar Variabel")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', ax=ax, square=True)
    ax.set_title('Korelasi Antar Variabel Kebiasaan Membaca')
    st.pyplot(fig)

# =============================================
# HALAMAN CLUSTERING ANALYSIS
# =============================================
elif menu == "🔍 Clustering Analysis":
    st.header("🔍 Analisis Clustering dengan K-Means")
    
    # Pilih fitur untuk clustering
    st.subheader("1. 🎯 Pemilihan Fitur untuk Clustering")
    
    default_features = ['buku_per_bulan', 'jam_baca_per_minggu', 'budaya_baca_skor', 'perpustakaan_freq']
    selected_features = st.multiselect(
        "Pilih fitur untuk clustering:",
        options=df.columns.tolist(),
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Pilih minimal 2 fitur untuk clustering.")
        st.stop()
    
    # Preprocessing
    X = df[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal k
    st.subheader("2. 📊 Penentuan Jumlah Cluster Optimal")
    
    max_k = st.slider("Maksimal jumlah cluster untuk diuji:", 2, 8, 6)
    
    # Calculate metrics
    wcss = []
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if silhouette_scores:
            optimal_k = np.argmax(silhouette_scores) + 2
            st.metric("🎯 Cluster Optimal Disarankan", optimal_k)
        else:
            optimal_k = 3
            st.metric("🎯 Cluster Optimal Disarankan", optimal_k)
    
    with col2:
        if silhouette_scores:
            best_score = max(silhouette_scores)
            st.metric("⭐ Silhouette Score Terbaik", f"{best_score:.3f}")
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    ax1.plot(range(2, max_k + 1), wcss, 'bo-', alpha=0.7, linewidth=2, markersize=8)
    ax1.set_xlabel('Jumlah Cluster (k)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title('Metode Elbow untuk Menentukan k Optimal')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    if silhouette_scores:
        ax2.plot(range(2, max_k + 1), silhouette_scores, 'ro-', alpha=0.7, linewidth=2, markersize=8)
        ax2.set_xlabel('Jumlah Cluster (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score untuk Setiap k')
        ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Perform clustering
    st.subheader("3. 🎨 Hasil Clustering")
    
    n_clusters = st.slider("Pilih jumlah cluster untuk analisis:", 2, max_k, optimal_k)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    df_clustered['cluster'] = 'Cluster ' + df_clustered['cluster'].astype(str)
    
    # Visualize clusters
    if len(selected_features) >= 2:
        st.subheader("📈 Visualisasi Cluster")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("Pilih fitur untuk sumbu X:", selected_features, index=0)
        with col2:
            y_feature = st.selectbox("Pilih fitur untuk sumbu Y:", selected_features, index=1)
        
        fig = px.scatter(df_clustered, x=x_feature, y=y_feature, color='cluster',
                        title=f'Visualisasi Cluster: {feature_descriptions[x_feature]} vs {feature_descriptions[y_feature]}',
                        labels={x_feature: feature_descriptions[x_feature],
                               y_feature: feature_descriptions[y_feature]},
                        hover_data=selected_features,
                        color_discrete_sequence=px.colors.qualitative.Set1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster sizes
    st.subheader("📊 Distribusi Jumlah Mahasiswa per Cluster")
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                    title='Persentase Mahasiswa per Cluster',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(cluster_counts.reset_index().rename(
            columns={'index': 'Cluster', 'cluster': 'Jumlah Mahasiswa'}
        ))

# =============================================
# HALAMAN PROFIL CLUSTER
# =============================================
elif menu == "🎯 Profil Cluster":
    st.header("🎯 Profil dan Karakteristik Cluster")
    
    # Pastikan clustering sudah dilakukan
    if 'cluster' not in df.columns:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu di menu 'Clustering Analysis'.")
        st.stop()
    
    # Cluster profiles
    st.subheader("📋 Profil Rata-rata Setiap Cluster")
    
    cluster_profiles = df.groupby('cluster').mean()
    st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Visualisasi perbandingan cluster
    st.subheader("📊 Perbandingan Karakteristik Cluster")
    
    features_to_compare = st.multiselect(
        "Pilih fitur untuk dibandingkan antar cluster:",
        options=df.select_dtypes(include=[np.number]).columns.tolist(),
        default=['buku_per_bulan', 'jam_baca_per_minggu', 'budaya_baca_skor', 'perpustakaan_freq']
    )
    
    if features_to_compare:
        # Bar chart comparison
        fig = go.Figure()
        
        for feature in features_to_compare:
            fig.add_trace(go.Bar(
                name=feature_descriptions[feature],
                x=cluster_profiles.index,
                y=cluster_profiles[feature],
                text=cluster_profiles[feature].round(2),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Perbandingan Rata-rata Fitur per Cluster',
            xaxis_title='Cluster',
            yaxis_title='Nilai Rata-rata',
            barmode='group',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretasi cluster
    st.subheader("🔍 Interpretasi Cluster")
    
    interpretation_data = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        
        avg_books = cluster_data['buku_per_bulan'].mean()
        avg_hours = cluster_data['jam_baca_per_minggu'].mean()
        avg_score = cluster_data['budaya_baca_skor'].mean()
        avg_library = cluster_data['perpustakaan_freq'].mean()
        
        # Tentukan tipe pembaca
        if avg_books > df['buku_per_bulan'].mean() and avg_hours > df['jam_baca_per_minggu'].mean():
            reader_type = "📚 PEMBACA AKTIF"
            description = "Mahasiswa dengan intensitas membaca tinggi"
            color = "🟢"
        elif avg_books < df['buku_per_bulan'].mean() and avg_hours < df['jam_baca_per_minggu'].mean():
            reader_type = "😴 PEMBACA PASIF"
            description = "Mahasiswa dengan intensitas membaca rendah"
            color = "🔴"
        else:
            reader_type = "⚡ PEMBACA SEDANG"
            description = "Mahasiswa dengan intensitas membaca menengah"
            color = "🟡"
        
        interpretation_data.append({
            'cluster': cluster,
            'type': reader_type,
            'color': color,
            'books': avg_books,
            'hours': avg_hours,
            'score': avg_score,
            'library': avg_library,
            'description': description
        })
    
    # Tampilkan interpretasi dalam columns
    cols = st.columns(len(interpretation_data))
    
    for idx, data in enumerate(interpretation_data):
        with cols[idx]:
            st.markdown(f"### {data['color']} {data['cluster']}")
            st.markdown(f"**{data['type']}**")
            st.markdown(f"*{data['description']}*")
            
            st.metric("📚 Buku/Bulan", f"{data['books']:.1f}")
            st.metric("⏰ Jam/Minggu", f"{data['hours']:.1f}")
            st.metric("⭐ Skor Budaya", f"{data['score']:.1f}")
            st.metric("🏛️ Kunjung Perpus", f"{data['library']:.1f}")

# =============================================
# HALAMAN PREDIKSI CLUSTER
# =============================================
elif menu == "📈 Prediksi Cluster":
    st.header("📈 Prediksi Cluster untuk Data Baru")
    
    st.markdown("""
    Masukkan data mahasiswa baru untuk memprediksi cluster mana yang paling cocok:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Demografi")
        usia = st.slider("Usia", 18, 24, 20)
        semester = st.slider("Semester", 1, 8, 4)
        budaya_baca_skor = st.slider("Skor Budaya Baca", 1, 10, 6)
    
    with col2:
        st.subheader("📖 Kebiasaan Membaca")
        buku_per_bulan = st.slider("Buku per Bulan", 0, 10, 3)
        jam_baca_per_minggu = st.slider("Jam Baca per Minggu", 0.0, 20.0, 8.0, 0.5)
        perpustakaan_freq = st.slider("Frekuensi Perpustakaan per Bulan", 0, 10, 2)
    
    # Prepare input data
    input_data = pd.DataFrame({
        'usia': [usia],
        'semester': [semester],
        'buku_per_bulan': [buku_per_bulan],
        'jam_baca_per_minggu': [jam_baca_per_minggu],
        'perpustakaan_freq': [perpustakaan_freq],
        'budaya_baca_skor': [budaya_baca_skor]
    })
    
    if st.button("🔮 Prediksi Cluster", type="primary"):
        # Gunakan fitur yang sama dengan clustering sebelumnya
        features_for_clustering = ['buku_per_bulan', 'jam_baca_per_minggu', 'budaya_baca_skor', 'perpustakaan_freq']
        X_full = df[features_for_clustering]
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X_full)
        
        # Re-train dengan jumlah cluster yang sama
        n_clusters = len(df['cluster'].unique()) if 'cluster' in df.columns else 3
        kmeans_full = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_full.fit(X_full_scaled)
        
        # Predict for new data
        input_scaled = scaler_full.transform(input_data[features_for_clustering])
        predicted_cluster = kmeans_full.predict(input_scaled)[0]
        cluster_probs = kmeans_full.transform(input_scaled)
        
        st.success(f"🎯 Mahasiswa ini termasuk dalam **Cluster {predicted_cluster}**")
        
        # Tampilkan confidence
        min_dist = np.min(cluster_probs)
        confidence = 1 / (1 + min_dist)  # Simple confidence measure
        st.metric("🎯 Tingkat Kepercayaan", f"{confidence*100:.1f}%")
        
        # Berikan rekomendasi
        st.subheader("💡 Rekomendasi:")
        
        recommendations = {
            0: "**Tingkatkan frekuensi membaca** dengan mulai dari genre yang disukai. Coba baca 15 menit setiap hari.",
            1: "**Pertahankan kebiasaan baik!** Coba eksplor genre baru untuk variasi.",
            2: "**Coba variasi** antara buku fisik dan e-book untuk pengalaman berbeda.",
            3: "**Manfaatkan perpustakaan kampus** untuk akses buku yang lebih luas dan gratis."
        }
        
        rec_key = predicted_cluster % len(recommendations)
        st.info(recommendations[rec_key])
        
        # Tampilkan karakteristik cluster
        if 'cluster' in df.columns:
            cluster_data = df[df['cluster'] == f'Cluster {predicted_cluster}']
            st.subheader(f"📊 Karakteristik Cluster {predicted_cluster}:")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📚 Rata-rata Buku/Bulan", f"{cluster_data['buku_per_bulan'].mean():.1f}")
            with col2:
                st.metric("⏰ Rata-rata Jam/Minggu", f"{cluster_data['jam_baca_per_minggu'].mean():.1f}")
            with col3:
                st.metric("⭐ Rata-rata Skor Budaya", f"{cluster_data['budaya_baca_skor'].mean():.1f}")
            with col4:
                st.metric("🏛️ Rata-rata Kunjung Perpus", f"{cluster_data['perpustakaan_freq'].mean():.1f}")

# =============================================
# HALAMAN TENTANG
# =============================================
elif menu == "ℹ️ Tentang":
    st.header("ℹ️ Tentang Sistem Ini")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📚 Sistem Clustering Kebiasaan Membaca Mahasiswa
        
        **Sistem ini menggunakan algoritma K-Means Clustering** untuk menganalisis dan mengelompokkan 
        mahasiswa berdasarkan kebiasaan membaca buku mereka.
        
        ### 🎯 Tujuan Sistem:
        1. **🔍 Identifikasi Pola**: Menemukan pola kebiasaan membaca di kalangan mahasiswa
        2. **📊 Segmentasi**: Mengelompokkan mahasiswa berdasarkan karakteristik membaca
        3. **💡 Rekomendasi**: Memberikan insight untuk pengembangan budaya baca
        4. **🔮 Prediksi**: Memprediksi kelompok untuk mahasiswa baru
        
        ### 🔧 Teknologi yang Digunakan:
        - **🤖 Algoritma**: K-Means Clustering
        - **⚙️ Preprocessing**: StandardScaler
        - **📈 Visualisasi**: Plotly, Matplotlib, Seaborn
        - **🌐 Framework**: Streamlit
        - **📊 Metode Evaluasi**: Elbow Method, Silhouette Score
        
        ### 📋 Spesifikasi Teknis:
        - **10 variabel** analisis
        - **250 sample data** mahasiswa
        - **Analisis multidimensi** untuk pattern recognition
        - **Visualisasi interaktif** dan real-time
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1507842217343-583bb7270b66?w=400", 
                caption="Perpustakaan Digital", use_column_width=True)
        
        st.info("""
        ### 👨‍💻 Developer Info
        **Dibuat untuk:** Tugas Sistem Pendukung Keputusan  
        **Metode:** Clustering (K-Means)  
        **Status:** Production Ready ✅
        """)
    
    st.success("**🚀 Sistem siap digunakan untuk analisis clustering kebiasaan membaca mahasiswa!**")

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Sistem Pendukung Keputusan - Clustering Kebiasaan Membaca | "
    "Metode: K-Means Clustering | Dibuat menggunakan Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
