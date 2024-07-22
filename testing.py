import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Baca data dari CSV
data = pd.read_csv("makanan12.csv")

# Drop kolom yang tidak diperlukan
data = data.drop(['id', 'kode', 'sumber', 'gambar', 'satuan'], axis=1)
data['jenis_pangan_encoded'] = pd.factorize(data['jenis_pangan'])[0] 

# Ubah nilai yang asalnya ',' menjadi '.'
numeric_cols = [ "jenis_pangan_encoded", "air_gram", "energi_kal", "protein_gram", "lemak_gram", "karbohidrat_gram",
    "serat_gram", "abu_gram", "kalsium_mg", "fosfor_mg", "zatbesi_mg", "natrium_mg",
    "kalium_mg", "tembaga_mg", "seng_mg", "retinol_mcg", "thiamin_mg", "riboflavin_mg",
    "niasin_mg", "vitc_mg", "bdd"
]

data[numeric_cols] = data[numeric_cols].replace({',': '.'}, regex=True)
data[numeric_cols] = data[numeric_cols].astype(float)

# Normalisasi data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[numeric_cols])

# Menggunakan Nearest Neighbors
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(data_normalized)


# Fungsi untuk mendapatkan rekomendasi makanan
def get_recommendations(food_names, allergy_list):
    # Cari indeks makanan yang cocok dengan nama makanan yang diberikan
    food_indices = [data[data['nama_bahan'] == food_name].index[0] for food_name in food_names]
    
    print("Makanan tersebut berada di indeks ke-", food_indices)
    
    # Mencari tetangga terdekat
    distances, indices = model.kneighbors(data_normalized[food_indices])
    
    print(distances)
    print(indices)
    
    # Mendapatkan nama dan skor kemiripan makanan
    recommendations = []
    jarak = []
    for idx_list, dist_list in zip(indices, distances):
        for idx, dist in zip(idx_list, dist_list):
            if idx not in food_indices:
                recommendations.append(data.iloc[idx]['nama_bahan'])
                jarak.append(dist)  # Cosine similarity is 1 - cosine distance
    
    return recommendations[:30], jarak[:30]  # Mengambil 10 teratas

# Contoh penggunaan
allergy_list = ['susu']  # Ganti dengan alergi yang dimiliki user
food_names = ['apel segar']  # Ganti dengan makanan yang dimiliki user
recommendations, jarak = get_recommendations(food_names, allergy_list)
print("Rekomendasi makanan:")
for idx, (food, score) in enumerate(zip(recommendations, jarak)):
    print(f"{idx+1}. {food} (Similarity Score: {score})")
