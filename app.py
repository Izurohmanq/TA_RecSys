import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# Load data
data = pd.read_csv("makanan9.csv")

# Drop kolom yang tidak diperlukan
data = data.drop(['id', 'kode', 'sumber', 'gambar', 'satuan'], axis=1)
data['jenis_pangan_encoded'] = pd.factorize(data['jenis_pangan'])[0]

# Ubah nilai yang asalnya ',' menjadi '.'
numeric_cols = ['jenis_pangan_encoded', 'energi_kal', 'protein_gram', 'lemak_gram', 'karbohidrat_gram', 'serat_gram',
                'kalsium_mg', 'fosfor_mg', 'zatbesi_mg', 'natrium_mg', 'kalium_mg', 'vitc_mg']
data[numeric_cols] = data[numeric_cols].replace({',': '.'}, regex=True)
data[numeric_cols] = data[numeric_cols].astype(float)

# Normalisasi data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[numeric_cols])

# Menggunakan Nearest Neighbors
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(data_normalized)

def get_recommendations(food_names, allergy_list):
    # Cari indeks makanan yang cocok dengan nama makanan yang diberikan
    food_indices = [data[data['nama_bahan'] == food_name].index[0] for food_name in food_names]
    
    # Filter makanan berdasarkan alergi
    filtered_data = data.copy()
    for allergy in allergy_list:
        filtered_data = filtered_data[~filtered_data['nama_bahan'].str.contains(allergy, case=False)]
        filtered_data = filtered_data[~filtered_data['jenis_pangan'].str.contains(allergy, case=False)]
        
    # 500 dari buku KIA batas natrium untuk ibu hamil, 900 dari AKG batas untuk ibu hamil 
    filtered_data = filtered_data[(filtered_data['natrium_mg'] < 500) | (filtered_data['retinol_mcg'] < 900)]
    
    # Mencari tetangga terdekat
    distances, indices = model.kneighbors(data_normalized[food_indices])
    
    # Mendapatkan nama dan skor kemiripan makanan
    recommendations = []
    # similarity_scores = []
    for idx_list, dist_list in zip(indices, distances):
        for idx, dist in zip(idx_list, dist_list):
            if idx not in food_indices:
                recommendations.append({
                    'nama_bahan': data.iloc[idx]['nama_bahan'],
                    'jenis_pangan': data.iloc[idx]['jenis_pangan'],
                    'gambar': '',
                    'satuan': '', 
                    'similarity_score': 1 - dist  # Cosine similarity is 1 - cosine distance
                })
    
    recommendations = recommendations[:10]  # Mengambil 10 teratas
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    request_data = request.json
    food_names = request_data.get('food_names')
    allergy_list = request_data.get('allergy_list')
    if not food_names:
        return jsonify({'error': "'food_names' parameter is required"}), 400
    if not allergy_list:
        allergy_list = []

    response = []
    for food_name in food_names:
        recommendations = get_recommendations([food_name], allergy_list)
        response.append({
            "pesan": f"Rekomendasi per 100 gram yang sesuai gizi pada makanan {food_name} adalah",
            "rekomendasi": recommendations
        })

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))
