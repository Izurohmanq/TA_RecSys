import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Baca data dari CSV
data = pd.read_csv("makanan4.csv", sep=";")

# Drop kolom yang tidak diperlukan
numeric_cols = ['air_gram', 'energi_kal', 'protein_gram', 'lemak_gram', 'karbohidrat_gram', 'serat_gram',
                'kalsium_mg', 'fosfor_mg', 'zatbesi_mg', 'natrium_mg', 'kalium_mg', 'tembaga_mg', 'seng_mg', 'vitc_mg']

# Save the other columns for later use in API response
metadata_cols = ['id', 'kode', 'sumber', 'gambar'] + numeric_cols

# Ubah nilai yang asalnya ',' menjadi '.'
data[numeric_cols] = data[numeric_cols].replace({',': '.'}, regex=True)

# Normalisasi data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[numeric_cols])

# Hitung similarity matrix (Cosine Similarity)
cosine_sim = cosine_similarity(data_normalized, data_normalized)

# Fungsi untuk mendapatkan rekomendasi makanan
def get_recommendations(food_names, allergy_list):
    # Cari indeks makanan yang cocok dengan nama makanan yang diberikan
    food_indices = [data[data['nama_bahan'] == food_name].index[0] for food_name in food_names]
    
    # Filter makanan berdasarkan alergi
    filtered_data = data.copy()
    for allergy in allergy_list:
        filtered_data = filtered_data[~filtered_data['nama_bahan'].str.contains(allergy, case=False)]
        filtered_data = filtered_data[~filtered_data['jenis_pangan'].str.contains(allergy, case=False)]
    
    # Hitung similarity antara makanan yang dimiliki user dengan makanan yang tersedia
    recommendations = {}
    for food_name, food_index in zip(food_names, food_indices):
        sim_scores = list(enumerate(cosine_sim[food_index]))
        sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'score'])
        avg_sim_scores = sim_scores_df.groupby('index')['score'].mean().reset_index()
        avg_sim_scores = avg_sim_scores.sort_values(by='score', ascending=False)
        
        # Ambil 10 makanan dengan similarity score tertinggi (kecuali makanan yang dimiliki user)
        top_similar_food_indices = avg_sim_scores['index'].iloc[1:11].tolist()
        top_similar_food_scores = avg_sim_scores['score'].iloc[1:11].tolist()
        
        top_recommendations = data.iloc[top_similar_food_indices]
        top_recommendations['similarity_score'] = top_similar_food_scores
        
        recommendations[food_name] = top_recommendations.to_dict(orient='records')
    
    return recommendations
