import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load data
data = pd.read_csv("makanan4.csv", sep=";")
data = data.drop(['id', 'kode', 'sumber', 'gambar'], axis=1)
numeric_cols = ['air_gram', 'energi_kal', 'protein_gram', 'lemak_gram', 'karbohidrat_gram', 'serat_gram', 'kalsium_mg', 'fosfor_mg', 'zatbesi_mg', 'natrium_mg', 'kalium_mg', 'tembaga_mg', 'seng_mg', 'vitc_mg']
data[numeric_cols] = data[numeric_cols].replace({',': '.'}, regex=True)
data[numeric_cols] = data[numeric_cols].astype(float)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data[numeric_cols])
cosine_sim = cosine_similarity(data_normalized, data_normalized)

def get_recommendations(food_name, allergy_list):
    food_index = data[data['nama_bahan'] == food_name].index[0]
    filtered_data = data.copy()
    for allergy in allergy_list:
        filtered_data = filtered_data[~filtered_data['nama_bahan'].str.contains(allergy, case=False)]
        filtered_data = filtered_data[~filtered_data['jenis_pangan'].str.contains(allergy, case=False)]
    sim_scores = list(enumerate(cosine_sim[food_index]))
    sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'score'])
    avg_sim_scores = sim_scores_df.groupby('index')['score'].mean().reset_index()
    avg_sim_scores = avg_sim_scores.sort_values(by='score', ascending=False)
    top_similar_food_indices = avg_sim_scores['index'].iloc[1:11].tolist()
    recommendations = filtered_data.iloc[top_similar_food_indices].copy()
    recommendations['similarity_score'] = avg_sim_scores['score'].iloc[1:11].tolist()
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
        recommendations = get_recommendations(food_name, allergy_list)
        response.append({
            "pesan": f"Rekomendasi yang sesuai gizi makan {food_name} adalah",
            "rekomendasi": recommendations.to_dict(orient='records')
        })

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))
