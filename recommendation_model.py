import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class FoodRecommender:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep=";")
        self.data = self.data.drop(['id', 'kode', 'sumber', 'gambar'], axis=1)
        numeric_cols = ['air_gram', 'energi_kal', 'protein_gram', 'lemak_gram', 'karbohidrat_gram', 'serat_gram',
                        'kalsium_mg', 'fosfor_mg', 'zatbesi_mg', 'natrium_mg', 'kalium_mg', 'tembaga_mg', 'vitc_mg']
        self.data[numeric_cols] = self.data[numeric_cols].replace({',': '.'}, regex=True)
        self.scaler = StandardScaler()
        self.data_normalized = self.scaler.fit_transform(self.data[numeric_cols])
        self.cosine_sim = cosine_similarity(self.data_normalized, self.data_normalized)

    def get_recommendations(self, food_names, allergy_list):
        food_indices = [self.data[self.data['nama_bahan'] == food_name].index[0] for food_name in food_names]
        
        filtered_data = self.data.copy()
        for allergy in allergy_list:
            filtered_data = filtered_data[~filtered_data['nama_bahan'].str.contains(allergy, case=False)]
            filtered_data = filtered_data[~filtered_data['jenis_pangan'].str.contains(allergy, case=False)]
        
        sim_scores = []
        for food_index in food_indices:
            sim_scores.extend(list(enumerate(self.cosine_sim[food_index])))
        
        sim_scores_df = pd.DataFrame(sim_scores, columns=['index', 'score'])
        avg_sim_scores = sim_scores_df.groupby('index')['score'].mean().reset_index()
        
        avg_sim_scores = avg_sim_scores.sort_values(by='score', ascending=False)
        
        top_similar_food_indices = avg_sim_scores['index'].iloc[1:11].tolist()
        top_similar_food_scores = avg_sim_scores['score'].iloc[1:11].tolist()
        
        result = self.data.iloc[top_similar_food_indices].copy()
        result['similarity_score'] = top_similar_food_scores
        return result

# Example usage
# recommender = FoodRecommender("makanan4.csv")
# recommendations = recommender.get_recommendations(["ikan asin kering"], ["susu"])
# print(recommendations)
