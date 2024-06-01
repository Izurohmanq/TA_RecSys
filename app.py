from flask import Flask, request, jsonify
from recommendation_model import FoodRecommender

app = Flask(__name__)
recommender = FoodRecommender("makanan4.csv")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    food_names = data.get('food_names', [])
    allergy_list = data.get('allergy_list', [])
    
    recommendations = recommender.get_recommendations(food_names, allergy_list)
    
    result = recommendations.to_dict(orient='records')
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
