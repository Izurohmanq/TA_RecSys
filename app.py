# app.py
import os
from flask import Flask, request, jsonify
from recommendation_model import get_recommendations

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        food_names = data['food_names']
        allergy_list = data['allergy_list']
        
        recommendations = get_recommendations(food_names, allergy_list)
        
        response = []
        for food_name, recs in recommendations.items():
            response.append({
                "pesan": f"rekomendasi yang sesuai gizi makan {food_name} adalah",
                "rekomendasi": recs
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))
