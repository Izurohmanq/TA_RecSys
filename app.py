import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# Load data
data = pd.read_csv("makanan15.csv")

# Drop kolom yang tidak diperlukan
data = data.drop(['id', 'kode', 'sumber', 'satuan'], axis=1)
data['jenis_pangan_encoded'] = pd.factorize(data['jenis_pangan'])[0]

# Ubah nilai yang asalnya ',' menjadi '.'
numeric_cols = ["jenis_pangan_encoded", "air_gram", "energi_kal", "protein_gram", "lemak_gram", "karbohidrat_gram",
    "serat_gram", "abu_gram", "kalsium_mg", "fosfor_mg", "zatbesi_mg", "natrium_mg",
    "kalium_mg", "tembaga_mg", "seng_mg", "retinol_mcg", "thiamin_mg", "riboflavin_mg",
    "niasin_mg", "vitc_mg", "bdd"]
# numeric_cols = [ "jenis_pangan_encoded", "air_gram", "energi_kal", "protein_gram", "lemak_gram", "karbohidrat_gram",
#     "serat_gram", "natrium_mg", "kalium_mg", "bdd"
# ]
data[numeric_cols] = data[numeric_cols].replace({',': '.'}, regex=True)
data[numeric_cols] = data[numeric_cols].astype(float)

# Normalisasi data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data[numeric_cols])

# Menggunakan Nearest Neighbors
model = NearestNeighbors(n_neighbors=4, metric='cosine')
model.fit(data_normalized)


def nutrition_need(umur, tb, bb, aktifitas, kondisi, waktu_makan):
    aktivitas_factor = {
        "Bed Rest": 1.1,
        "bergerak terbatas": 1.2,
        "bisa jalan": 1.3
    }

    if aktifitas not in aktivitas_factor:
        return False

    aktifitas = aktivitas_factor[aktifitas]

    # Calculate BMR and macronutrient needs
    BMR_perempuan = 655 + (9.6 * bb) + (1.8 * tb) - (4.7 * umur)
    total_kalori = BMR_perempuan * aktifitas
    karbohidrat = (0.6 * total_kalori) / 4
    lemak = (0.25 * total_kalori) / 9
    protein = (0.15 * total_kalori) / 4

    batas_atas = {
        'energi_kal': total_kalori,
        'karbohidrat_gram': karbohidrat,
        'lemak_gram': lemak,
        'protein_gram': protein,
        'serat_gram': 0,
        'air_gram': 0,
        'vitc_mg': 0,
        'retinol_mcg': 0,
        'kalsium_mg': 0,
        'fosfor_mg': 0,
        'zatbesi_mg': 0,
        'kalium_mg': 0,
        'natrium_mg': 0,
        'tembaga_mg': 0
    }

    age_based_limits = [
        (11, 12, {'serat_gram': 27, 'air_gram': 1850, 'vitc_mg': 50, 'retinol_mcg': 600, 'kalsium_mg': 1200,
         'fosfor_mg': 1250, 'zatbesi_mg': 8, 'kalium_mg': 4400, 'natrium_mg': 1400, 'tembaga_mg': 700}),
        (13, 15, {'serat_gram': 27, 'air_gram': 2100, 'vitc_mg': 65, 'retinol_mcg': 600, 'kalsium_mg': 1200,
         'fosfor_mg': 1250, 'zatbesi_mg': 15, 'kalium_mg': 4800, 'natrium_mg': 1500, 'tembaga_mg': 795}),
        (16, 18, {'serat_gram': 29, 'air_gram': 2150, 'vitc_mg': 75, 'retinol_mcg': 600, 'kalsium_mg': 1200,
         'fosfor_mg': 1250, 'zatbesi_mg': 15, 'kalium_mg': 5000, 'natrium_mg': 1600, 'tembaga_mg': 890}),
        (19, 29, {'serat_gram': 32, 'air_gram': 2350, 'vitc_mg': 75, 'retinol_mcg': 600, 'kalsium_mg': 1000,
         'fosfor_mg': 700, 'zatbesi_mg': 18, 'kalium_mg': 4700, 'natrium_mg': 1500, 'tembaga_mg': 900}),
        (30, 49, {'serat_gram': 30, 'air_gram': 2350, 'vitc_mg': 75, 'retinol_mcg': 600, 'kalsium_mg': 1000,
         'fosfor_mg': 700, 'zatbesi_mg': 18, 'kalium_mg': 4700, 'natrium_mg': 1500, 'tembaga_mg': 900}),
        (50, 64, {'serat_gram': 25, 'air_gram': 2350, 'vitc_mg': 75, 'retinol_mcg': 600, 'kalsium_mg': 1200,
         'fosfor_mg': 700, 'zatbesi_mg': 8, 'kalium_mg': 4700, 'natrium_mg': 1400, 'tembaga_mg': 900}),
        (65, 80, {'serat_gram': 22, 'air_gram': 1550, 'vitc_mg': 75, 'retinol_mcg': 600, 'kalsium_mg': 1200,
         'fosfor_mg': 700, 'zatbesi_mg': 8, 'kalium_mg': 4700, 'natrium_mg': 1200, 'tembaga_mg': 900}),
        (80, float('inf'), {'serat_gram': 20, 'air_gram': 1400, 'vitc_mg': 75, 'retinol_mcg': 600, 'kalsium_mg': 1200,
         'fosfor_mg': 700, 'zatbesi_mg': 8, 'kalium_mg': 4700, 'natrium_mg': 1000, 'tembaga_mg': 900})
    ]

    for min_age, max_age, limits in age_based_limits:
        if min_age <= umur <= max_age:
            batas_atas.update(limits)
            break

    kondisi_based_adjustments = {
        'hamil_trim_1': {'energi_kal': 180, 'karbohidrat_gram': 25, 'lemak_gram': 2.3, 'protein_gram': 1, 'serat_gram': 3, 'air_gram': 300, 'vitc_mg': 10, 'retinol_mcg': 300, 'kalsium_mg': 200, 'tembaga_mg': 100},
        'hamil_trim_2': {'energi_kal': 300, 'karbohidrat_gram': 40, 'lemak_gram': 2.3, 'protein_gram': 10, 'serat_gram': 4, 'air_gram': 300, 'vitc_mg': 10, 'retinol_mcg': 300, 'kalsium_mg': 200, 'zatbesi_mg': 9, 'tembaga_mg': 100},
        'hamil_trim_3': {'energi_kal': 300, 'karbohidrat_gram': 40, 'lemak_gram': 2.3, 'protein_gram': 30, 'serat_gram': 4, 'air_gram': 300, 'vitc_mg': 10, 'retinol_mcg': 300, 'kalsium_mg': 200, 'zatbesi_mg': 9, 'tembaga_mg': 100},
        'menyusui_6_awal': {'energi_kal': 330, 'karbohidrat_gram': 45, 'lemak_gram': 2.2, 'protein_gram': 20, 'serat_gram': 5, 'air_gram': 800, 'vitc_mg': 45, 'retinol_mcg': 350, 'kalsium_mg': 200, 'zatbesi_mg': 9, 'kalium_mg': 400, 'tembaga_mg': 100},
        'menyusui_6_kedua': {'energi_kal': 400, 'karbohidrat_gram': 55, 'lemak_gram': 2.4, 'protein_gram': 15, 'serat_gram': 5, 'air_gram': 700, 'vitc_mg': 25, 'retinol_mcg': 350, 'kalsium_mg': 200, 'zatbesi_mg': 9, 'kalium_mg': 400, 'tembaga_mg': 100}
    }

    if kondisi in kondisi_based_adjustments:
        adjustments = kondisi_based_adjustments[kondisi]
        for key, value in adjustments.items():
            batas_atas[key] += value

    # Membagi batas atas dengan waktu makan
    for key in batas_atas:
        batas_atas[key] /= waktu_makan

    return batas_atas


def get_recommendations(nutrition_limit, food_names):
    recommendations = {}
    for food_name in food_names:
        all_distances, indices = model.kneighbors(data_normalized)
        food_data = data[data['nama_bahan'] == food_name]

        if food_data.empty:
            continue

        recommendations[food_name] = {
            "within_limits": [],
            "exceeding_limits": []
        }

        food_index = food_data.index[0]
        neighbors = indices[food_index]

        for neighbor_index in neighbors:
            neighbor_data = data.iloc[neighbor_index].to_dict()
            within_limits = neighbor_data['natrium_mg'] <= nutrition_limit['natrium_mg']
            akg = (neighbor_data['energi_kal'] /
                   nutrition_limit['energi_kal']) * 100
            # Menambahkan nilai AKG pada data makanan
            neighbor_data['akg'] = akg

            if within_limits:
                recommendations[food_name]["within_limits"].append(
                    neighbor_data)
            else:
                recommendations[food_name]["exceeding_limits"].append(
                    neighbor_data)
    return recommendations


@app.route('/nutrition', methods=['POST'])
def nutrition_endpoint():
    data = request.get_json()
    umur = data['umur']
    tb = data['tb']
    bb = data['bb']
    aktifitas = data['aktifitas']
    kondisi = data['kondisi']
    waktu_makan = data['waktu_makan']
    food_names = data['food_names']

    limits = nutrition_need(umur, tb, bb, aktifitas, kondisi, waktu_makan)

    if not limits:
        return jsonify({"error": "Invalid activity level"}), 400

    recommendations = get_recommendations(limits, food_names)

    output = []
    output.append({
        "pesan_kebutuhan": "Berikut adalah kebutuhan gizimu",
        "kebutuhan_gizi": limits
    })
    for food_name in food_names:
        if food_name in recommendations:
            within_limits = recommendations[food_name]["within_limits"]
            exceeding_limits = recommendations[food_name]["exceeding_limits"]
            if within_limits:
                output.append({
                    "pesan_rekomendasi": f"Rekomendasi per 100 gram yang sesuai gizi pada makanan {food_name} adalah",
                    "rekomendasi": within_limits
                })
            if exceeding_limits:
                output.append({
                    "pesan_hindar": f"Makanan yang sebaiknya dikurangi untuk dikonsumsi karena melebihi batas kebutuhan gizi yang sesuai gizi pada makanan {food_name} adalah",
                    "rekomendasi": exceeding_limits
                })

    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5003)))
