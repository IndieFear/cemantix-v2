import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from gensim.models import KeyedVectors

app = Flask(__name__)
CORS(app)  # Autoriser toutes les origines

def download_model_from_dropbox(url, model_path):
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Modèle téléchargé avec succès.")
    else:
        print(f"Erreur lors du téléchargement (code {response.status_code}).")

# URL Dropbox modifiée pour téléchargement direct
model_url = "https://www.dropbox.com/scl/fi/opt31kzh1jvvtdm0epcn9/frWac_no_postag_no_phrase_500_skip_cut100.bin?rlkey=k5mekwixk76cvsycw7ii159pl&st=6rxtlqyt&dl=1"
model_path = "frWac_no_postag_no_phrase_500_skip_cut100.bin"

download_model_from_dropbox(model_url, model_path)

# Charger le modèle Word2Vec
print("Chargement du modèle...")
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("Modèle chargé avec succès.")

# Calculer les 1000 mots les plus proches une fois au démarrage
target_word = "animal"
top_1000_words = model.most_similar(target_word, topn=1000)


@app.route('/similarity', methods=['POST'])
def similarity():
    """
    Endpoint pour calculer la similarité entre deux mots.
    JSON attendu : { "word1": "mot1", "word2": "mot2" }
    """
    data = request.get_json()
    word1 = data.get('word1')
    word2 = data.get('word2')

    try:
        # Calculer la similarité entre les deux mots
        sim = model.similarity(word1, word2)
        position = next((index + 1 for index, (word, _) in enumerate(top_1000_words) if word == word1), None)

        return jsonify({'similarity': float(sim), 'position': position})
    except KeyError:
        return jsonify({'error': 'Un ou plusieurs mots non trouvés dans le modèle.'}), 404


if __name__ == '__main__':
    # Lancement de l'application Flask sur le port dynamique
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT', 5000))
