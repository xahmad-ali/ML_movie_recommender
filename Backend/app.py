from flask import Flask, request, jsonify
from Model import movie_recommender  # Import your recommender function from model.py
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# API route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()  # Get the JSON data from the frontend
    movie_name = data['movie_name']  # Extract the movie name

    try:
        recommendations = movie_recommender(movie_name)
        # val= json.dumps(recommendations)
        # print(val)
        # Return the recommendations as JSON
        return {"recommended_movies": recommendations}
    except:
        return jsonify({"error": "Movie not found!"}), 404



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
