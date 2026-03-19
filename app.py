from flask import Flask, request, jsonify, render_template
import pickle
import json
import random

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("data.json", "r") as f:
    data = json.load(f)

def get_bot_response(user_text):
    vec = vectorizer.transform([user_text])
    intent = model.predict(vec)[0]

    for item in data:
        if item["intent"] == intent:
            return random.choice(item["responses"])

    return "Sorry, I didn't understand that."

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["message"]
    print("Received:", user_message) 
    bot_reply = get_bot_response(user_message)
    print("Reply:", bot_reply)       
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
