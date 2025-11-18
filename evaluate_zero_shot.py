import json
from src.chatbot import Chatbot

chatbot = Chatbot()

with open("data/intents_test_only.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)["intents"]

total = 0
correct = 0

print("Real-world Zero-Shot Evaluation – 100 unseen English phrases\n")
for intent in test_data:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        total += 1
        intents_list = chatbot.predict_intent(pattern, None)
        predicted = intents_list[0]["intent"] if intents_list else "None"
        
        if predicted == tag:
            correct += 1
        print(f"{'Correct' if predicted == tag else 'Wrong'}  {pattern[:50]:50} → {predicted} (expected {tag})")

print("\n" + "="*60)
print(f"Zero-Shot Real-World Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
print("="*60)