from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import streamlit as st

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_toxicity(text):
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=128, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

    return pred, probs[0][pred].item()


st.title("Определение токсичности текста")
st.write("Введите текст, чтобы определить его токсичность:")

user_input = st.text_area("Текст")

if st.button("Предсказать"):
    if user_input != "":
        pred, prob = predict_toxicity(user_input)

        result = "Токсичный" if pred == 1 else "Не токсичный"
        st.write(f"Результат: {result}")
        st.write(f"Вероятность: {prob:.2f}")
    else:
        st.error("Пожалуйста, введите текст")
