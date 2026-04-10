import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# ================= MODEL =================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ================= LOAD =================
@st.cache_resource
def load_all():
    # time series
    model1 = LSTMModel()
    model1.load_state_dict(torch.load("model1.pth", map_location="cpu"))
    model1.eval()

    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")

    # nlp
    with open("vocab.json", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open("idx2word.json", encoding="utf-8") as f:
        idx2word = json.load(f)

    vocab_size = len(word2idx) + 1

    model2 = NextWordLSTM(vocab_size)
    model2.load_state_dict(torch.load("model2.pth", map_location="cpu"))
    model2.eval()

    return model1, x_scaler, y_scaler, model2, word2idx, idx2word


model1, x_scaler, y_scaler, model2, word2idx, idx2word = load_all()

# ================= UI =================
st.title("LSTM Demo ")

menu = st.sidebar.radio("Chọn bài", ["Time Series", "NLP"])

# ================= TIME SERIES =================
if menu == "Time Series":
    st.header("Dự đoán chuỗi sin")

    seq_input = st.text_input("Nhập 10 giá trị", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    if st.button("Predict"):
        try:
            seq = np.array([float(x) for x in seq_input.split(",")])

            # scale
            seq_scaled = x_scaler.transform(seq.reshape(-1,1)).reshape(1, -1, 1)

            x_tensor = torch.tensor(seq_scaled, dtype=torch.float32)

            with torch.no_grad():
                pred = model1(x_tensor)

            pred = y_scaler.inverse_transform(pred.numpy())[0][0]

            st.success(f"Dự đoán: {pred:.4f}")

            # plot
            plt.figure()
            plt.plot(seq, label="Input")
            plt.scatter(len(seq), pred)
            plt.legend()
            st.pyplot(plt)

        except:
            st.error("Input sai format")

# ================= NLP =================
else:
    st.header("Dự đoán từ tiếp theo")

    text = st.text_input("Nhập câu:", "tôi thích ăn")

    if st.button("Predict word"):
        words = text.split()
        seq = [word2idx.get(w, 0) for w in words]
        seq = seq[-3:]

        seq = torch.tensor(seq).unsqueeze(0)

        with torch.no_grad():
            output = model2(seq)
            _, pred = torch.max(output, 1)

        word = idx2word.get(str(pred.item()), "UNK")

        st.success(f"Từ tiếp theo: {word}")