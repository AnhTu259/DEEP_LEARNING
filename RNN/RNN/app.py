import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ===== MODEL =====
class RNNModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    model = RNNModel()
    model.load_state_dict(torch.load("model_RNN.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ===== UI =====
st.title("📈 Dự đoán Time Series bằng RNN")

st.write("Nhập 20 timestep, mỗi timestep gồm 3 giá trị (cách nhau bằng dấu phẩy)")

# ví dụ mẫu
default_input = "0.1,0.2,0.3\n" * 20

user_input = st.text_area("Input (20 dòng):", default_input, height=300)

# ===== PREPROCESS =====
def preprocess(text):
    lines = text.strip().split("\n")
    
    if len(lines) != 20:
        raise ValueError("Phải nhập đúng 20 dòng")
    
    data = []
    for line in lines:
        nums = [float(x) for x in line.split(",")]
        if len(nums) != 3:
            raise ValueError("Mỗi dòng phải có 3 số")
        data.append(nums)
    
    return torch.tensor([data], dtype=torch.float32)

# ===== PREDICT =====
if st.button("Dự đoán"):
    try:
        x = preprocess(user_input)

        with torch.no_grad():
            output = model(x)

        st.success(f"📊 Kết quả dự đoán: {output.item():.4f}")

    except Exception as e:
        st.error(f"Lỗi: {e}")