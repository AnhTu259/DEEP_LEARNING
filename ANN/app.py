import streamlit as st
import torch
import torch.nn as nn

# ===== Model =====
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load model
model = ANN()
model.load_state_dict(torch.load("model_ann.pth", map_location=torch.device('cpu')))
model.eval()

# ===== UI =====
st.title("🎯 ANN Classification Demo")

st.write("Nhập tọa độ điểm (x, y)")

x = st.number_input("X", value=0.0)
y = st.number_input("Y", value=0.0)

if st.button("Predict"):
    input_tensor = torch.tensor([[x, y]], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)

    result = int(output.item() > 0.5)

    if result == 0:
        st.success("👉 Class 0 (trong vòng tròn)")
    else:
        st.error("👉 Class 1 (vành ngoài)")