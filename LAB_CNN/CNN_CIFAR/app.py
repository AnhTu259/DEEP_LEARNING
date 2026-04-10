import streamlit as st
import torch
import import_ipynb
from CIFAR10 import CNN
from utils import preprocess

classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model_CIFAR.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

st.title("CIFAR10 Classifier")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    st.image(file, caption="Input Image", use_column_width=True)

    img = preprocess(file)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    st.success(f"Prediction: {classes[pred.item()]}")
    st.info(f"Confidence: {conf.item()*100:.2f}%")

    # Top 3
    top3_prob, top3_idx = torch.topk(probs, 3)

    st.write("Top 3 Predictions:")
    for i in range(3):
        st.write(f"{classes[top3_idx[0][i]]}: {top3_prob[0][i]*100:.2f}%")