import streamlit as st
import torch
import import_ipynb
from PIL import Image
from torchvision import transforms

# import từ notebook
from CIFAR10 import CNN   # nếu file bạn tên khác thì sửa lại

# ===== CONFIG =====
num_classes = 15

classes = [
    # bạn nên lấy đúng từ train_dataset.classes
    # tạm thời ví dụ:
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    # ... (điền đủ 15 class của bạn)
]

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    model = CNN(num_classes)
    model.load_state_dict(torch.load("model_Plant.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ===== PREPROCESS =====
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess(image):
    img = Image.open(image).convert("RGB")
    return transform(img).unsqueeze(0)

# ===== UI =====
st.title("🌿 Plant Disease Classification")

file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

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