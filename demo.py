# import streamlit as st
# import requests
# from PIL import Image
# import io

# # Flask API endpoint
# FLASK_API_URL = "https://liammatt5-tyre-detection.hf.space/predict"

# # st.title("Tyre Defect Detection")
# st.write("Upload an image of a tyre.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     img_bytes = io.BytesIO()
#     image.save(img_bytes, format="JPEG")

#     if st.button("Predict"):
#         response = requests.post(FLASK_API_URL, files={"file": img_bytes.getvalue()})
        
#         if response.status_code == 200:
#             result = response.json().get("result", "Unknown")
#             st.success(f"Prediction: {result}")
#         else:
#             st.error(f"Error: {response.json().get('error', 'Something went wrong')}")
