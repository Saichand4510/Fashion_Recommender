import streamlit as st
import requests


API_URL = "https://fashion-recommender-3044.onrender.com/recommend"  # change this after deployment

st.set_page_config(page_title="Fashion Recommender", layout="centered")

st.title("👗 Fashion Color Recommender")


user_query = st.text_input("Enter your requirement",
                          placeholder="e.g., I need outfit for a wedding")

uploaded_file = st.file_uploader("Upload clothing image", type=["jpg", "png", "jpeg","webp"])


if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)


if st.button("Get Recommendation"):

    if not user_query:
        st.warning("Please enter your query")
    else:
        with st.spinner("Processing..."):
            try:
                files = {}
                data = {"user_query": user_query}

                if uploaded_file:
                    files["image"] = (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )

        
                response = requests.post(API_URL, data=data, files=files)

                result = response.json()

                if result["status"] == "success":

                    if result["colors"]:
                        st.subheader("Detected Colors")

                        cols = st.columns(len(result["colors"]))
                        for i, color in enumerate(result["colors"]):
                            with cols[i]:
                                st.color_picker(
                                    label=f"{color}",
                                    value=color,
                                    disabled=True
                                )
 ############# in local uncomment this for suiting the backend of segmentation approach and comment above for loop in local .
                        # for i, c in enumerate(result["colors"]):
                        #     with cols[i]:
                        #         st.color_picker(
                        #             label=f"{c['hex']}",
                        #             value=c["hex"],
                        #             disabled=True
                        #         )  
                  

             
                    st.subheader("Recommendation")
                    st.write(result["recommendation"])

                else:
                    st.error(result["message"])

            except Exception as e:
                st.error(f"Error: {str(e)}")
