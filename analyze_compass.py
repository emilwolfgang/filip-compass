import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytesseract
import re
import os

# --- Tesseract Configuration ---
# If running locally on your Mac, use the Conda dictionary path.
# If running on Streamlit Cloud, it will skip this and use the default Linux path.
mac_tessdata_path = '/opt/anaconda3/share/tessdata/'
if os.path.exists(mac_tessdata_path):
    os.environ['TESSDATA_PREFIX'] = mac_tessdata_path

# --- Extraction Functions ---
def extract_time(img):
    h, w = img.shape[:2]
    # Top 12% of height, Left 35% of width
    x1, y1 = 0, 0
    x2, y2 = int(w * 0.35), int(h * 0.12)
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 100, 255, cv2.THRESH_BINARY_INV)
    
    text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789:')
    match = re.search(r'(\d{1,2}:\d{2})', text)
    return match.group(1) if match else None

def extract_direction(img):
    h, w = img.shape[:2]
    # The large text is generally between 60% and 75% down the screen
    x1, y1 = int(w * 0.1), int(h * 0.60)
    x2, y2 = int(w * 0.9), int(h * 0.75)
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Erode to thicken the thin iOS font
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    text = pytesseract.image_to_string(thresh, config='--psm 7')
    match = re.search(r'^(\d{1,3})', text.strip())
    return int(match.group(1)) if match else None

# --- Main Web App UI ---
def main():
    # Set up the webpage title and layout
    st.set_page_config(page_title="Compass OCR Tracker", layout="centered")
    st.title("🧭 Compass & Wind Tracker")
    st.write("Upload your iOS compass screenshots to instantly generate a timeline.")

    # 1. The File Uploader
    uploaded_files = st.file_uploader("Select Screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} image(s)...")
        results = []

        # 2. Process each uploaded file in memory
        for file in uploaded_files:
            # Convert Streamlit's file bytes into an image OpenCV can read
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                time_str = extract_time(img)
                direction = extract_direction(img)
                
                results.append({
                    "Filename": file.name,
                    "Time": time_str,
                    "Direction": direction
                })

        df = pd.DataFrame(results)

        # 3. Clean and Sort the Data
        df_clean = df.dropna(subset=['Time', 'Direction']).copy()
        
        if df_clean.empty:
            st.error("Could not read Time or Direction from the uploaded images. Please check the screenshots.")
            return

        # Convert to datetime for chronological sorting
        df_clean["Time_DT"] = pd.to_datetime(df_clean["Time"], format="%H:%M")
        df_clean = df_clean.sort_values(by="Time_DT")

        # Display the raw data table on the website
        st.subheader("Extracted Data")
        # Display the clean dataframe (hiding the datetime object column for a cleaner look)
        st.dataframe(df_clean.drop(columns=['Time_DT']), use_container_width=True)

        # 4. Plotting the Graph
        st.subheader("Timeline Graph")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(df_clean["Time_DT"], df_clean["Direction"], marker="o", linestyle="-", color="#1f77b4", linewidth=2)
        
        # Format X-axis as a timeline
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Direction (degrees)")
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Draw the plot in the Streamlit app
        st.pyplot(fig)

        # 5. Provide a CSV Download Button
        csv_data = df_clean.drop(columns=['Time_DT']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="compass_results.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()