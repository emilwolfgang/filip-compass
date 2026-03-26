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
mac_tessdata_path = '/opt/anaconda3/share/tessdata/'
if os.path.exists(mac_tessdata_path):
    os.environ['TESSDATA_PREFIX'] = mac_tessdata_path

# --- Extraction Functions ---
def extract_time(img):
    h, w = img.shape[:2]
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
    x1, y1 = int(w * 0.1), int(h * 0.60)
    x2, y2 = int(w * 0.9), int(h * 0.75)
    roi = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    text = pytesseract.image_to_string(thresh, config='--psm 7')
    match = re.search(r'^(\d{1,3})', text.strip())
    return int(match.group(1)) if match else None

# --- Main Web App UI ---
def main():
    st.set_page_config(page_title="Compass OCR Tracker", layout="centered")
    st.title("🧭 Compass & Wind Tracker")
    st.write("Upload your iOS compass screenshots to instantly generate a timeline.")

    uploaded_files = st.file_uploader("Select Screenshots", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        # Create a unique ID based on the uploaded files so we know if they change
        upload_hash = "".join([f.name + str(f.size) for f in uploaded_files])

        # Only run the heavy OCR extraction if these are new files
        if 'last_hash' not in st.session_state or st.session_state['last_hash'] != upload_hash:
            st.info(f"Processing {len(uploaded_files)} image(s)... This may take a moment.")
            results = []

            for file in uploaded_files:
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
            
            # Save the raw data into Streamlit's memory
            st.session_state['raw_data'] = pd.DataFrame(results)
            st.session_state['last_hash'] = upload_hash

        # Load the data from memory
        df = st.session_state['raw_data']

        if df.empty:
            st.error("Could not read any data. Please check the screenshots.")
            return

        # --- The Interactive Editor ---
        st.subheader("✏️ Extracted Data (Editable)")
        st.write("Click any cell to fix a misread number. Select a row and press **Delete** (or click the trash icon) to remove it entirely.")
        
        edited_df = st.data_editor(
            df,
            num_rows="dynamic", # This specific setting enables the add/delete row functionality
            use_container_width=True
        )

        # Clean up the user-edited data before plotting
        df_clean = edited_df.dropna(subset=['Time', 'Direction']).copy()
        
        # Safely convert the edited Time strings back into datetime objects
        df_clean["Time_DT"] = pd.to_datetime(df_clean["Time"], format="%H:%M", errors='coerce')
        df_clean = df_clean.dropna(subset=['Time_DT']).sort_values(by="Time_DT")

        if not df_clean.empty:
            st.subheader("📈 Timeline Graph")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(df_clean["Time_DT"], df_clean["Direction"], marker="o", linestyle="-", color="#1f77b4", linewidth=2)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
            
            ax.set_xlabel("Time")
            ax.set_ylabel("Direction (degrees)")
            ax.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            st.pyplot(fig)

            # CSV Download now uses the newly edited data
            csv_data = df_clean.drop(columns=['Time_DT']).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Edited Results as CSV",
                data=csv_data,
                file_name="edited_compass_results.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
