import streamlit as st
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import requests
import re
import PyPDF2
import zipfile

# --- Your existing styling and UI setup here (not repeated for brevity) ---

def main():
    st.set_page_config(
        page_title="Bitwave Actions to Form 8949 Converter",
        page_icon="₿",
        layout="wide"
    )
    
    # -- [Include your full previous CSS styling and UI markdown here; skipped for brevity] --

    # Sidebar for configuration
    st.sidebar.markdown("## Configuration")
    st.sidebar.markdown("**Form 8949 Type:**")
    form_type = st.sidebar.selectbox(
        "",
        [
            "Part I - Short-term (Box B) - Basis NOT reported", 
            "Part I - Short-term (Box A) - Basis reported",
            "Part I - Short-term (Box C) - Various situations",
            "Part II - Long-term (Box B) - Basis NOT reported",
            "Part II - Long-term (Box A) - Basis reported",
            "Part II - Long-term (Box C) - Various situations"
        ],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Taxpayer Information:**")
    taxpayer_name = st.sidebar.text_input("Full Name", placeholder="Enter your full name")
    taxpayer_ssn = st.sidebar.text_input("Social Security Number", placeholder="XXX-XX-XXXX")

    st.markdown('### 🗓️ Step 1: Select Tax Year')
    tax_year = st.selectbox(
        "Choose the tax year you're filing for:",
        [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018],
        index=2
    )
    st.info(f"📅 Processing transactions for tax year **{tax_year}**")

    st.markdown('### 📂 Step 2: Upload Bitwave Actions Report')
    uploaded_file = st.file_uploader(
        "Choose your Bitwave actions CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            # Normalize headers: strip whitespace just in case
            df_raw.columns = df_raw.columns.str.strip()

            required_columns = [
                'action', 'asset', 'timestamp', 'lotId',
                'proceeds', 'costBasisRelieved',
                'shortTermGainLoss', 'longTermGainLoss'
            ]

            missing_columns = [col for col in required_columns if col not in df_raw.columns]
            if missing_columns:
                st.error(f"❌ Missing columns: {', '.join(missing_columns)}")
                st.info("Please upload the correct Bitwave actions CSV export.")
                transactions = None
            else:
                transactions = extract_bitwave_transactions(df_raw, tax_year)
                if transactions:
                    st.success(f"🎯 Extracted {len(transactions)} sell transactions for {tax_year}!")
                    # Display summary or other UI as needed (omitted for brevity)
                else:
                    st.error(f"❌ No sell transactions found for {tax_year}.")
                    transactions = None

        except Exception as e:
            st.error(f"Error reading Bitwave file: {e}")
            transactions = None
    else:
        transactions = None

    # Step 3: Output, generation, etc. (omitted for brevity; use your existing logic)

# Function to clean and parse currency
def clean_currency_value(value):
    if pd.isna(value) or value == '' or value == '-':
        return 0.0
    str_val = str(value).strip()
    is_negative = False
    if '(' in str_val and ')' in str_val:
        is_negative = True
        str_val = str_val.replace('(', '').replace(')', '')
    str_val = re.sub(r'[,$\s]', '', str_val)
    try:
        val = float(str_val)
        return -val if is_negative else val
    except:
        return 0.0

def extract_bitwave_transactions(df, target_year):
    lot_map = {}
    buy_transactions = df[df['action'] == 'buy'].copy()

    for _, row in buy_transactions.iterrows():
        lot_id = row.get('lotId')
        if pd.notna(lot_id):
            lot_map[lot_id] = {
                'buy_date': pd.to_datetime(row['timestamp']),
                'asset': row['asset'],
                'cost_basis_acquired': clean_currency_value(row.get('costBasisAcquired', 0))
            }

    sell_transactions = df[df['action'] == 'sell'].copy()
    form8949_transactions = []

    for _, row in sell_transactions.iterrows():
        try:
            sell_date = pd.to_datetime(row['timestamp'])
            if sell_date.year != target_year:
                continue

            lot_id = row.get('lotId')
            lot_info = lot_map.get(lot_id, {})
            buy_date = lot_info.get('buy_date')

            proceeds = clean_currency_value(row.get('proceeds', 0))
            cost_basis = clean_currency_value(row.get('costBasisRelieved', 0))

            if proceeds <= 0 and cost_basis <= 0:
                continue
            
            short_term_gl = clean_currency_value(row.get('shortTermGainLoss', 0))
            long_term_gl = clean_currency_value(row.get('longTermGainLoss', 0))

            calculated_gain_loss = proceeds - cost_basis
            reported_gain_loss = short_term_gl + long_term_gl

            is_short_term = abs(short_term_gl) > 0.01
            is_long_term = abs(long_term_gl) > 0.01

            if buy_date and sell_date:
                holding_days = (sell_date - buy_date).days
                if holding_days <= 365:
                    is_short_term = True
                    is_long_term = False
                else:
                    is_short_term = False
                    is_long_term = True

            transaction = {
                'asset': row['asset'],
                'description': f"{row['asset']} cryptocurrency",
                'date_acquired': buy_date or sell_date,
                'date_sold': sell_date,
                'proceeds': proceeds,
                'cost_basis': cost_basis,
                'gain_loss': calculated_gain_loss,
                'reported_gain_loss': reported_gain_loss,
                'short_term_gain_loss': short_term_gl,
                'long_term_gain_loss': long_term_gl,
                'is_short_term': is_short_term,
                'is_long_term': is_long_term,
                'lot_id': lot_id
            }
            form8949_transactions.append(transaction)
        except Exception as e:
            continue

    return form8949_transactions

# --- Include your other functions (PDF generation, CSV generation, etc.) with same column name fixes ---

if __name__ == "__main__":
    main()
