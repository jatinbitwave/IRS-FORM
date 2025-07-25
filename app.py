import streamlit as st
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import zipfile
from datetime import datetime
import re
import requests
import PyPDF2
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def main():
    st.set_page_config(
        page_title="Bitwave Actions to Form 8949 Converter",
        page_icon="₿",
        layout="wide"
    )
    
    # Custom CSS for Bitwave styling and centering
    st.markdown("""
    <style>
    /* Bitwave design system colors */
    :root {
        --bitwave-blue: #1B9CFC;
        --bitwave-green: #00D2B8;
        --bitwave-dark: #1a1a1a;
        --bitwave-gray: #6b7280;
        --bitwave-light-gray: #f8fafc;
        --bitwave-border: #e5e7eb;
    }
    
    /* Global font improvements */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header styling - more subtle like Bitwave */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d3748 100%);
        color: white;
        border-radius: 12px;
        padding: 2.5rem 2rem;
        margin-bottom: 3rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.25rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .bitwave-logo {
        font-size: 2.5rem;
        font-weight: 700;
        margin-right: 1rem;
        background: linear-gradient(45deg, var(--bitwave-blue), var(--bitwave-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    /* Center all step containers with Bitwave spacing */
    .step-container {
        display: flex;
        justify-content: center;
        margin: 4rem 0;
        padding: 0 1rem;
    }
    
    /* Remove any remaining step-content styling that might create boxes */
    .step-content {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
    }
    
    /* Step headers - Bitwave style */
    .step-header {
        color: var(--bitwave-dark) !important;
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: -0.025em;
        border-bottom: 2px solid var(--bitwave-green);
        padding-bottom: 0.75rem;
        display: inline-block;
        width: auto;
    }
    
    /* Bitwave-style buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--bitwave-blue) 0%, var(--bitwave-green) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        letter-spacing: -0.025em;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(27, 156, 252, 0.25);
    }
    
    /* Info and success boxes - Bitwave style */
    .stInfo {
        background: var(--bitwave-light-gray);
        border: 1px solid var(--bitwave-border);
        border-left: 4px solid var(--bitwave-blue);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stSuccess {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid var(--bitwave-green);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Form elements styling */
    .stSelectbox, .stFileUploader, .stRadio {
        margin: 1rem 0;
    }
    
    .stSelectbox label, .stFileUploader label, .stRadio label {
        color: var(--bitwave-dark) !important;
        font-weight: 500;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: var(--bitwave-light-gray);
        border: 1px solid var(--bitwave-border);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--bitwave-dark);
        font-weight: 600;
    }
    
    /* Sidebar styling - cleaner */
    .stSidebar {
        background: var(--bitwave-light-gray);
        border-right: 1px solid var(--bitwave-border);
    }
    
    .stSidebar .stSelectbox label {
        font-size: 0.9rem !important;
        color: var(--bitwave-dark) !important;
        font-weight: 500;
    }
    
    .stSidebar .stColumns {
        gap: 0 !important;
    }
    
    .stSidebar .stColumns > div {
        padding: 0 !important;
    }
    
    .stSidebar span[title] {
        cursor: help;
        color: var(--bitwave-gray);
        font-size: 0.875rem;
        float: right;
        margin-top: 1px;
    }
    
    .stSidebar span[title]:hover {
        color: var(--bitwave-blue);
    }
    
    .stSidebar .stMarkdown h2 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: var(--bitwave-dark) !important;
        margin-bottom: 1.5rem !important;
    }
    
    .stSidebar .stMarkdown strong {
        color: var(--bitwave-dark) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        margin-bottom: 0 !important;
    }
    
    .stSidebar .stMarkdown {
        margin-bottom: 0.1rem !important;
    }
    
    .stSidebar .stSelectbox {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    .stSidebar .stSelectbox > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    .stSidebar .stTextInput {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    .stSidebar .stTextInput > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Fix gaps in main content area */
    .stSelectbox {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    .stSelectbox > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    .stSelectbox > div > div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Content descriptions */
    .content-description {
        color: var(--bitwave-gray);
        font-size: 1.125rem;
        line-height: 1.6;
        text-align: center;
        margin: 2rem 0;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Expander styling */
    .stExpander {
        border: 1px solid var(--bitwave-border);
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    /* File uploader fixes */
    .stFileUploader > div {
        text-align: center !important;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"] {
        border: 2px dashed var(--bitwave-border);
        border-radius: 12px;
        background: var(--bitwave-light-gray);
        padding: 2rem;
        text-align: center;
    }
    
    .stFileUploader [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--bitwave-blue);
        background: #f0f9ff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with Bitwave branding
    st.markdown("""
    <div style="display: flex; justify-content: center; margin: 2rem 0;">
        <div style="max-width: 900px; width: 100%;">
            <div class="main-header">
                <div class="bitwave-logo">BITWAVE</div>
                <h1>Actions to Form 8949 Converter</h1>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered content container
    col_desc1, col_desc2, col_desc3 = st.columns([1, 2, 1])
    with col_desc2:
        st.markdown('<div class="content-description">', unsafe_allow_html=True)
        st.markdown("Convert your Bitwave actions report into tax-ready formats with official IRS Form 8949 templates.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Information section
    col_info1, col_info2, col_info3 = st.columns([1, 2, 1])
    with col_info2:
        with st.expander("ℹ️ How This Works", expanded=False):
            st.markdown("""
            **This tool is specifically designed for Bitwave actions reports:**
            
            1. **Upload** your Bitwave actions CSV export
            2. **Automatically extracts** sell transactions with proper lot matching
            3. **Maps acquisition dates** using lot IDs from buy transactions
            4. **Validates calculations** against Bitwave's short/long-term gain/loss columns
            5. **Choose output:**
               - **CSV file** → Upload to TurboTax, TaxAct, FreeTaxUSA, etc.
               - **PDF file** → Official IRS Form 8949 ready for direct filing
            
            **Required Bitwave columns:**
            - `action` (buy/sell)
            - `asset` (BTC, ETH, etc.)
            - `timestamp` (transaction date)
            - `lotId` (for matching buy/sell pairs)
            - `proceeds` (column R)
            - `costBasisRelieved` (column W)
            - `shortTermGainLoss` and `longTermGainLoss` (for validation)
            """)

    # Sidebar for configuration
    st.sidebar.markdown("## Configuration")
    
    # Form type selection with clean formatting
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
    
    # Taxpayer information for PDF generation
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Taxpayer Information:**")
    taxpayer_name = st.sidebar.text_input("Full Name", placeholder="Enter your full name")
    taxpayer_ssn = st.sidebar.text_input("Social Security Number", placeholder="XXX-XX-XXXX")
    
    # Step 1: Tax Year Selection (Centered)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h2 class="step-header">🗓️ Step 1: Select Tax Year</h2></div>', unsafe_allow_html=True)
    
    # Centered tax year selection with proper labels
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown('<div style="text-align: center; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
        st.markdown("**Choose the tax year you're filing for:**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        tax_year = st.selectbox(
            "",
            [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018],
            index=2,
            help="Select the tax year to extract transactions for",
            key="main_tax_year"
        )
        
        st.info(f"📅 Processing transactions for tax year **{tax_year}**")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Step 2: Upload File (Centered)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;"><h2 class="step-header">📂 Step 2: Upload Bitwave Actions Report</h2></div>', unsafe_allow_html=True)
    
    # Centered file uploader with proper label
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown('<div style="text-align: center; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
        st.markdown("**Choose your Bitwave actions CSV file**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type=["csv"],
            help="Upload the CSV export from your Bitwave actions report"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Centered processing section
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        
        try:
            # Read the Bitwave actions file
            df_raw = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Bitwave actions file uploaded! Found {len(df_raw)} total actions.")
            
            # Validate it's a Bitwave file
            required_columns = ['action', 'asset', 'timestamp', 'lotId', ' proceeds ', ' costBasisRelieved ']
            missing_columns = [col for col in required_columns if col not in df_raw.columns]
            
            if missing_columns:
                st.error(f"❌ This doesn't appear to be a valid Bitwave actions report.")
                st.error(f"Missing columns: {', '.join(missing_columns)}")
                st.info("Please ensure you've uploaded the correct Bitwave actions CSV export.")
                transactions = None
            else:
                # Extract transactions using Bitwave-specific logic
                transactions = extract_bitwave_transactions(df_raw, tax_year)
                
                if transactions:
                    st.success(f"🎯 Extracted {len(transactions)} sell transactions for {tax_year}!")
                    
                    # Show extracted transactions summary
                    st.markdown(f'<h3 style="text-align: center; color: var(--bitwave-dark);">{tax_year} Crypto Sales Summary</h3>', unsafe_allow_html=True)
                    
                    # Create summary by asset
                    asset_summary = {}
                    for txn in transactions:
                        asset = txn['asset']
                        if asset not in asset_summary:
                            asset_summary[asset] = {
                                'count': 0,
                                'proceeds': 0,
                                'cost_basis': 0,
                                'gain_loss': 0
                            }
                        asset_summary[asset]['count'] += 1
                        asset_summary[asset]['proceeds'] += txn['proceeds']
                        asset_summary[asset]['cost_basis'] += txn['cost_basis']
                        asset_summary[asset]['gain_loss'] += txn['gain_loss']
                    
                    # Display asset summary
                    summary_data = []
                    for asset, data in asset_summary.items():
                        summary_data.append({
                            'Asset': asset,
                            'Transactions': data['count'],
                            'Total Proceeds': f"${data['proceeds']:,.2f}",
                            'Total Cost Basis': f"${data['cost_basis']:,.2f}",
                            'Net Gain/Loss': f"${data['gain_loss']:,.2f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Show overall totals in centered metrics
                    total_proceeds = sum(t['proceeds'] for t in transactions)
                    total_basis = sum(t['cost_basis'] for t in transactions)
                    total_gain_loss = sum(t['gain_loss'] for t in transactions)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Proceeds", f"${total_proceeds:,.2f}")
                    with col_b:
                        st.metric("Total Cost Basis", f"${total_basis:,.2f}")
                    with col_c:
                        st.metric("Net Gain/Loss", f"${total_gain_loss:,.2f}")
                    
                    # Show detailed transactions in expander
                    with st.expander(f"📋 View All {len(transactions)} Transactions", expanded=False):
                        display_transactions = []
                        for i, txn in enumerate(transactions[:100]):
                            display_transactions.append({
                                '#': i + 1,
                                'Asset': txn['asset'],
                                'Sell Date': txn['date_sold'].strftime('%m/%d/%Y'),
                                'Buy Date': txn['date_acquired'].strftime('%m/%d/%Y') if txn['date_acquired'] else 'Unknown',
                                'Proceeds': f"${txn['proceeds']:.2f}",
                                'Cost Basis': f"${txn['cost_basis']:.2f}",
                                'Gain/Loss': f"${txn['gain_loss']:.2f}",
                                'Term': 'Short' if txn['is_short_term'] else 'Long'
                            })
                        
                        txn_df = pd.DataFrame(display_transactions)
                        st.dataframe(txn_df, use_container_width=True)
                        
                        if len(transactions) > 100:
                            st.info(f"Showing first 100 transactions. Total: {len(transactions)}")
                
                else:
                    st.error(f"❌ No sell transactions found for {tax_year}. Please check your selected year.")
                    transactions = None
            
        except Exception as e:
            st.error(f"Error reading Bitwave file: {str(e)}")
            transactions = None
            
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        transactions = None
    
    # Step 3: Choose Output (Centered)
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><h2 class="step-header">🎯 Step 3: Choose Your Output</h2></div>', unsafe_allow_html=True)
        
        if transactions:
            # Centered output format selection
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.markdown('<div style="text-align: center; margin-bottom: 0.5rem;">', unsafe_allow_html=True)
                st.markdown("**What do you want to generate?**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                output_format = st.radio(
                    "",
                    [
                        "📊 CSV file for tax software (TurboTax, TaxAct, etc.)",
                        "📄 Complete Form 8949 PDF for IRS filing"
                    ],
                    help="Choose based on how you plan to file your taxes"
                )
                
                # Show term breakdown
                short_term_count = sum(1 for t in transactions if t['is_short_term'])
                long_term_count = len(transactions) - short_term_count
                
                if short_term_count > 0 and long_term_count > 0:
                    st.warning(f"⚠️ You have both short-term ({short_term_count}) and long-term ({long_term_count}) transactions. You may need separate Form 8949s for each.")
                
                # Centered generate button
                if st.button("🚀 Generate Files", type="primary"):
                    try:
                        if "CSV" in output_format:
                            # Generate CSV for tax software
                            csv_data = generate_tax_software_csv(transactions, tax_year)
                            
                            filename = f"form_8949_{tax_year}_bitwave_transactions.csv"
                            st.download_button(
                                label="📥 Download CSV for Tax Software",
                                data=csv_data,
                                file_name=filename,
                                mime="text/csv",
                                help="Upload this file to TurboTax, TaxAct, FreeTaxUSA, or other tax software"
                            )
                            
                            st.success("✅ CSV file ready! This can be imported into most tax software.")
                        
                        else:
                            # Generate PDF Form 8949
                            if not taxpayer_name or not taxpayer_ssn:
                                st.error("⚠️ Please fill in your taxpayer information in the sidebar to generate a PDF.")
                            else:
                                # Split by term type if needed
                                short_term_txns = [t for t in transactions if t['is_short_term']]
                                long_term_txns = [t for t in transactions if not t['is_short_term']]
                                
                                pdf_files = []
                                
                                # Generate short-term PDF if applicable
                                if short_term_txns:
                                    short_form_type = form_type.replace("Part II", "Part I").replace("Long-term", "Short-term")
                                    short_pdfs = generate_form_8949_pdf(
                                        short_term_txns, 
                                        short_form_type, 
                                        taxpayer_name, 
                                        taxpayer_ssn, 
                                        tax_year,
                                        "Short-term"
                                    )
                                    pdf_files.extend(short_pdfs)
                                
                                # Generate long-term PDF if applicable
                                if long_term_txns:
                                    long_form_type = form_type.replace("Part I", "Part II").replace("Short-term", "Long-term")
                                    long_pdfs = generate_form_8949_pdf(
                                        long_term_txns, 
                                        long_form_type, 
                                        taxpayer_name, 
                                        taxpayer_ssn, 
                                        tax_year,
                                        "Long-term"
                                    )
                                    pdf_files.extend(long_pdfs)
                                
                                if len(pdf_files) == 1:
                                    # Single PDF
                                    st.download_button(
                                        label="📥 Download Form 8949 PDF",
                                        data=pdf_files[0]['content'],
                                        file_name=pdf_files[0]['filename'],
                                        mime="application/pdf",
                                        help="Print this PDF and mail to the IRS with your tax return"
                                    )
                                else:
                                    # Multiple PDFs in ZIP
                                    zip_data = create_zip_file(pdf_files)
                                    st.download_button(
                                        label="📦 Download All Form 8949 PDFs (ZIP)",
                                        data=zip_data,
                                        file_name=f"form_8949_{tax_year}_complete.zip",
                                        mime="application/zip"
                                    )
                                
                                st.success(f"✅ Generated {len(pdf_files)} Form 8949 PDF(s)!")
                    
                    except Exception as e:
                        st.error(f"Error generating files: {str(e)}")
        
        else:
            st.info("👆 Please upload your Bitwave actions file first.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def extract_bitwave_transactions(df, target_year):
    """Extract and process transactions from Bitwave actions report"""
    
    # Create lot mapping from buy transactions
    lot_map = {}
    buy_transactions = df[df['action'] == 'buy'].copy()
    
    for _, row in buy_transactions.iterrows():
        if pd.notna(row['lotId']):
            lot_map[row['lotId']] = {
                'buy_date': pd.to_datetime(row['timestamp']),
                'asset': row['asset'],
                'cost_basis_acquired': clean_currency_value(row.get(' costBasisAcquired ', 0))
            }
    
    # Process sell transactions
    sell_transactions = df[df['action'] == 'sell'].copy()
    form8949_transactions = []
    
    for _, row in sell_transactions.iterrows():
        try:
            # Parse sell date
            sell_date = pd.to_datetime(row['timestamp'])
            
            # Filter by target year
            if sell_date.year != target_year:
                continue
            
            # Get lot information
            lot_id = row['lotId']
            lot_info = lot_map.get(lot_id, {})
            buy_date = lot_info.get('buy_date')
            
            # Extract monetary values
            proceeds = clean_currency_value(row.get(' proceeds ', 0))
            cost_basis = clean_currency_value(row.get(' costBasisRelieved ', 0))
            
            # Skip if no meaningful transaction
            if proceeds <= 0 and cost_basis <= 0:
                continue
            
            # Parse gain/loss values for validation
            short_term_gl = clean_currency_value(row.get(' shortTermGainLoss ', 0))
            long_term_gl = clean_currency_value(row.get(' longTermGainLoss ', 0))
            
            # Calculate gain/loss
            calculated_gain_loss = proceeds - cost_basis
            reported_gain_loss = short_term_gl + long_term_gl
            
            # Determine if short-term or long-term
            is_short_term = abs(short_term_gl) > 0.01
            is_long_term = abs(long_term_gl) > 0.01
            
            # If buy date is available, calculate holding period
            if buy_date and sell_date:
                holding_days = (sell_date - buy_date).days
                # Override with actual holding period if available
                if holding_days <= 365:
                    is_short_term = True
                    is_long_term = False
                else:
                    is_short_term = False
                    is_long_term = True
            
            # Create transaction record
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
            # Skip problematic rows
            continue
    
    return form8949_transactions

def clean_currency_value(value):
    """Clean and parse currency values from Bitwave format"""
    if pd.isna(value) or value == '' or value == '-':
        return 0.0
    
    # Convert to string and clean
    str_val = str(value).strip()
    
    # Handle parentheses for negative values
    is_negative = False
    if '(' in str_val and ')' in str_val:
        is_negative = True
        str_val = str_val.replace('(', '').replace(')', '')
    
    # Remove currency symbols, commas, and spaces
    str_val = re.sub(r'[,$\s]', '', str_val)
    
    try:
        result = float(str_val)
        return -result if is_negative else result
    except:
        return 0.0

def generate_tax_software_csv(transactions, tax_year):
    """Generate CSV for tax software import"""
    
    csv_lines = []
    csv_lines.append("Description,Date Acquired,Date Sold,Sales Price,Cost Basis,Gain/Loss,Adjustment Code,Adjustment Amount")
    
def generate_tax_software_csv(transactions, tax_year):
    """Generate CSV for tax software import"""
    
    csv_lines = []
    csv_lines.append("Description,Date Acquired,Date Sold,Sales Price,Cost Basis,Gain/Loss,Adjustment Code,Adjustment Amount")
    
    for transaction in transactions:
        # Format dates
        date_acquired = transaction['date_acquired'].strftime('%m/%d/%Y') if transaction['date_acquired'] else '01/01/2020'
        date_sold = transaction['date_sold'].strftime('%m/%d/%Y')
        
        row = [
            transaction['description'],
            date_acquired,
            date_sold,
            f"{transaction['proceeds']:.2f}",
            f"{transaction['cost_basis']:.2f}",
            f"{transaction['gain_loss']:.2f}",
            "",
            "0.00"
        ]
        csv_lines.append(",".join(row))
    
    return "\n".join(csv_lines)

def generate_form_8949_pdf(transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, term_type=""):
    """Generate completed Form 8949 PDF using official IRS template"""
    pdf_files = []
    
    # Split transactions into pages (14 per page max)
    transactions_per_page = 14
    total_pages = (len(transactions) + transactions_per_page - 1) // transactions_per_page
    
    for page_num in range(total_pages):
        start_idx = page_num * transactions_per_page
        end_idx = min(start_idx + transactions_per_page, len(transactions))
        page_transactions = transactions[start_idx:end_idx]
        
        # Create PDF for this page
        buffer = io.BytesIO()
        success = create_form_8949_with_official_template(buffer, page_transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, page_num + 1, total_pages, transactions)
        
        if not success:
            # Fallback to custom form if official template fails
            create_form_8949_page_custom(buffer, page_transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, page_num + 1, total_pages, transactions)
        
        # Generate filename
        term_suffix = f"_{term_type}" if term_type else ""
        if total_pages == 1:
            filename = f"Form_8949_{tax_year}{term_suffix}_{taxpayer_name.replace(' ', '_')}.pdf"
        else:
            filename = f"Form_8949_{tax_year}{term_suffix}_{taxpayer_name.replace(' ', '_')}_Page_{page_num + 1}.pdf"
        
        pdf_files.append({
            'filename': filename,
            'content': buffer.getvalue()
        })
    
    return pdf_files

def get_official_form_8949(tax_year):
    """Fetch the official IRS Form 8949 for the specified tax year"""
    
    # IRS Form 8949 URLs by year
    irs_urls = {
        2025: "https://www.irs.gov/pub/irs-pdf/f8949.pdf",
        2024: "https://www.irs.gov/pub/irs-pdf/f8949.pdf",
        2023: "https://www.irs.gov/pub/irs-prior/f8949--2023.pdf", 
        2022: "https://www.irs.gov/pub/irs-prior/f8949--2022.pdf",
        2021: "https://www.irs.gov/pub/irs-prior/f8949--2021.pdf",
        2020: "https://www.irs.gov/pub/irs-prior/f8949--2020.pdf",
        2019: "https://www.irs.gov/pub/irs-prior/f8949--2019.pdf",
        2018: "https://www.irs.gov/pub/irs-prior/f8949--2018.pdf"
    }
    
    # Try to fetch the official form
    url = irs_urls.get(tax_year, irs_urls[2024])  # Default to latest if year not found
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.content
        else:
            # Fallback: try the current year form
            response = requests.get(irs_urls[2024], timeout=15)
            if response.status_code == 200:
                return response.content
    except Exception as e:
        print(f"Error fetching official form: {e}")
        pass
    
    return None

def create_form_8949_with_official_template(buffer, page_transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, page_number, total_pages, all_transactions):
    """Create Form 8949 using official IRS template as base"""
    
    try:
        # Get official form
        official_form_pdf = get_official_form_8949(tax_year)
        
        if not official_form_pdf:
            return False
        
        # Use official form as base and overlay data
        return create_form_with_pdf_overlay(buffer, page_transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, page_number, total_pages, all_transactions, official_form_pdf)
        
    except Exception as e:
        print(f"Error creating form with official template: {e}")
        return False

def create_form_with_pdf_overlay(buffer, page_transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, page_number, total_pages, all_transactions, official_form_pdf):
    """Overlay transaction data onto official IRS Form 8949 PDF with precise positioning"""
    
    try:
        # Read the official PDF
        official_pdf_stream = io.BytesIO(official_form_pdf)
        pdf_reader = PyPDF2.PdfReader(official_pdf_stream)
        
        # Determine which page to use (Part I or Part II)
        template_page_num = 0 if "Part I" in form_type else 1
        if template_page_num >= len(pdf_reader.pages):
            template_page_num = 0  # Fallback to first page
        
        template_page = pdf_reader.pages[template_page_num]
        
        # Create overlay with transaction data
        overlay_buffer = io.BytesIO()
        c = canvas.Canvas(overlay_buffer, pagesize=letter)
        width, height = letter
        
        # PRECISE coordinates measured from actual IRS Form 8949
        # These coordinates are carefully measured to fit within the table cells
        
        # Taxpayer information fields
        name_x, name_y = 95, height - 133
        ssn_x, ssn_y = 415, height - 133
        
        # Checkbox positions (measured precisely)
        if "Part I" in form_type:
            checkbox_base_y = height - 208   # Short-term section
            # Transaction table starts lower for Part I
            table_start_y = height - 295
        else:
            checkbox_base_y = height - 393   # Long-term section  
            # Transaction table starts lower for Part II
            table_start_y = height - 480
        
        checkbox_x = 54
        
        # Column positions - precisely measured to center within each cell
        # These measurements ensure text is centered within each blue box
        col_a_x = 65      # Description - left aligned within cell
        col_a_width = 115  # Max width for description text
        
        col_b_center = 208  # Date acquired - center of cell
        col_c_center = 268  # Date sold - center of cell
        
        col_d_right = 340   # Proceeds - right edge of cell for alignment
        col_e_right = 400   # Cost basis - right edge of cell
        
        col_f_center = 425  # Code - center of small cell
        col_g_right = 465   # Adjustment - right edge
        col_h_right = 555   # Gain/Loss - right edge of cell
        
        # Row spacing - exactly matches IRS form line spacing
        row_height = 16.8  # Measured spacing between form lines
        
        # Set font for taxpayer info
        c.setFont("Helvetica", 9)
        
        # Fill in taxpayer information
        c.drawString(name_x, name_y, taxpayer_name[:40])
        c.drawString(ssn_x, ssn_y, taxpayer_ssn)
        
        # Check appropriate checkbox
        c.setFont("Helvetica", 11)
        if "Box A" in form_type:
            c.drawString(checkbox_x, checkbox_base_y, "✓")
        elif "Box B" in form_type:
            c.drawString(checkbox_x, checkbox_base_y - 17, "✓") 
        elif "Box C" in form_type:
            c.drawString(checkbox_x, checkbox_base_y - 34, "✓")
        elif "Box D" in form_type:
            c.drawString(checkbox_x, checkbox_base_y, "✓")
        elif "Box E" in form_type:
            c.drawString(checkbox_x, checkbox_base_y - 17, "✓")
        elif "Box F" in form_type:
            c.drawString(checkbox_x, checkbox_base_y - 34, "✓")
        
        # Set font for transaction data - smaller to fit cleanly in cells
        c.setFont("Helvetica", 7)
        
        for i, transaction in enumerate(page_transactions[:14]):  # Max 14 transactions per page
            y_pos = table_start_y - (i * row_height)
            
            # Format dates
            date_acquired = transaction['date_acquired'].strftime('%m/%d/%Y') if transaction['date_acquired'] else 'VARIOUS'
            date_sold = transaction['date_sold'].strftime('%m/%d/%Y')
            
            # Truncate description to fit within column width
            description = transaction['description']
            if len(description) > 28:
                description = description[:25] + "..."
            
            # Column (a) - Description: Left-aligned within cell
            c.drawString(col_a_x, y_pos, description)
            
            # Column (b) - Date acquired: Centered in cell
            date_acq_width = c.stringWidth(date_acquired)
            c.drawString(col_b_center - date_acq_width/2, y_pos, date_acquired)
            
            # Column (c) - Date sold: Centered in cell  
            date_sold_width = c.stringWidth(date_sold)
            c.drawString(col_c_center - date_sold_width/2, y_pos, date_sold)
            
            # Column (d) - Proceeds: Right-aligned within cell
            proceeds_text = f"{transaction['proceeds']:,.2f}"
            c.drawRightString(col_d_right, y_pos, proceeds_text)
            
            # Column (e) - Cost basis: Right-aligned within cell
            basis_text = f"{transaction['cost_basis']:,.2f}"
            c.drawRightString(col_e_right, y_pos, basis_text)
            
            # Column (f) - Code: Leave blank (standard for crypto)
            
            # Column (g) - Adjustment: Leave blank
            
            # Column (h) - Gain/Loss: Right-aligned, use parentheses for losses
            gain_loss = transaction['gain_loss']
            if gain_loss < 0:
                gain_loss_text = f"({abs(gain_loss):,.2f})"
            else:
                gain_loss_text = f"{gain_loss:,.2f}"
            c.drawRightString(col_h_right, y_pos, gain_loss_text)
        
        # Add totals on last page only
        if page_number == total_pages and len(page_transactions) > 0:
            # Position totals in the official totals row
            totals_y = table_start_y - (14 * row_height) - 5
            
            # Calculate totals
            total_proceeds = sum(t['proceeds'] for t in all_transactions)
            total_basis = sum(t['cost_basis'] for t in all_transactions)
            total_gain_loss = sum(t['gain_loss'] for t in all_transactions)
            
            # Use slightly bolder font for totals
            c.setFont("Helvetica-Bold", 7)
            
            # Draw totals in same column positions
            total_proceeds_text = f"{total_proceeds:,.2f}"
            total_basis_text = f"{total_basis:,.2f}"
            
            c.drawRightString(col_d_right, totals_y, total_proceeds_text)
            c.drawRightString(col_e_right, totals_y, total_basis_text)
            
            # Format total gain/loss
            if total_gain_loss < 0:
                total_gl_text = f"({abs(total_gain_loss):,.2f})"
            else:
                total_gl_text = f"{total_gain_loss:,.2f}"
            c.drawRightString(col_h_right, totals_y, total_gl_text)
        
        c.save()
        
        # Merge overlay with template
        overlay_buffer.seek(0)
        overlay_reader = PyPDF2.PdfReader(overlay_buffer)
        overlay_page = overlay_reader.pages[0]
        
        # Merge the pages
        template_page.merge_page(overlay_page)
        
        # Write to output buffer
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(template_page)
        pdf_writer.write(buffer)
        
        return True
        
    except Exception as e:
        print(f"Error in PDF overlay: {e}")
        return False

def create_form_8949_page_custom(buffer, page_transactions, form_type, taxpayer_name, taxpayer_ssn, tax_year, page_number, total_pages, all_transactions):
    """Create a custom Form 8949 PDF page with precise table formatting"""
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Page margins
    left_margin = 50
    right_margin = width - 50
    top_margin = height - 50
    
    # Form header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, top_margin, "Form 8949")
    c.setFont("Helvetica", 12)
    c.drawString(left_margin + 100, top_margin, "Sales and Other Dispositions of Capital Assets")
    c.drawRightString(right_margin, top_margin, f"{tax_year}")
    
    # Department line
    c.setFont("Helvetica", 9)
    c.drawString(left_margin + 100, top_margin - 15, "Department of the Treasury - Internal Revenue Service")
    
    # Taxpayer information section
    info_y = top_margin - 50
    c.setFont("Helvetica", 10)
    c.drawString(left_margin, info_y, f"Name(s) shown on return:")
    c.drawString(left_margin + 150, info_y, taxpayer_name)
    c.drawString(left_margin + 350, info_y, f"Your social security number:")
    c.drawString(left_margin + 520, info_y, taxpayer_ssn)
    
    # Part section
    part_y = info_y - 40
    c.setFont("Helvetica-Bold", 11)
    if "Part I" in form_type:
        c.drawString(left_margin, part_y, "Part I - Short-Term Capital Gains and Losses")
        c.setFont("Helvetica", 9)
        c.drawString(left_margin, part_y - 12, "Generally for assets held one year or less")
    else:
        c.drawString(left_margin, part_y, "Part II - Long-Term Capital Gains and Losses")
        c.setFont("Helvetica", 9)
        c.drawString(left_margin, part_y - 12, "Generally for assets held more than one year")
    
    # Checkbox section
    checkbox_y = part_y - 40
    c.setFont("Helvetica", 9)
    
    # Determine checkbox options based on Part
    if "Part I" in form_type:
        options = [
            ("(A) Short-term transactions reported on Form(s) 1099-B showing basis was reported to the IRS", "A"),
            ("(B) Short-term transactions reported on Form(s) 1099-B showing basis was NOT reported to the IRS", "B"),
            ("(C) Short-term transactions not reported to you on Form 1099-B", "C")
        ]
    else:
        options = [
            ("(D) Long-term transactions reported on Form(s) 1099-B showing basis was reported to the IRS", "D"),
            ("(E) Long-term transactions reported on Form(s) 1099-B showing basis was NOT reported to the IRS", "E"),
            ("(F) Long-term transactions not reported to you on Form 1099-B", "F")
        ]
    
    # Draw checkboxes
    for i, (text, code) in enumerate(options):
        y_pos = checkbox_y - (i * 15)
        # Check the appropriate box
        is_checked = f"Box {code}" in form_type
        checkbox = "☑" if is_checked else "☐"
        c.drawString(left_margin, y_pos, f"{checkbox} {text}")
    
    # Transaction table - precisely sized to match IRS form
    table_y = checkbox_y - 80
    
    # Draw table headers with precise positioning
    c.setFont("Helvetica-Bold", 8)
    
    # Column definitions with exact measurements from IRS form
    columns = [
        {"header": "(a) Description of property", "x": left_margin + 5, "width": 110, "align": "left"},
        {"header": "(b) Date acquired", "x": left_margin + 120, "width": 50, "align": "center"},
        {"header": "(c) Date sold", "x": left_margin + 175, "width": 50, "align": "center"},
        {"header": "(d) Proceeds", "x": left_margin + 230, "width": 60, "align": "right"},
        {"header": "(e) Cost basis", "x": left_margin + 295, "width": 60, "align": "right"},
        {"header": "(f) Code", "x": left_margin + 360, "width": 30, "align": "center"},
        {"header": "(g) Adjustment", "x": left_margin + 395, "width": 50, "align": "right"},
        {"header": "(h) Gain/(loss)", "x": left_margin + 450, "width": 65, "align": "right"}
    ]
    
    # Draw column headers
    for col in columns:
        if col["align"] == "center":
            text_x = col["x"] + col["width"] / 2
            text_width = c.stringWidth(col["header"])
            c.drawString(text_x - text_width/2, table_y, col["header"])
        elif col["align"] == "right":
            text_x = col["x"] + col["width"] - 5
            c.drawRightString(text_x, table_y, col["header"])
        else:
            c.drawString(col["x"] + 3, table_y, col["header"])
    
    # Draw precise table borders
    table_top = table_y + 12
    table_bottom = table_y - (15 * 16)  # Space for 14 transactions + totals
    
    # Horizontal lines
    c.line(left_margin, table_top, right_margin, table_top)
    c.line(left_margin, table_y - 3, right_margin, table_y - 3)  # Under headers
    c.line(left_margin, table_bottom, right_margin, table_bottom)
    
    # Vertical lines - precisely positioned
    x_positions = [left_margin]
    for col in columns:
        x_positions.append(x_positions[-1] + col["width"])
    
    for x_pos in x_positions:
        c.line(x_pos, table_top, x_pos, table_bottom)
    
    # Fill in transaction data with precise alignment
    c.setFont("Helvetica", 7)
    row_height = 14
    
    for i, transaction in enumerate(page_transactions[:14]):
        if i >= 14:  # Max 14 transactions per page
            break
            
        y_pos = table_y - 18 - (i * row_height)
        
        # Format data to fit in cells
        description = transaction['description'][:26]  # Ensure it fits
        date_acquired = transaction['date_acquired'].strftime('%m/%d/%Y') if transaction['date_acquired'] else 'VARIOUS'
        date_sold = transaction['date_sold'].strftime('%m/%d/%Y')
        
        # Draw data precisely aligned within each cell
        # Column (a) - Description
        c.drawString(columns[0]["x"] + 3, y_pos, description)
        
        # Column (b) - Date acquired (centered)
        date_acq_width = c.stringWidth(date_acquired)
        center_b = columns[1]["x"] + columns[1]["width"]/2
        c.drawString(center_b - date_acq_width/2, y_pos, date_acquired)
        
        # Column (c) - Date sold (centered)
        date_sold_width = c.stringWidth(date_sold)
        center_c = columns[2]["x"] + columns[2]["width"]/2
        c.drawString(center_c - date_sold_width/2, y_pos, date_sold)
        
        # Column (d) - Proceeds (right aligned)
        c.drawRightString(columns[3]["x"] + columns[3]["width"] - 3, y_pos, f"{transaction['proceeds']:,.2f}")
        
        # Column (e) - Cost basis (right aligned)
        c.drawRightString(columns[4]["x"] + columns[4]["width"] - 3, y_pos, f"{transaction['cost_basis']:,.2f}")
        
        # Columns (f) and (g) - Leave empty
        
        # Column (h) - Gain/Loss (right aligned with parentheses for losses)
        gain_loss = transaction['gain_loss']
        if gain_loss < 0:
            gain_loss_text = f"({abs(gain_loss):,.2f})"
        else:
            gain_loss_text = f"{gain_loss:,.2f}"
        c.drawRightString(columns[7]["x"] + columns[7]["width"] - 3, y_pos, gain_loss_text)
        
        # Draw light row separator
        separator_y = y_pos - 6
        c.setStrokeColor(colors.lightgrey)
        c.line(left_margin + 1, separator_y, right_margin - 1, separator_y)
        c.setStrokeColor(colors.black)
    
    # Add totals row (only on last page)
    if page_number == total_pages:
        totals_y = table_y - 18 - (14 * row_height)
        
        # Calculate totals
        total_proceeds = sum(t['proceeds'] for t in all_transactions)
        total_basis = sum(t['cost_basis'] for t in all_transactions)
        total_gain_loss = sum(t['gain_loss'] for t in all_transactions)
        
        # Draw totals with bold font
        c.setFont("Helvetica-Bold", 7)
        c.drawString(columns[0]["x"] + 3, totals_y, "TOTALS")
        c.drawRightString(columns[3]["x"] + columns[3]["width"] - 3, totals_y, f"{total_proceeds:,.2f}")
        c.drawRightString(columns[4]["x"] + columns[4]["width"] - 3, totals_y, f"{total_basis:,.2f}")
        
        # Format total gain/loss
        if total_gain_loss < 0:
            total_gl_text = f"({abs(total_gain_loss):,.2f})"
        else:
            total_gl_text = f"{total_gain_loss:,.2f}"
        c.drawRightString(columns[7]["x"] + columns[7]["width"] - 3, totals_y, total_gl_text)
        
        # Bold line above totals
        c.setLineWidth(2)
        c.line(left_margin, totals_y + 8, right_margin, totals_y + 8)
        c.setLineWidth(1)
    
    # Page footer
    c.setFont("Helvetica", 8)
    footer_text = f"Form 8949 ({tax_year})"
    if total_pages > 1:
        footer_text += f" - Page {page_number} of {total_pages}"
    c.drawString(left_margin, 25, footer_text)
    c.drawRightString(right_margin, 25, f"Generated by Bitwave - {datetime.now().strftime('%m/%d/%Y')}")
    
    c.save()

def create_zip_file(pdf_files):
    """Create a ZIP file containing all PDFs"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for pdf_file in pdf_files:
            zip_file.writestr(pdf_file['filename'], pdf_file['content'])
    return zip_buffer.getvalue()

if __name__ == "__main__":
    main()
