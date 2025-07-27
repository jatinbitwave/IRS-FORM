import streamlit as st
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import io
import zipfile
from datetime import datetime
import re
import requests
import PyPDF2
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image
import base64

def main():
    """Main Streamlit application for Form 8949 generation from Bitwave actions reports"""
    st.set_page_config(
        page_title="Form 8949 Generator - Bitwave Edition",
        page_icon="📋",
        layout="wide"
    )
    
    # Professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-header {
        color: #1e3c72;
        border-bottom: 2px solid #2a5298;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .instruction-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📋 Bitwave Form 8949 Generator</h1>
        <p>This tool helps you convert a Bitwave actions report (csv file) to an official IRS Form 8949</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📖 Instructions & Bitwave Requirements", expanded=False):
        st.markdown("""
        ### Required Bitwave CSV Columns:
        - **action**: Transaction type ('sell' transactions will be processed)
        - **asset**: Cryptocurrency symbol (e.g., "BTC", "ETH", "SOL")
        - **assetUnitAdj**: Amount of cryptocurrency sold (used for description)
        - **timestamp**: Sale date in format "YYYY-MM-DD HH:MM:SS UTC" or Unix timestamp
        - **timestampSEC**: Alternative Unix timestamp field
        - **lotId**: Unique lot identifier for matching acquisitions
        - **lotAcquisitionTimestampSEC**: Acquisition timestamp in seconds
        - **fairMarketValueDisposed**: Fair market value at disposal (proceeds)
        - **costBasisRelieved**: Cost basis of sold assets
        - **shortTermGainLoss**: Short-term gain/loss from Bitwave
        - **longTermGainLoss**: Long-term gain/loss from Bitwave
        - **txnExchangeRate**: Exchange rate at transaction time (backup for proceeds calculation)
        
        ### PDF Generation Options:
        - **High-Quality PDF**: Creates a single, complete PDF with all data and formatting
        - **Form-Fillable PDF**: Uses official IRS forms with your data filled in
        - **Vector Graphics**: Ensures text remains selectable and printable
        """)
    
    # Taxpayer Information Section
    st.markdown('<h2 class="section-header">👤 Taxpayer Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        taxpayer_name = st.text_input(
            "Full Name (as shown on tax return)",
            placeholder="John and Jane Smith"
        )
    with col2:
        taxpayer_ssn = st.text_input(
            "Social Security Number",
            placeholder="123-45-6789"
        )
    
    # Form Configuration
    st.markdown('<h2 class="section-header">⚙️ Form Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        tax_year = st.selectbox(
            "Tax Year",
            [2024, 2023, 2022, 2021, 2020],
            index=0
        )
    
    with col2:
        default_box_type = st.selectbox(
            "Default Box Type",
            [
                "Box B - Basis NOT reported to IRS",
                "Box A - Basis reported to IRS", 
                "Box C - Various situations"
            ],
            index=0,
            help="For crypto transactions, Box B is typically correct as exchanges rarely report basis to IRS"
        )
    
    with col3:
        pdf_generation_mode = st.selectbox(
            "PDF Generation Mode",
            [
                "High-Quality Complete PDF",
                "Official IRS Form Template",
                "Custom Professional Layout"
            ],
            index=0,
            help="Choose how you want your PDF generated"
        )
    
    # File Upload Section
    st.markdown('<h2 class="section-header">📁 Upload Bitwave Actions Report</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose your Bitwave actions CSV export",
        type=['csv'],
        help="Upload the CSV file exported from Bitwave's Actions Report"
    )
    
    if uploaded_file is not None:
        try:
            # Read and validate CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Bitwave actions report uploaded successfully! Found {len(df)} total actions.")
            
            # Display column information for debugging
            st.info(f"📊 Columns found in CSV: {list(df.columns)}")
            
            # Display preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Validate required columns for Bitwave format with flexible column matching
            required_bitwave_columns = [
                'action', 'asset', 'assetUnitAdj', 'lotId', 'lotAcquisitionTimestampSEC',
                'shortTermGainLoss', 'longTermGainLoss', 'costBasisRelieved'
            ]
            
            # Check for timestamp columns (flexible matching)
            timestamp_column = None
            for col in df.columns:
                if col in ['timestamp', 'timestampSEC', 'date', 'saleDate']:
                    timestamp_column = col
                    break
            
            # Check for proceeds-related columns
            proceeds_column = None
            for col in df.columns:
                if col in ['fairMarketValueDisposed', 'proceeds', 'saleProceeds', 'disposalValue']:
                    proceeds_column = col
                    break
            
            # Check for exchange rate column as backup for proceeds calculation
            exchange_rate_column = None
            for col in df.columns:
                if col in ['txnExchangeRate', 'exchangeRate', 'rate']:
                    exchange_rate_column = col
                    break
            
            missing_columns = []
            for col in required_bitwave_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if timestamp_column is None:
                missing_columns.append('timestamp or timestampSEC')
            
            if proceeds_column is None and exchange_rate_column is None:
                missing_columns.append('fairMarketValueDisposed or txnExchangeRate')
            
            if missing_columns:
                st.error(f"❌ Missing required Bitwave columns: {', '.join(missing_columns)}")
                st.info("Please ensure you've uploaded a complete Bitwave actions report CSV.")
                return
            
            # Process Bitwave transactions
            transactions, validation_warnings = process_bitwave_transactions_fixed(df, tax_year, proceeds_column, exchange_rate_column, timestamp_column)
            
            # Enhanced debugging information
            sell_count = len(df[df['action'] == 'sell'])
            buy_count = len(df[df['action'] == 'buy'])
            
            st.info(f"📊 Bitwave Data Summary:")
            st.write(f"• Total actions in report: {len(df)}")
            st.write(f"• Sell actions found: {sell_count}")
            st.write(f"• Buy actions found: {buy_count}")
            st.write(f"• Valid transactions processed: {len(transactions)}")
            
            # Show date range information if we have transactions
            if len(transactions) > 0:
                earliest_sale = min(t['date_sold'] for t in transactions)
                latest_sale = max(t['date_sold'] for t in transactions)
                st.write(f"• Transaction date range: {earliest_sale.strftime('%m/%d/%Y')} to {latest_sale.strftime('%m/%d/%Y')}")
            elif len(df[df['action'] == 'sell']) > 0:
                # Show overall date range in the data to help user select correct tax year
                st.write("• **Date range analysis:**")
                sell_sample = df[df['action'] == 'sell'].head(100)  # Sample for performance
                if timestamp_column in sell_sample.columns:
                    # Try to parse sample timestamps for date range analysis
                    sample_timestamps = []
                    for ts in sell_sample[timestamp_column].dropna().head(10):
                        try:
                            parsed_date = parse_flexible_timestamp(ts)
                            if parsed_date:
                                sample_timestamps.append(parsed_date)
                        except:
                            continue
                    
                    if sample_timestamps:
                        earliest_date = min(sample_timestamps)
                        latest_date = max(sample_timestamps)
                        st.write(f"  Sample shows transactions from {earliest_date.strftime('%m/%d/%Y')} to {latest_date.strftime('%m/%d/%Y')}")
                        st.write(f"  Consider selecting tax year {earliest_date.year} or {latest_date.year}")
            
            if not transactions:
                st.error("❌ No valid sell transactions could be processed from the Bitwave actions report.")
                if sell_count > 0:
                    st.error(f"⚠️ Found {sell_count} sell actions but none could be processed. Check validation warnings above.")
                else:
                    st.error("⚠️ No sell actions found in the CSV. Please ensure this is a complete Bitwave actions report.")
                return
            
            # Display validation warnings if any
            if validation_warnings:
                with st.expander("⚠️ Validation Warnings", expanded=True):
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    for warning in validation_warnings:
                        st.warning(warning)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Separate short-term and long-term using Bitwave's classification
            short_term, long_term = separate_bitwave_transactions_by_term(transactions)
            
            # Display summary
            st.markdown('<h2 class="section-header">📈 Transaction Summary</h2>', unsafe_allow_html=True)
            
            total_actions = len(df)
            sell_actions = len(df[df['action'] == 'sell'])
            buy_actions = len(df[df['action'] == 'buy'])
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Actions", total_actions)
            with col2:
                st.metric("Sell Actions", sell_actions)
            with col3:
                st.metric("Valid Transactions", len(transactions))
            with col4:
                st.metric("Short-term", len(short_term))
            with col5:
                st.metric("Long-term", len(long_term))
            
            # Net gain/loss summary
            col1, col2, col3 = st.columns(3)
            with col1:
                total_proceeds = sum(t['proceeds'] for t in transactions)
                st.metric("Total Proceeds", f"${total_proceeds:,.2f}")
            with col2:
                total_basis = sum(t['cost_basis'] for t in transactions)
                st.metric("Total Cost Basis", f"${total_basis:,.2f}")
            with col3:
                total_gain_loss = sum(t['gain_loss'] for t in transactions)
                st.metric("Net Gain/Loss", f"${total_gain_loss:,.2f}")
            
            # Asset breakdown
            with st.expander("💰 Asset Breakdown", expanded=False):
                asset_summary = {}
                for transaction in transactions:
                    asset = transaction['description']
                    if asset not in asset_summary:
                        asset_summary[asset] = {'count': 0, 'proceeds': 0, 'gain_loss': 0}
                    asset_summary[asset]['count'] += 1
                    asset_summary[asset]['proceeds'] += transaction['proceeds']
                    asset_summary[asset]['gain_loss'] += transaction['gain_loss']
                
                summary_df = pd.DataFrame([
                    {
                        'Asset': asset,
                        'Transactions': data['count'],
                        'Total Proceeds': f"${data['proceeds']:,.2f}",
                        'Net Gain/Loss': f"${data['gain_loss']:,.2f}"
                    }
                    for asset, data in asset_summary.items()
                ])
                st.dataframe(summary_df, use_container_width=True)
            
            # Generate Forms
            if st.button("🚀 Generate Form 8949 PDFs", type="primary"):
                if not taxpayer_name or not taxpayer_ssn:
                    st.error("⚠️ Please enter taxpayer name and SSN before generating forms.")
                    return
                
                with st.spinner("Generating official Form 8949 PDFs from Bitwave data..."):
                    pdf_files = generate_all_forms_enhanced(
                        short_term, 
                        long_term, 
                        taxpayer_name, 
                        taxpayer_ssn, 
                        tax_year, 
                        default_box_type,
                        pdf_generation_mode
                    )
                
                if pdf_files:
                    if len(pdf_files) == 1:
                        # Single PDF download
                        st.download_button(
                            label="📥 Download Complete Form 8949 PDF",
                            data=pdf_files[0]['content'],
                            file_name=pdf_files[0]['filename'],
                            mime="application/pdf"
                        )
                    else:
                        # Multiple PDFs in ZIP
                        zip_data = create_zip_file(pdf_files)
                        st.download_button(
                            label="📦 Download All Forms (ZIP)",
                            data=zip_data,
                            file_name=f"Form_8949_{tax_year}_Bitwave_Complete.zip",
                            mime="application/zip"
                        )
                    
                    st.success(f"✅ Generated {len(pdf_files)} high-quality Form 8949 PDF(s) successfully from Bitwave data!")
                    
                    # Show summary of what was generated
                    if short_term:
                        st.info(f"📄 Part I (Short-term): {len(short_term)} transactions")
                    if long_term:
                        st.info(f"📄 Part II (Long-term): {len(long_term)} transactions")
                        
                else:
                    st.error("❌ Failed to generate PDF files. Please check your data and try again.")
        
        except Exception as e:
            st.error(f"❌ Error processing Bitwave actions report: {str(e)}")
            st.info("Please ensure you've uploaded a valid Bitwave actions CSV export.")

def parse_flexible_timestamp(timestamp_value):
    """Parse timestamp in various formats and remove UTC suffix"""
    if pd.isna(timestamp_value) or timestamp_value == '' or timestamp_value == 0:
        return None
    
    # Convert to string for processing
    ts_str = str(timestamp_value).strip()
    
    # Handle Unix timestamp (numeric)
    try:
        # If it's a pure number, treat as Unix timestamp
        if ts_str.replace('.', '').isdigit():
            unix_ts = float(ts_str)
            # Handle both seconds and milliseconds timestamps
            if unix_ts > 1e10:  # Likely milliseconds
                unix_ts = unix_ts / 1000
            return pd.to_datetime(unix_ts, unit='s')
    except:
        pass
    
    # Handle string timestamps
    try:
        # Remove UTC suffix if present
        if ts_str.endswith(' UTC'):
            ts_str = ts_str[:-4].strip()
        
        # Parse the cleaned timestamp
        return pd.to_datetime(ts_str)
    except:
        pass
    
    # Last resort: try pandas flexible parsing
    try:
        return pd.to_datetime(timestamp_value, errors='coerce')
    except:
        return None

def process_bitwave_transactions_fixed(df, tax_year, proceeds_column, exchange_rate_column, timestamp_column):
    """Fixed version: Process Bitwave actions report into standardized transaction format for specified tax year"""
    transactions = []
    validation_warnings = []
    
    # Filter for sell actions only
    sell_actions = df[df['action'] == 'sell'].copy()
    
    if len(sell_actions) == 0:
        validation_warnings.append("No 'sell' actions found in the Bitwave report.")
        return transactions, validation_warnings
    
    # Define tax year date range
    tax_year_start = pd.Timestamp(f'{tax_year}-01-01')
    tax_year_end = pd.Timestamp(f'{tax_year}-12-31 23:59:59')
    
    st.info(f"Processing {len(sell_actions)} sell transactions from Bitwave actions report...")
    st.info(f"📅 Filtering for tax year {tax_year}: {tax_year_start.strftime('%B %d, %Y')} to {tax_year_end.strftime('%B %d, %Y')}")
    
    # Show which columns we're using
    st.info(f"🕐 Using '{timestamp_column}' column for sale dates")
    if proceeds_column:
        st.info(f"💰 Using '{proceeds_column}' column for proceeds")
    elif exchange_rate_column:
        st.info(f"💰 Calculating proceeds from assetUnitAdj × {exchange_rate_column}")
    
    processed_count = 0
    error_count = 0
    filtered_out_count = 0
    
    for _, row in sell_actions.iterrows():
        try:
            # Extract and validate sale date using flexible timestamp parsing
            try:
                timestamp_value = row[timestamp_column]
                date_sold = parse_flexible_timestamp(timestamp_value)
                if date_sold is None:
                    raise ValueError(f"Could not parse timestamp: {timestamp_value}")
            except Exception as e:
                validation_warnings.append(f"Invalid sale timestamp for transaction {row.get('txnId', 'unknown')}: {str(e)}")
                error_count += 1
                continue
            
            try:
                # Convert acquisition timestamp from seconds to datetime
                lot_acq_timestamp = row['lotAcquisitionTimestampSEC']
                if pd.isna(lot_acq_timestamp) or lot_acq_timestamp == 0 or lot_acq_timestamp == '':
                    raise ValueError("Missing acquisition timestamp")
                
                # Ensure it's a number and convert to datetime
                lot_acq_seconds = float(lot_acq_timestamp)
                date_acquired = pd.to_datetime(lot_acq_seconds, unit='s', errors='coerce')
                if pd.isna(date_acquired):
                    raise ValueError("Invalid acquisition date conversion")
                    
            except Exception as e:
                validation_warnings.append(f"Invalid acquisition timestamp for transaction {row.get('txnId', 'unknown')}: {str(e)}")
                error_count += 1
                continue
            
            # Calculate holding period
            holding_days = (date_sold - date_acquired).days
            
            # Filter by tax year - only include transactions sold within the specified tax year
            if not (tax_year_start <= date_sold <= tax_year_end):
                filtered_out_count += 1
                continue
            
            # Calculate proceeds using available columns
            proceeds = 0.0
            if proceeds_column and proceeds_column in row and not pd.isna(row[proceeds_column]):
                proceeds = clean_bitwave_currency_value(row[proceeds_column])
            elif exchange_rate_column and exchange_rate_column in row:
                # Calculate proceeds from amount × exchange rate
                asset_amount = abs(clean_bitwave_currency_value(row['assetUnitAdj']))
                exchange_rate = clean_bitwave_currency_value(row[exchange_rate_column])
                if asset_amount > 0 and exchange_rate > 0:
                    proceeds = asset_amount * exchange_rate
                else:
                    validation_warnings.append(f"Cannot calculate proceeds for {row.get('txnId', 'unknown')}: assetUnitAdj={asset_amount}, rate={exchange_rate}")
                    error_count += 1
                    continue
            else:
                validation_warnings.append(f"No proceeds data available for transaction {row.get('txnId', 'unknown')}")
                error_count += 1
                continue
            
            # Get cost basis
            cost_basis = clean_bitwave_currency_value(row['costBasisRelieved'])
            
            # Get gain/loss from Bitwave's calculations
            short_term_gl = clean_bitwave_currency_value(row['shortTermGainLoss'])
            long_term_gl = clean_bitwave_currency_value(row['longTermGainLoss'])
            
            # Determine if short-term or long-term based on Bitwave's classification
            if short_term_gl != 0 and long_term_gl == 0:
                is_short_term = True
                bitwave_gain_loss = short_term_gl
            elif long_term_gl != 0 and short_term_gl == 0:
                is_short_term = False
                bitwave_gain_loss = long_term_gl
            else:
                # Fallback to holding period calculation if Bitwave classification is unclear
                is_short_term = holding_days <= 365
                bitwave_gain_loss = short_term_gl + long_term_gl
                if short_term_gl != 0 and long_term_gl != 0:
                    validation_warnings.append(
                        f"Transaction {row.get('txnId', 'unknown')}: Both short and long-term gains reported. Using combined total."
                    )
            
            # Calculate gain/loss and validate against Bitwave
            calculated_gain_loss = proceeds - cost_basis
            
            # Validate calculated vs Bitwave gain/loss (allow small rounding differences)
            if abs(calculated_gain_loss - bitwave_gain_loss) > 0.02:
                validation_warnings.append(
                    f"Asset {row['asset']} on {date_sold.strftime('%m/%d/%Y')}: "
                    f"Calculated G/L ${calculated_gain_loss:.2f} vs Bitwave G/L ${bitwave_gain_loss:.2f}"
                )
            
            # Create transaction record
            transaction = {
                'description': f"{abs(row['assetUnitAdj']):.8f} {row['asset']}".rstrip('0').rstrip('.'),  # Format: "22 SOL"
                'date_acquired': date_acquired,
                'date_sold': date_sold,
                'proceeds': proceeds,
                'cost_basis': cost_basis,
                'gain_loss': bitwave_gain_loss,  # Use Bitwave's calculation for accuracy
                'is_short_term': is_short_term,
                'holding_days': holding_days,
                'lot_id': row['lotId'],
                'txn_id': row.get('txnId', 'unknown')
            }
            
            transactions.append(transaction)
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            validation_warnings.append(f"Error processing row {row.get('txnId', 'unknown')}: {str(e)}")
            continue
    
    # Add summary information
    if processed_count > 0:
        st.success(f"✅ Successfully processed {processed_count} transactions for tax year {tax_year}")
    if error_count > 0:
        st.warning(f"⚠️ Skipped {error_count} transactions due to data issues")
    if filtered_out_count > 0:
        st.info(f"📅 Filtered out {filtered_out_count} transactions outside tax year {tax_year}")
    
    return transactions, validation_warnings

def clean_bitwave_currency_value(value):
    """Clean and parse currency values from Bitwave format"""
    if pd.isna(value) or value == '' or value == '-' or value == ' -   ':
        return 0.0
    
    # Convert to string and clean
    str_val = str(value).strip()
    
    # Handle Bitwave's format for zero/empty values
    if str_val in ['-', ' -   ', '', 'null', 'None']:
        return 0.0
    
    # Handle parentheses for negative values (Bitwave format)
    is_negative = False
    if '(' in str_val and ')' in str_val:
        is_negative = True
        str_val = str_val.replace('(', '').replace(')', '')
    
    # Remove currency symbols, commas, spaces
    str_val = re.sub(r'[,$\s]', '', str_val)
    
    try:
        result = float(str_val)
        return -result if is_negative else result
    except (ValueError, TypeError):
        return 0.0

def separate_bitwave_transactions_by_term(transactions):
    """Separate transactions using Bitwave's short/long-term classification"""
    short_term = [t for t in transactions if t['is_short_term']]
    long_term = [t for t in transactions if not t['is_short_term']]
    return short_term, long_term

def generate_all_forms_enhanced(short_term, long_term, taxpayer_name, taxpayer_ssn, tax_year, default_box_type, pdf_mode):
    """Generate all required Form 8949 PDFs with enhanced options"""
    pdf_files = []
    
    # Generate short-term forms (Part I)
    if short_term:
        short_term_pdfs = generate_form_8949_pages_enhanced(
            short_term,
            "Part I",
            taxpayer_name,
            taxpayer_ssn,
            tax_year,
            default_box_type,
            "Short_Term",
            pdf_mode
        )
        pdf_files.extend(short_term_pdfs)
    
    # Generate long-term forms (Part II)
    if long_term:
        long_term_pdfs = generate_form_8949_pages_enhanced(
            long_term,
            "Part II",
            taxpayer_name,
            taxpayer_ssn,
            tax_year,
            default_box_type,
            "Long_Term",
            pdf_mode
        )
        pdf_files.extend(long_term_pdfs)
    
    return pdf_files

def generate_form_8949_pages_enhanced(transactions, part_type, taxpayer_name, taxpayer_ssn, tax_year, box_type, term_suffix, pdf_mode):
    """Generate Form 8949 pages with enhanced PDF handling"""
    pdf_files = []
    
    # Split transactions into pages (14 per page maximum)
    transactions_per_page = 14
    total_pages = (len(transactions) + transactions_per_page - 1) // transactions_per_page
    
    for page_num in range(total_pages):
        start_idx = page_num * transactions_per_page
        end_idx = min(start_idx + transactions_per_page, len(transactions))
        page_transactions = transactions[start_idx:end_idx]
        
        # Create PDF for this page
        buffer = io.BytesIO()
        
        # Choose PDF generation method based on mode
        if pdf_mode == "High-Quality Complete PDF":
            success = create_high_quality_complete_pdf(
                buffer, page_transactions, part_type, taxpayer_name,
                taxpayer_ssn, tax_year, box_type, page_num + 1, total_pages, transactions
            )
        elif pdf_mode == "Official IRS Form Template":
            success = create_form_with_official_template_enhanced(
                buffer, page_transactions, part_type, taxpayer_name, 
                taxpayer_ssn, tax_year, box_type, page_num + 1, total_pages, transactions
            )
        else:  # Custom Professional Layout
            success = create_professional_custom_form(
                buffer, page_transactions, part_type, taxpayer_name,
                taxpayer_ssn, tax_year, box_type, page_num + 1, total_pages, transactions
            )
        
        if not success:
            # Fallback to custom form
            create_professional_custom_form(
                buffer, page_transactions, part_type, taxpayer_name,
                taxpayer_ssn, tax_year, box_type, page_num + 1, total_pages, transactions
            )
        
        # Generate filename
        if total_pages == 1:
            filename = f"Form_8949_{tax_year}_{term_suffix}_Complete_{taxpayer_name.replace(' ', '_')}.pdf"
        else:
            filename = f"Form_8949_{tax_year}_{term_suffix}_Page_{page_num + 1}_Complete_{taxpayer_name.replace(' ', '_')}.pdf"
        
        pdf_files.append({
            'filename': filename,
            'content': buffer.getvalue()
        })
    
    return pdf_files

def create_high_quality_complete_pdf(buffer, transactions, part_type, taxpayer_name, taxpayer_ssn, tax_year, box_type, page_num, total_pages, all_transactions):
    """Create a high-quality, complete PDF with all data in one unified document"""
    try:
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        
        # Create document with proper margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Build story (content elements)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.black
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            alignment=TA_LEFT,
            textColor=colors.black
        )
        
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            alignment=TA_LEFT
        )
        
        # Form header
        story.append(Paragraph(f"Form 8949 - Sales and Other Dispositions of Capital Assets", title_style))
        story.append(Paragraph(f"Tax Year {tax_year}", header_style))
        story.append(Spacer(1, 20))
        
        # Taxpayer information
        story.append(Paragraph(f"<b>Taxpayer Name:</b> {taxpayer_name}", info_style))
        story.append(Paragraph(f"<b>Social Security Number:</b> {taxpayer_ssn}", info_style))
        story.append(Paragraph(f"<b>Generated from:</b> Bitwave Actions Report", info_style))
        story.append(Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", info_style))
        story.append(Spacer(1, 20))
        
        # Part header
        if part_type == "Part I":
            story.append(Paragraph("Part I - Short-Term Capital Gains and Losses", header_style))
            story.append(Paragraph("Assets held one year or less", info_style))
        else:
            story.append(Paragraph("Part II - Long-Term Capital Gains and Losses", header_style))
            story.append(Paragraph("Assets held more than one year", info_style))
        
        story.append(Paragraph(f"<b>Box Type:</b> {box_type}", info_style))
        story.append(Spacer(1, 20))
        
        # Create transaction table
        table_data = []
        
        # Table headers
        headers = [
            'Description\nof Property',
            'Date\nAcquired',
            'Date\nSold',
            'Proceeds\n(Sales Price)',
            'Cost or Other\nBasis',
            'Code(s)',
            'Adjustment',
            'Gain or (Loss)'
        ]
        table_data.append(headers)
        
        # Add transaction data
        for transaction in transactions:
            row = [
                transaction['description'],
                transaction['date_acquired'].strftime('%m/%d/%Y'),
                transaction['date_sold'].strftime('%m/%d/%Y'),
                f"${transaction['proceeds']:,.2f}",
                f"${transaction['cost_basis']:,.2f}",
                "",  # Code column - typically blank for crypto
                "",  # Adjustment column - typically blank
                f"${transaction['gain_loss']:,.2f}" if transaction['gain_loss'] >= 0 
                else f"(${abs(transaction['gain_loss']):,.2f})"
            ]
            table_data.append(row)
        
        # Add totals row if this is the last page
        if page_num == total_pages:
            total_proceeds = sum(t['proceeds'] for t in all_transactions)
            total_basis = sum(t['cost_basis'] for t in all_transactions)
            total_gain_loss = sum(t['gain_loss'] for t in all_transactions)
            
            totals_row = [
                'TOTALS',
                '',
                '',
                f"${total_proceeds:,.2f}",
                f"${total_basis:,.2f}",
                '',
                '',
                f"${total_gain_loss:,.2f}" if total_gain_loss >= 0 
                else f"(${abs(total_gain_loss):,.2f})"
            ]
            table_data.append(totals_row)
        
        # Create and style the table
        table = Table(table_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 1*inch, 1*inch, 0.5*inch, 0.7*inch, 1*inch])
        
        table_style = TableStyle([
            # Header row styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows styling
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),      # Description - left align
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),    # Dates - center align
            ('ALIGN', (3, 1), (-1, -1), 'RIGHT'),    # Numbers - right align
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue]),
            
            # Grid lines
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
            
            # Totals row styling (if present)
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightyellow) if page_num == total_pages else ('BACKGROUND', (0, -1), (-1, -1), colors.white),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold') if page_num == total_pages else ('FONTNAME', (0, -1), (-1, -1), 'Helvetica'),
        ])
        
        table.setStyle(table_style)
        story.append(table)
        
        # Add page footer information
        story.append(Spacer(1, 30))
        if total_pages > 1:
            story.append(Paragraph(f"Page {page_num} of {total_pages}", info_style))
        
        # Summary information
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Summary Information:</b>", header_style))
        
        page_proceeds = sum(t['proceeds'] for t in transactions)
        page_basis = sum(t['cost_basis'] for t in transactions)
        page_gain_loss = sum(t['gain_loss'] for t in transactions)
        
        story.append(Paragraph(f"• Transactions on this page: {len(transactions)}", info_style))
        story.append(Paragraph(f"• Page proceeds: ${page_proceeds:,.2f}", info_style))
        story.append(Paragraph(f"• Page cost basis: ${page_basis:,.2f}", info_style))
        story.append(Paragraph(f"• Page net gain/loss: ${page_gain_loss:,.2f}", info_style))
        
        if page_num == total_pages:
            total_proceeds = sum(t['proceeds'] for t in all_transactions)
            total_basis = sum(t['cost_basis'] for t in all_transactions)
            total_gain_loss = sum(t['gain_loss'] for t in all_transactions)
            
            story.append(Spacer(1, 10))
            story.append(Paragraph("<b>Total Summary (All Pages):</b>", header_style))
            story.append(Paragraph(f"• Total transactions: {len(all_transactions)}", info_style))
            story.append(Paragraph(f"• Total proceeds: ${total_proceeds:,.2f}", info_style))
            story.append(Paragraph(f"• Total cost basis: ${total_basis:,.2f}", info_style))
            story.append(Paragraph(f"• Total net gain/loss: ${total_gain_loss:,.2f}", info_style))
        
        # Build the PDF
        doc.build(story)
        return True
        
    except Exception as e:
        st.error(f"Error creating high-quality PDF: {e}")
        return False

def create_form_with_official_template_enhanced(buffer, transactions, part_type, taxpayer_name, taxpayer_ssn, tax_year, box_type, page_num, total_pages, all_transactions):
    """Enhanced version of official template creation with better PDF handling"""
    try:
        # Get official IRS Form 8949 PDF
        official_pdf = get_official_form_8949_enhanced(tax_year)
        if not official_pdf:
            return False
        
        # Read the official PDF
        official_pdf_stream = io.BytesIO(official_pdf)
        pdf_reader = PyPDF2.PdfReader(official_pdf_stream)
        
        # Select appropriate page (Part I = page 1, Part II = page 2)
        template_page_num = 0 if part_type == "Part I" else 1
        if template_page_num >= len(pdf_reader.pages):
            template_page_num = 0
        
        template_page = pdf_reader.pages[template_page_num]
        
        # Create overlay with transaction data
        overlay_buffer = io.BytesIO()
        c = canvas.Canvas(overlay_buffer, pagesize=letter)
        width, height = letter
        
        # Enhanced positioning for different tax years and parts
        if part_type == "Part I":
            # Part I (Page 1) positioning - adjusted for better alignment
            name_field_x = 75
            name_field_y = 690
            ssn_field_x = 550
            ssn_field_y = 690
            checkbox_base_y = 552
            table_start_y = 425
        else:  # Part II
            # Part II (Page 2) positioning - adjusted for better alignment
            name_field_x = 75
            name_field_y = 725
            ssn_field_x = 550
            ssn_field_y = 725
            checkbox_base_y = 587
            table_start_y = 465
        
        checkbox_x = 52
        
        # Enhanced column positions with better spacing
        col_positions = {
            'description': 50,
            'date_acquired': 195,
            'date_sold': 255,
            'proceeds': 330,
            'cost_basis': 400,
            'code': 455,
            'adjustment': 495,
            'gain_loss': 565
        }
        
        # Row spacing to match form's ruled line spacing
        row_height = 24.0
        
        # Fill taxpayer information with better font handling
        c.setFont("Helvetica", 10)
        c.drawString(name_field_x, name_field_y, taxpayer_name[:50])  # Increased character limit
        c.drawRightString(ssn_field_x, ssn_field_y, taxpayer_ssn)
        
        # Check appropriate box with better positioning
        c.setFont("Helvetica", 12)
        box_letter = box_type.split()[1]  # Extract A, B, or C
        
        if part_type == "Part I":
            if box_letter == "A":
                c.drawString(checkbox_x, checkbox_base_y, "✓")
            elif box_letter == "B": 
                c.drawString(checkbox_x, checkbox_base_y - 20, "✓")
            elif box_letter == "C":
                c.drawString(checkbox_x, checkbox_base_y - 40, "✓")
        else:  # Part II - maps A->D, B->E, C->F
            if box_letter == "A":  # Maps to Box D for long-term
                c.drawString(checkbox_x, checkbox_base_y, "✓")
            elif box_letter == "B":  # Maps to Box E for long-term
                c.drawString(checkbox_x, checkbox_base_y - 20, "✓")
            elif box_letter == "C":  # Maps to Box F for long-term
                c.drawString(checkbox_x, checkbox_base_y - 40, "✓")
        
        # Enhanced font size for better readability
        c.setFont("Helvetica", 6)
        
        # Fill transaction data with enhanced formatting
        for i, transaction in enumerate(transactions[:14]):  # Maximum 14 transactions per page
            y_pos = table_start_y - (i * row_height)
            
            # Enhanced description formatting
            description = transaction['description'][:25]  # Increased length
            date_acquired = transaction['date_acquired'].strftime('%m/%d/%Y')
            date_sold = transaction['date_sold'].strftime('%m/%d/%Y')
            
            # Column (a) - Description: Left-aligned with better truncation
            c.drawString(col_positions['description'], y_pos, description)
            
            # Column (b) - Date acquired: Centered with better positioning
            date_acq_width = c.stringWidth(date_acquired)
            c.drawString(col_positions['date_acquired'] - date_acq_width/2, y_pos, date_acquired)
            
            # Column (c) - Date sold: Centered with better positioning
            date_sold_width = c.stringWidth(date_sold)
            c.drawString(col_positions['date_sold'] - date_sold_width/2, y_pos, date_sold)
            
            # Enhanced number formatting with better truncation logic
            proceeds_text = format_currency_for_form(transaction['proceeds'])
            c.drawRightString(col_positions['proceeds'], y_pos, proceeds_text)
            
            basis_text = format_currency_for_form(transaction['cost_basis'])
            c.drawRightString(col_positions['cost_basis'], y_pos, basis_text)
            
            # Column (f) - Code: Leave blank for crypto transactions
            
            # Column (g) - Adjustment: Leave blank (no adjustments for crypto)
            
            # Column (h) - Gain/Loss with enhanced formatting
            gain_loss = transaction['gain_loss']
            gain_loss_text = format_currency_for_form(gain_loss, show_parentheses=True)
            c.drawRightString(col_positions['gain_loss'], y_pos, gain_loss_text)
        
        # Enhanced totals formatting on final page
        if page_num == total_pages and len(transactions) > 0:
            totals_y = table_start_y - (14 * row_height) - 15
            
            # Calculate totals for ALL transactions
            total_proceeds = sum(t['proceeds'] for t in all_transactions)
            total_basis = sum(t['cost_basis'] for t in all_transactions)
            total_gain_loss = sum(t['gain_loss'] for t in all_transactions)
            
            # Use bold font for totals
            c.setFont("Helvetica-Bold", 6)
            
            # Draw totals with enhanced formatting
            c.drawRightString(col_positions['proceeds'], totals_y, format_currency_for_form(total_proceeds))
            c.drawRightString(col_positions['cost_basis'], totals_y, format_currency_for_form(total_basis))
            c.drawRightString(col_positions['gain_loss'], totals_y, format_currency_for_form(total_gain_loss, show_parentheses=True))
        
        c.save()
        
        # Enhanced PDF merging
        overlay_buffer.seek(0)
        overlay_reader = PyPDF2.PdfReader(overlay_buffer)
        overlay_page = overlay_reader.pages[0]
        
        # Combine template and data overlay
        template_page.merge_page(overlay_page)
        
        # Write final PDF to buffer with enhanced settings
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(template_page)
        
        # Add metadata
        pdf_writer.add_metadata({
            '/Title': f'Form 8949 {tax_year} - {part_type}',
            '/Author': 'Bitwave Form Generator',
            '/Subject': f'Tax Form 8949 for {taxpayer_name}',
            '/Creator': 'Enhanced Bitwave PDF Generator'
        })
        
        pdf_writer.write(buffer)
        return True
        
    except Exception as e:
        st.error(f"Error creating enhanced official template: {e}")
        return False

def get_official_form_8949_enhanced(tax_year):
    """Enhanced version with better error handling and fallbacks"""
    irs_urls = {
        2024: "https://www.irs.gov/pub/irs-pdf/f8949.pdf",
        2023: "https://www.irs.gov/pub/irs-prior/f8949--2023.pdf", 
        2022: "https://www.irs.gov/pub/irs-prior/f8949--2022.pdf",
        2021: "https://www.irs.gov/pub/irs-prior/f8949--2021.pdf",
        2020: "https://www.irs.gov/pub/irs-prior/f8949--2020.pdf"
    }
    
    url = irs_urls.get(tax_year, irs_urls[2024])
    
    try:
        # Enhanced request with better headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=30, headers=headers)
        if response.status_code == 200 and response.content:
            # Validate that we got a PDF
            if response.content.startswith(b'%PDF'):
                return response.content
            else:
                st.warning("Downloaded file is not a valid PDF")
        else:
            st.warning(f"HTTP {response.status_code}: Could not download form from IRS website")
    except requests.exceptions.Timeout:
        st.warning("Timeout downloading official form - will use custom template")
    except Exception as e:
        st.warning(f"Could not download official form: {e}")
    
    return None

def format_currency_for_form(amount, show_parentheses=False):
    """Enhanced currency formatting for form fields"""
    if amount == 0:
        return "0.00"
    
    abs_amount = abs(amount)
    
    # Format with appropriate precision
    if abs_amount >= 1000000:  # Millions
        formatted = f"{abs_amount:,.0f}"
    elif abs_amount >= 1000:   # Thousands
        formatted = f"{abs_amount:,.2f}"
    else:                      # Under 1000
        formatted = f"{abs_amount:.2f}"
    
    # Handle negative values
    if amount < 0 and show_parentheses:
        return f"({formatted})"
    elif amount < 0:
        return f"-{formatted}"
    else:
        return formatted

def create_professional_custom_form(buffer, transactions, part_type, taxpayer_name, taxpayer_ssn, tax_year, box_type, page_num, total_pages, all_transactions):
    """Create a professional custom form with enhanced layout"""
    try:
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Enhanced header design
        c.setFillColor(colors.darkblue)
        c.rect(40, height - 80, width - 80, 50, fill=True, stroke=False)
        
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredText(width/2, height - 55, f"Form 8949 ({tax_year})")
        c.setFont("Helvetica", 12)
        c.drawCentredText(width/2, height - 70, "Sales and Other Dispositions of Capital Assets")
        
        # Reset color for rest of form
        c.setFillColor(colors.black)
        
        # Enhanced source attribution
        c.setFont("Helvetica", 8)
        c.drawString(50, height - 90, "Generated from Bitwave Actions Report")
        c.drawRightString(width - 50, height - 90, f"Generated: {datetime.now().strftime('%m/%d/%Y %I:%M %p')}")
        
        # Enhanced taxpayer information section
        c.setFillColor(colors.lightgrey)
        c.rect(40, height - 130, width - 80, 30, fill=True, stroke=True)
        c.setFillColor(colors.black)
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, height - 115, f"Taxpayer: {taxpayer_name}")
        c.drawString(350, height - 115, f"SSN: {taxpayer_ssn}")
        c.drawString(50, height - 125, f"Tax Year: {tax_year}")
        c.drawString(350, height - 125, f"Page: {page_num} of {total_pages}")
        
        # Enhanced part header
        c.setFont("Helvetica-Bold", 14)
        if part_type == "Part I":
            c.drawString(50, height - 160, "Part I - Short-Term Capital Gains and Losses")
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 175, "Assets held one year or less")
        else:
            c.drawString(50, height - 160, "Part II - Long-Term Capital Gains and Losses")
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 175, "Assets held more than one year")
        
        # Box type with enhanced styling
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, height - 195, f"☑ {box_type}")
        
        # Enhanced table with better spacing and alignment
        table_y_start = height - 230
        
        # Table headers with enhanced styling
        c.setFillColor(colors.darkblue)
        c.rect(40, table_y_start - 25, width - 80, 25, fill=True, stroke=True)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        
        headers = [
            ("Description of Property", 50, 120),
            ("Date Acquired", 170, 70), 
            ("Date Sold", 240, 70),
            ("Proceeds", 310, 80),
            ("Cost Basis", 390, 80),
            ("Code", 470, 40),
            ("Adjustment", 510, 70),
            ("Gain/(Loss)", 580, 80)
        ]
        
        for header, x_pos, width_col in headers:
            c.drawString(x_pos, table_y_start - 15, header)
        
        # Reset color for data rows
        c.setFillColor(colors.black)
        
        # Enhanced transaction data with alternating row colors
        row_height = 18
        c.setFont("Helvetica", 7)
        
        for i, transaction in enumerate(transactions[:14]):
            y_pos = table_y_start - 45 - (i * row_height)
            
            # Alternating row background
            if i % 2 == 0:
                c.setFillColor(colors.lightblue)
                c.rect(40, y_pos - 2, width - 80, row_height - 2, fill=True, stroke=False)
            
            c.setFillColor(colors.black)
            
            # Enhanced data formatting
            description = transaction['description'][:28]
            date_acquired = transaction['date_acquired'].strftime('%m/%d/%Y')
            date_sold = transaction['date_sold'].strftime('%m/%d/%Y')
            
            c.drawString(50, y_pos, description)
            c.drawCentredText(205, y_pos, date_acquired)
            c.drawCentredText(275, y_pos, date_sold)
            c.drawRightString(385, y_pos, format_currency_for_form(transaction['proceeds']))
            c.drawRightString(465, y_pos, format_currency_for_form(transaction['cost_basis']))
            c.drawCentredText(490, y_pos, "")  # Code column
            c.drawCentredText(545, y_pos, "")  # Adjustment column
            c.drawRightString(650, y_pos, format_currency_for_form(transaction['gain_loss'], show_parentheses=True))
        
        # Enhanced totals section on final page
        if page_num == total_pages and len(transactions) > 0:
            totals_y = table_y_start - 45 - (14 * row_height) - 10
            
            # Totals background
            c.setFillColor(colors.yellow)
            c.rect(40, totals_y - 2, width - 80, row_height - 2, fill=True, stroke=True)
            c.setFillColor(colors.black)
            
            # Calculate totals
            total_proceeds = sum(t['proceeds'] for t in all_transactions)
            total_basis = sum(t['cost_basis'] for t in all_transactions)
            total_gain_loss = sum(t['gain_loss'] for t in all_transactions)
            
            c.setFont("Helvetica-Bold", 8)
            c.drawString(50, totals_y, "TOTALS")
            c.drawRightString(385, totals_y, format_currency_for_form(total_proceeds))
            c.drawRightString(465, totals_y, format_currency_for_form(total_basis))
            c.drawRightString(650, totals_y, format_currency_for_form(total_gain_loss, show_parentheses=True))
        
        # Enhanced footer with summary
        footer_y = 60
        c.setFont("Helvetica", 8)
        
        page_summary = f"Page {page_num}: {len(transactions)} transactions"
        if page_num == total_pages:
            page_summary += f" | Total: {len(all_transactions)} transactions"
        
        c.drawString(50, footer_y, page_summary)
        c.drawRightString(width - 50, footer_y, f"Bitwave Form Generator - {datetime.now().strftime('%m/%d/%Y')}")
        
        # Add border around entire form
        c.setStrokeColor(colors.darkblue)
        c.setLineWidth(2)
        c.rect(30, 30, width - 60, height - 60, fill=False, stroke=True)
        
        c.save()
        return True
        
    except Exception as e:
        st.error(f"Error creating professional custom form: {e}")
        return False

def create_zip_file(pdf_files):
    """Create ZIP file containing all PDFs with enhanced metadata"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        # Add a readme file with information
        readme_content = f"""Form 8949 Generated from Bitwave Actions Report

Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Total PDF files: {len(pdf_files)}

Files included:
"""
        for pdf_file in pdf_files:
            readme_content += f"- {pdf_file['filename']}\n"
            zip_file.writestr(pdf_file['filename'], pdf_file['content'])
        
        readme_content += """
Instructions:
1. Review each PDF for accuracy
2. Print or save as needed for tax filing
3. Consult with a tax professional if you have questions

Generated by: Enhanced Bitwave Form 8949 Generator
"""
        
        zip_file.writestr("README.txt", readme_content)
    
    return zip_buffer.getvalue()

if __name__ == "__main__":
    main()
