# Complete Crypto Actions to Form 8949 Converter

A comprehensive tool that automatically converts crypto trading reports into tax-ready formats. Upload your actions report and get either a CSV for tax software or a completed Form 8949 PDF for direct IRS filing.

## 🎯 What This Tool Does

**Complete End-to-End Solution:**
- 📂 Upload your crypto actions/transaction reports
- 🔄 Automatically extract and convert transaction data  
- 📊 Generate CSV files for tax software (TurboTax, TaxAct, etc.)
- 📄 Create completed Form 8949 PDFs ready for IRS mailing
- 🧮 Calculate all gains/losses automatically
- 📋 Handle multiple pages and complex transaction volumes

## 🚀 Step-by-Step Setup Guide

### Step 1: Create GitHub Repository
1. Go to [GitHub.com](https://github.com) and create an account
2. Click "New repository"
3. Name it: `crypto-form8949-converter`
4. Make it **Public**
5. Check "Add a README file"
6. Click "Create repository"

### Step 2: Upload Files
Upload these 3 files to your repository:

1. **app.py** - Main application (copy from first code box above)
2. **requirements.txt** - Dependencies (copy from second code box above)  
3. **README.md** - Instructions (copy from third code box above)

**Critical:** Save the main file as exactly `app.py`

### Step 3: Deploy to Streamlit
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Verify main file is `app.py`
6. Click "Deploy!"

### Step 4: Wait for Build (2-3 minutes)
Your app will be live at: `https://[username]-crypto-form8949-converter-app-streamlit.app`

## 📋 How to Use the Tool

### Step 1: Upload Your Data
**Supported file formats:**
- CSV files from exchanges (Coinbase, Binance, etc.)
- Excel files (.xlsx, .xls)
- Actions reports (like your original file)
- Tax software exports
- Custom transaction lists

### Step 2: Configure Settings
**In the sidebar:**
- **Tax Year:** Select 2022, 2023, etc.
- **Form Type:** Choose appropriate option:
  - **Part I (Box B)** - Most common for crypto (short-term, basis not reported)
  - **Part II (Box B)** - For long-term holdings (held >1 year)
- **Taxpayer Info:** Enter name and SSN (required for PDF generation)

### Step 3: Choose Output Format
**Two powerful options:**

**Option 1: CSV for Tax Software**
- Perfect for TurboTax, TaxAct, FreeTaxUSA, etc.
- Import directly into tax software
- Faster filing process
- Automatic calculations in tax software

**Option 2: Complete Form 8949 PDF**
- Official IRS form, completely filled out
- Print and mail directly to IRS
- No additional software needed
- Professional formatting

### Step 4: Download and File
- Download your generated files
- Follow provided instructions for your chosen method
- Keep copies for your records

## 📊 Supported File Formats

### Format 1: Actions Reports (Like Yours)
```
Row Labels, Sum of costBasisRelieved, Sum of proceeds
2022,,
BTC, 4287538.11, 4286732.78
ETH, 130607.70, 127257.06
ADA, 75473.34, 69969.52
```

### Format 2: Transaction Lists
```
Date, Asset, Type, Amount, Price, Total, Cost_Basis
2022-03-15, BTC, SELL, 1.5, 45000, 67500, 65000
2022-06-20, ETH, SELL, 10, 1800, 18000, 19000
```

### Format 3: Exchange Exports
```
Date, Pair, Side, Amount, Price, Fee, Total
2022-01-15, BTC/USD, SELL, 0.5, 42000, 25, 20975
2022-03-10, ETH/USD, SELL, 5, 2800, 15, 13985
```

## 🎯 Key Features

### Smart Data Detection
- **Auto-recognizes** different file formats
- **Extracts** crypto transactions automatically
- **Filters** by tax year
- **Calculates** gains/losses

### Professional Output
- **CSV files** compatible with all major tax software
- **PDF forms** that match official IRS Format 8949
- **Multiple pages** handled automatically
- **Proper formatting** for both digital and print filing

### Tax Compliance
- **Accurate calculations** for all gains/losses
- **Proper categorization** of short-term vs long-term
- **IRS-compliant formatting** for all outputs
- **Complete documentation** for audit purposes

## 💼 Perfect For

- **Individual investors** with crypto trading
- **Tax preparers** handling multiple clients
- **Day traders** with high transaction volumes
- **Anyone** who needs Form 8949 for crypto taxes

## 📖 Tax Software Instructions

### TurboTax
1. Federal Taxes → Wages & Income → Investment Income
2. Select "Stocks, Mutual Funds, Bonds, Other"
3. Choose "Import from CSV" 
4. Upload your downloaded CSV file
5. Review and continue

### TaxAct
1. Federal Return → Income → Investment Income
2. Select "Capital Gains and Losses"
3. Choose "Import transactions"
4. Upload CSV file
5. Verify data and proceed

### FreeTaxUSA
1. Income → Investment Income → Capital Gains/Losses
2. Select "Import from file"
3. Upload your CSV
4. Review imported transactions

### H&R Block
1. Federal → Income → Investment Income
2. Select "Capital Gains and Losses"
3. Choose "Import" option
4. Upload CSV file

## 📮 IRS Mailing Instructions

### For PDF Form 8949:
1. **Print** all pages on white paper
2. **Sign** your Form 1040
3. **Attach** Form 8949 to your tax return
4. **Include** Schedule D if required
5. **Mail** to your state's IRS processing center

### IRS Addresses by State:
- **California, Hawaii:** Fresno, CA 93888
- **New York, Connecticut:** Holtsville, NY 11742
- **Texas, New Mexico:** Austin, TX 73301
- **Florida, Georgia:** Atlanta, GA 39901
- **Check IRS.gov** for your specific state address

## 🔧 Advanced Features

### Multiple Tax Years
- Process different years separately
- Compare year-over-year performance
- Handle carryover losses properly

### Bulk Processing
- Handle thousands of transactions
- Automatic pagination for PDFs
- Optimized for large datasets

### Flexible Input
- Works with any CSV/Excel format
- Smart column detection
- Handles various date formats

## ⚠️ Important Tax Notes

### Cryptocurrency Tax Rules
- **Short-term:** Assets held ≤ 1 year (taxed as ordinary income)
- **Long-term:** Assets held > 1 year (preferential tax rates)
- **Like-kind exchanges:** Not applicable to crypto (post-2017)
- **Mining/staking:** Treated as ordinary income

### Record Keeping
- **Keep all transaction records** for at least 3 years
- **Document cost basis** for each purchase
- **Track dates carefully** for holding period determination
- **Save generated forms** for your records

### Professional Advice
- This tool helps with **form preparation only**
- **Consult a tax professional** for complex situations
- **Verify calculations** before filing
- **Keep detailed records** of all transactions

## 🆘 Troubleshooting

### File Upload Issues
- **Ensure file is CSV or Excel format**
- **Check for proper column headers**
- **Verify data contains transaction information**
- **Try different tax year if no data found**

### Data Extraction Problems
- **Review raw data preview** to verify format
- **Check that amounts are numeric** (no special characters)
- **Ensure dates include the target tax year**
- **Remove any summary/total rows** from your data

### PDF Generation Errors
- **Fill in taxpayer name and SSN** completely
- **Choose appropriate form type** for your situation
- **Verify all transaction data** is valid
- **Try generating CSV first** to test data quality

### Tax Software Import Issues
- **Use exact CSV format** generated by tool
- **Don't modify the CSV** after download
- **Follow software-specific** import instructions
- **Contact tax software support** if import fails

## 📁 File Structure
```
your-repository/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md          # This instruction file
```

## 🔄 Updates and Maintenance

### Annual Updates
- **Tax year selection** updated automatically
- **Form layouts** updated for IRS changes
- **Tax software compatibility** maintained

### Feature Requests
- **Modify app.py** to add new features
- **Update requirements.txt** for new libraries
- **Test thoroughly** before deploying changes

## 🎉 Success! You're Ready

Once deployed, your tool will:
1. **Accept any crypto transaction file**
2. **Automatically extract** relevant data
3. **Generate tax-ready outputs**
4. **Save hours** of manual data entry
5. **Ensure IRS compliance**

**Your live app:** `https://[username]-crypto-form8949-converter-app-streamlit.app`

---

**Disclaimer:** This tool is for educational and convenience purposes. Always consult with a qualified tax professional for tax advice and verify all calculations before filing.
