mkdir -p ~/.streamlit

echo "\
[general]\n\
email = \"email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

# create and activate environment
echo "Creating and activating environment"
conda create --name crypto python=3.7 jcopml matplotlib numpy openpyxl pandas Pillow plotly PyYAML scikit-learn seaborn sklearn streamlit yfinance -c conda-forge
conda activate crypto

# update TA-Lib
echo "Updating TA-Lib"
conda install -y -c conda-forge ta-lib

# run streamlit app
echo "Running streamlit app"
streamlit run main.py
