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

#!/bin/bash

# Install TA-Lib
conda activate base
conda install -y -c conda-forge ta-lib

# Install other libraries using pip
pip install -r requirements.txt

# Run the application
streamlit run main.py

