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

# create a new conda environment
conda create --name crypto python=3.7 -y

# activate the conda environment
conda activate crypto

# install TA-Lib library
conda install -c conda-forge ta-lib -y

# install dependencies from requirements.txt
pip install -r requirements.txt

# start the application
streamlit run main.py

