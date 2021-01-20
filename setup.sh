  GNU nano 5.3                                                                                         setup.sh                                                                                         Modified
mkdir -p ~/.streamlit/
echo "[general]
email = \"fanamby@dsi-program.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
