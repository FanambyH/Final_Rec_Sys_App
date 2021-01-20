mkdir -p ~/.streamlit/
echo "[general]
email = \"fanamby@dsi-program.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
