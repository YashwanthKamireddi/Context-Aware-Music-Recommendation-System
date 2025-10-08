# Clear certificate environment variables
$env:REQUESTS_CA_BUNDLE=$null
$env:CURL_CA_BUNDLE=$null
$env:SSL_CERT_FILE=$null

# Start the server with the correct Python from virtual environment
& "C:\Users\yashw\Context-Aware-Music-Recommendation-System\.venv\Scripts\python.exe" backend/server.py
