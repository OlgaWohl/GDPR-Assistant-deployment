# GDPR RAG Chatbot

Deployment-ready Retrieval-Augmented Generation chatbot for GDPR questions.
It uses official regulatory guidance and European enforcement cases stored in
an included local Qdrant vector database.

## Deployment contents

This repository includes:

- the ready Qdrant collection in `qdrant_data/`;
- the RAG and answer-generation logic;
- the Streamlit frontend;
- email verification and question-limit logic.

Raw source PDFs and CSV files, notebooks, downloaded model files, and the
ingestion pipeline are intentionally not included. Do not run an ingestion
step on the server: the ready vector collection is used directly.

## Requirements

- Python 3.12 or newer
- Internet access for OpenAI API calls
- Disk space and a writable Hugging Face cache

The query embedding model is `BAAI/bge-base-en-v1.5`, matching the model used
to create the included Qdrant collection. Sentence Transformers may download
this model automatically on first startup, so the first start requires
internet access and can take longer.

## Server setup

```bash
git clone <repository-url>
cd <repository-directory>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```text
OPENAI_API_KEY=your_openai_api_key

SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USER=your_smtp_username
SMTP_PASSWORD=your_smtp_password
EMAIL_FROM=your_sender@example.com

ADMIN_EMAIL=admin@example.com
USERS_DATABASE_PATH=/home/ow/app-data/gdpr-users.db

# Local development only; omit or set false in production
DEV_SHOW_VERIFICATION_CODE=false
```

The SMTP variables enable delivery of login codes from a dedicated sender
account. Store all credentials in `.env` and never commit that file. The
implementation uses Python's standard-library `smtplib`; no additional email
package is required.

Create the persistent user-database directory and protect the environment
file:

```bash
mkdir -p /home/ow/app-data
chmod 700 /home/ow/app-data
chmod 600 .env
```

Start the application:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8200
```

Run this command from a user-level systemd service for automatic restart and
startup after server reboot. Set the service `WorkingDirectory` to the cloned
project directory and use the Streamlit executable from `.venv/bin/`.

## Access control

Visitors request a six-digit code by email before accessing the chat. Codes
expire after 10 minutes. Each verified email initially receives three
questions.

Set `ADMIN_EMAIL` to the administrator's email address. After that address is
verified through the same email-code flow, only that session can see the admin
panel, access-management controls, and debug mode. Public users never see
those controls or internal retrieval/debug information.

Verification state and usage counts are stored in SQLite at
`USERS_DATABASE_PATH`. Keep this database outside the Git checkout so user data
survives deployments.

`DEV_SHOW_VERIFICATION_CODE` defaults to false. Set it to `true` only for local
testing if you need the verification code shown in the UI. Keep it unset or
false in production. With the production default, missing or invalid SMTP
configuration never exposes verification codes in the UI.
