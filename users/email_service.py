import os
import ssl
import smtplib
from email.message import EmailMessage


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _smtp_port():
    return int(os.getenv("SMTP_PORT", "587"))


def _smtp_uses_tls():
    return _env_flag("SMTP_USE_TLS", default=True)


def development_code_enabled():
    return _env_flag("DEV_SHOW_VERIFICATION_CODE", default=False)


def _smtp_configured():
    required_values = [
        os.getenv("SMTP_HOST"),
        os.getenv("SMTP_USER"),
        os.getenv("SMTP_PASSWORD"),
        os.getenv("EMAIL_FROM"),
    ]
    return all(required_values)


def send_verification_code(email, code):
    if not _smtp_configured():
        if development_code_enabled():
            return {"sent": False, "dev_code": code}
        raise RuntimeError("SMTP is not fully configured")

    message = EmailMessage()
    message["Subject"] = "Your GDPR Assistant access code"
    message["From"] = os.getenv("EMAIL_FROM")
    message["To"] = email
    message.set_content(
        "Your GDPR Assistant verification code is: "
        f"{code}\n\nThis code expires in 10 minutes."
    )

    with smtplib.SMTP(os.getenv("SMTP_HOST"), _smtp_port(), timeout=15) as smtp:
        if _smtp_uses_tls():
            smtp.starttls(context=ssl.create_default_context())

        smtp.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
        smtp.send_message(message)

    return {
        "sent": True,
        "dev_code": code if development_code_enabled() else None,
    }
