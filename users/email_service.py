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


def send_email(to_email, subject, body):
    if not _smtp_configured():
        raise RuntimeError("SMTP is not fully configured")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = os.getenv("EMAIL_FROM")
    message["To"] = to_email
    message.set_content(body)

    with smtplib.SMTP(os.getenv("SMTP_HOST"), _smtp_port(), timeout=15) as smtp:
        if _smtp_uses_tls():
            smtp.starttls(context=ssl.create_default_context())

        smtp.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
        smtp.send_message(message)

    return {"sent": True}


def send_verification_code(email, code):
    if not _smtp_configured():
        if development_code_enabled():
            return {"sent": False, "dev_code": code}
        raise RuntimeError("SMTP is not fully configured")

    result = send_email(
        email,
        "Your GDPR Assistant access code",
        "Your GDPR Assistant verification code is: "
        f"{code}\n\nThis code expires in 10 minutes.",
    )

    return {
        "sent": result["sent"],
        "dev_code": code if development_code_enabled() else None,
    }


def send_access_request_notification(to_email, user_email, purpose, comment, created_at):
    comment_text = comment or "No additional comments."
    body = (
        "A GDPR Assistant user requested more access.\n\n"
        f"User email: {user_email}\n"
        f"Purpose: {purpose}\n"
        f"Submitted at: {created_at}\n\n"
        f"Comment:\n{comment_text}\n"
    )
    return send_email(
        to_email,
        "New GDPR Assistant access request",
        body,
    )


def send_access_granted_notification(to_email, questions_remaining, question_limit):
    body = (
        "Your GDPR Assistant access has been updated.\n\n"
        f"You now have {questions_remaining} questions remaining "
        f"out of a total limit of {question_limit}.\n\n"
        "You can return to GDPR Assistant and continue using your verified email."
    )
    return send_email(
        to_email,
        "Your GDPR Assistant access has been updated",
        body,
    )
