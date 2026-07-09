import hashlib
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from users import database
from users.email_service import send_verification_code


DEFAULT_QUESTIONS_PER_USER = database.DEFAULT_QUESTION_LIMIT
CODE_EXPIRY_MINUTES = 10
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
VERIFICATION_REQUEST_MESSAGE = (
    "If this email is allowed, a verification code has been sent."
)
VERIFICATION_FAILURE_MESSAGE = (
    "The code is invalid or expired. Please request a new code."
)
logger = logging.getLogger(__name__)


def normalize_email(email):
    return email.strip().lower()


def is_valid_email(email):
    return bool(EMAIL_PATTERN.match(normalize_email(email)))


def _new_code():
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_code(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def request_verification_code(email):
    email = normalize_email(email)

    if not is_valid_email(email):
        return {
            "ok": False,
            "email": email,
            "message": "Please enter a valid email address.",
            "dev_code": None,
        }

    code = _new_code()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(minutes=CODE_EXPIRY_MINUTES)
    ).isoformat()

    database.upsert_verification_code(email, _hash_code(code), expires_at)

    try:
        email_result = send_verification_code(email, code)
    except Exception:
        logger.exception("Verification email delivery failed")
        email_result = {"dev_code": None}

    return {
        "ok": True,
        "email": email,
        "message": VERIFICATION_REQUEST_MESSAGE,
        "dev_code": email_result["dev_code"],
    }


def verify_email_code(email, code):
    email = normalize_email(email)
    code = code.strip()
    user = database.get_user(email)

    if not user or not user["verification_code"]:
        return False, VERIFICATION_FAILURE_MESSAGE

    expires_at = datetime.fromisoformat(user["code_expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return False, VERIFICATION_FAILURE_MESSAGE

    if not secrets.compare_digest(user["verification_code"], _hash_code(code)):
        return False, VERIFICATION_FAILURE_MESSAGE

    database.mark_user_verified(email)
    return True, "Email verified."


def is_verified(email):
    user = database.get_user(normalize_email(email))
    return bool(user and user["verified_at"])


def get_question_usage(email):
    user = database.get_user(normalize_email(email))
    used = user["question_count"] if user else 0
    limit = user["question_limit"] if user else DEFAULT_QUESTIONS_PER_USER
    remaining = max(limit - used, 0)

    return {
        "used": used,
        "remaining": remaining,
        "limit": limit,
    }


def can_ask_question(email):
    if not is_verified(email):
        return False

    return get_question_usage(email)["remaining"] > 0


def record_question(email):
    question_count = database.increment_question_count(normalize_email(email))
    usage = get_question_usage(email)

    return {
        "used": question_count,
        "remaining": usage["remaining"],
        "limit": usage["limit"],
    }


def grant_more_access(email, extra_questions):
    email = normalize_email(email)

    if not is_valid_email(email):
        return {
            "ok": False,
            "message": "Please enter a valid email address.",
            "usage": None,
        }

    if extra_questions < 1:
        return {
            "ok": False,
            "message": "Please add at least 1 question.",
            "usage": None,
        }

    user = database.grant_extra_questions(email, extra_questions)
    remaining = max(user["question_limit"] - user["question_count"], 0)

    return {
        "ok": True,
        "message": (
            f"Access updated for {email}. "
            f"Questions remaining: {remaining} of {user['question_limit']}."
        ),
        "usage": {
            "used": user["question_count"],
            "remaining": remaining,
            "limit": user["question_limit"],
        },
    }
