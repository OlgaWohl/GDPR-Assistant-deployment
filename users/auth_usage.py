import hashlib
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

from users import database
from users.email_service import (
    send_access_request_notification,
    send_verification_code,
)


DEFAULT_QUESTIONS_PER_USER = database.DEFAULT_QUESTION_LIMIT
CODE_EXPIRY_MINUTES = 10
ACCESS_REQUEST_COOLDOWN_HOURS = 24
USER_ROLE_OPTIONS = (
    "Lawyer",
    "Data Protection Officer (DPO)",
    "Compliance professional",
    "Software developer",
    "Startup founder",
    "Student",
    "Private user",
    "Other",
)
ANSWER_LENGTH_OPTIONS = ("Too short", "About right", "Too long")
USE_AGAIN_OPTIONS = (
    "Definitely",
    "Probably",
    "Not sure",
    "Probably not",
    "Definitely not",
)
SATISFACTION_RATINGS = (1, 2, 3, 4, 5)
MAX_ACCESS_REQUEST_COMMENT_LENGTH = 1000
MAX_CUSTOM_ROLE_LENGTH = 120
MAX_FEEDBACK_COMMENT_LENGTH = 2000
ACCESS_REQUEST_SUBMITTED_MESSAGE = (
    "Your request has been submitted. The administrator will review it shortly."
)
ACCESS_REQUEST_RECENT_MESSAGE = (
    "You already submitted a request recently. "
    "The administrator will review it shortly."
)
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


def get_access_request_recipient():
    return normalize_email(
        os.getenv("ACCESS_REQUEST_EMAIL") or os.getenv("ADMIN_EMAIL", "")
    )


def _sanitize_access_request_comment(comment):
    sanitized = " ".join((comment or "").split())
    return sanitized[:MAX_ACCESS_REQUEST_COMMENT_LENGTH]


def _sanitize_short_text(value, max_length):
    sanitized = " ".join((value or "").split())
    return sanitized[:max_length]


def _sanitize_multiline_text(value, max_length):
    sanitized = (value or "").strip()
    return sanitized[:max_length]


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


def submit_access_request(
    email,
    user_role,
    custom_role,
    answer_length_rating,
    satisfaction_rating,
    use_again,
    comments,
):
    email = normalize_email(email)
    user_role = (user_role or "").strip()
    custom_role = _sanitize_short_text(custom_role, MAX_CUSTOM_ROLE_LENGTH)
    answer_length_rating = (answer_length_rating or "").strip()
    use_again = (use_again or "").strip()
    comments = _sanitize_multiline_text(comments, MAX_FEEDBACK_COMMENT_LENGTH)

    if not is_valid_email(email) or not is_verified(email):
        return {
            "ok": False,
            "created": False,
            "message": "Please verify your email before requesting more access.",
        }

    if user_role not in USER_ROLE_OPTIONS:
        return {
            "ok": False,
            "created": False,
            "message": "Please select what best describes you.",
        }

    if user_role == "Other" and not custom_role:
        return {
            "ok": False,
            "created": False,
            "message": "Please specify your role or background.",
        }

    if answer_length_rating not in ANSWER_LENGTH_OPTIONS:
        return {
            "ok": False,
            "created": False,
            "message": "Please rate the length of the answers.",
        }

    if satisfaction_rating not in SATISFACTION_RATINGS:
        return {
            "ok": False,
            "created": False,
            "message": "Please select an overall satisfaction rating.",
        }

    if use_again not in USE_AGAIN_OPTIONS:
        return {
            "ok": False,
            "created": False,
            "message": "Please tell us whether you would use this assistant again.",
        }

    feedback = database.create_feedback(
        email,
        user_role,
        custom_role if user_role == "Other" else "",
        answer_length_rating,
        satisfaction_rating,
        use_again,
        comments,
    )

    cooldown_start = (
        datetime.now(timezone.utc) - timedelta(hours=ACCESS_REQUEST_COOLDOWN_HOURS)
    ).isoformat()
    recent_request = database.get_recent_access_request(email, cooldown_start)
    if recent_request:
        return {
            "ok": True,
            "created": False,
            "message": ACCESS_REQUEST_RECENT_MESSAGE,
        }

    role_for_request = custom_role if user_role == "Other" else user_role
    purpose = "Additional access request"
    request_comment = _sanitize_access_request_comment(
        " | ".join(
            value
            for value in (
                f"Feedback ID: {feedback['id']}",
                f"Role: {role_for_request}",
                f"Answer length: {answer_length_rating}",
                f"Satisfaction: {satisfaction_rating}/5",
                f"Use again: {use_again}",
                f"Comments: {comments}" if comments else "",
            )
            if value
        )
    )

    access_request = database.create_access_request(email, purpose, request_comment)
    recipient = get_access_request_recipient()

    if not recipient:
        logger.warning("Access request notification skipped: no recipient configured")
        database.update_access_request_email_status(
            access_request["id"],
            "not_configured",
        )
        return {
            "ok": True,
            "created": True,
            "message": ACCESS_REQUEST_SUBMITTED_MESSAGE,
        }

    try:
        send_access_request_notification(
            recipient,
            access_request["email"],
            access_request["purpose"],
            access_request["comment"],
            access_request["created_at"],
        )
    except Exception:
        logger.exception("Access request email delivery failed")
        database.update_access_request_email_status(access_request["id"], "failed")
    else:
        database.update_access_request_email_status(access_request["id"], "sent")

    return {
        "ok": True,
        "created": True,
        "message": ACCESS_REQUEST_SUBMITTED_MESSAGE,
    }


def list_access_requests(limit=100):
    return [dict(row) for row in database.list_access_requests(limit)]


def list_feedback():
    return [dict(row) for row in database.list_feedback()]
