import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "users.db"
DEFAULT_QUESTION_LIMIT = 3


def get_database_path():
    return Path(os.getenv("USERS_DATABASE_PATH", DEFAULT_DATABASE_PATH))


@contextmanager
def get_connection():
    database_path = get_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                verification_code TEXT,
                code_expires_at TEXT,
                verified_at TEXT,
                question_count INTEGER NOT NULL DEFAULT 0,
                question_limit INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        if "question_limit" not in columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN question_limit INTEGER NOT NULL DEFAULT 3"
            )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS access_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                purpose TEXT NOT NULL,
                comment TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                email_delivery_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_access_requests_email_created_at
            ON access_requests(email, created_at)
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                email TEXT NOT NULL,
                user_role TEXT NOT NULL,
                custom_role TEXT,
                answer_length_rating TEXT NOT NULL,
                satisfaction_rating INTEGER NOT NULL,
                use_again TEXT NOT NULL,
                comments TEXT
            )
            """
        )


def upsert_verification_code(email, code, expires_at):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (
                email,
                verification_code,
                code_expires_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                verification_code = excluded.verification_code,
                code_expires_at = excluded.code_expires_at,
                updated_at = excluded.updated_at
            """,
            (email, code, expires_at, now, now),
        )


def get_user(email):
    init_db()

    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()


def mark_user_verified(email):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE users
            SET
                verified_at = ?,
                verification_code = NULL,
                code_expires_at = NULL,
                updated_at = ?
            WHERE email = ?
            """,
            (now, now, email),
        )


def increment_question_count(email):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE users
            SET
                question_count = question_count + 1,
                updated_at = ?
            WHERE email = ?
            """,
            (now, email),
        )

        return connection.execute(
            "SELECT question_count FROM users WHERE email = ?",
            (email,),
        ).fetchone()["question_count"]


def grant_extra_questions(email, extra_questions):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        user = connection.execute(
            "SELECT email FROM users WHERE email = ?",
            (email,),
        ).fetchone()

        if user:
            connection.execute(
                """
                UPDATE users
                SET
                    question_limit = question_limit + ?,
                    updated_at = ?
                WHERE email = ?
                """,
                (extra_questions, now, email),
            )
        else:
            connection.execute(
                """
                INSERT INTO users (
                    email,
                    question_limit,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (email, DEFAULT_QUESTION_LIMIT + extra_questions, now, now),
            )

        return connection.execute(
            """
            SELECT email, question_count, question_limit
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()


def get_recent_access_request(email, since):
    init_db()

    with get_connection() as connection:
        return connection.execute(
            """
            SELECT *
            FROM access_requests
            WHERE email = ?
              AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email, since),
        ).fetchone()


def create_access_request(email, purpose, comment):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO access_requests (
                email,
                purpose,
                comment,
                status,
                email_delivery_status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, 'pending', 'pending', ?, ?)
            """,
            (email, purpose, comment, now, now),
        )
        return connection.execute(
            "SELECT * FROM access_requests WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()


def update_access_request_email_status(request_id, email_delivery_status):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE access_requests
            SET
                email_delivery_status = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (email_delivery_status, now, request_id),
        )


def list_access_requests(limit=100):
    init_db()

    with get_connection() as connection:
        return connection.execute(
            """
            SELECT email, purpose, comment, created_at, status
            FROM access_requests
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def create_feedback(
    email,
    user_role,
    custom_role,
    answer_length_rating,
    satisfaction_rating,
    use_again,
    comments,
):
    init_db()
    now = utc_now_iso()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO feedback (
                created_at,
                email,
                user_role,
                custom_role,
                answer_length_rating,
                satisfaction_rating,
                use_again,
                comments
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                email,
                user_role,
                custom_role,
                answer_length_rating,
                satisfaction_rating,
                use_again,
                comments,
            ),
        )
        return connection.execute(
            "SELECT * FROM feedback WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()


def list_feedback():
    init_db()

    with get_connection() as connection:
        return connection.execute(
            """
            SELECT
                id,
                created_at,
                email,
                user_role,
                custom_role,
                answer_length_rating,
                satisfaction_rating,
                use_again,
                comments
            FROM feedback
            ORDER BY created_at DESC
            """
        ).fetchall()
