import os

import streamlit as st
from rag_pipeline import answer_question
from users.auth_usage import (
    can_ask_question,
    grant_more_access,
    get_question_usage,
    normalize_email,
    request_verification_code,
    verify_email_code,
    record_question,
)

st.set_page_config(page_title="GDPR Assistant", page_icon="⚖️")

st.title("⚖️ GDPR Assistant")

st.caption("AI assistant for GDPR questions.")

# st.info(
#    "You can ask standalone questions or short follow-up questions. "
#    "The assistant rewrites follow-ups using recent chat context before searching."
# )

if "messages" not in st.session_state:
    st.session_state.messages = []


def is_admin_email(email):
    admin_email = normalize_email(os.getenv("ADMIN_EMAIL", ""))
    return bool(admin_email) and normalize_email(email) == admin_email


def render_admin_tools():
    st.sidebar.header("Admin")
    debug_enabled = st.sidebar.checkbox("Debug mode", value=False)

    with st.sidebar.form("grant_access_form"):
        email = st.text_input("User email")
        extra_questions = st.number_input(
            "Add questions",
            min_value=1,
            max_value=1000,
            value=3,
            step=1,
        )
        grant_access = st.form_submit_button("Add access")

    if grant_access:
        result = grant_more_access(email, int(extra_questions))
        if result["ok"]:
            st.sidebar.success(result["message"])
        else:
            st.sidebar.error(result["message"])

    return debug_enabled


def render_email_auth():
    if st.session_state.get("authenticated_email"):
        return st.session_state.authenticated_email

    st.subheader("Email access")

    with st.form("email_request_form"):
        email = st.text_input(
            "Email",
            value=st.session_state.get("pending_email", ""),
        )
        request_code = st.form_submit_button("Send code")

    if request_code:
        result = request_verification_code(email)
        if result["ok"]:
            st.session_state.pending_email = result["email"]
            st.session_state.dev_verification_code = result["dev_code"]
            st.success(result["message"])
        else:
            st.error(result["message"])

    pending_email = st.session_state.get("pending_email")
    if pending_email:
        st.caption(f"Code requested for {pending_email}")

        if st.session_state.get("dev_verification_code"):
            st.warning(
                "Local development mode only. "
                f"Verification code: {st.session_state.dev_verification_code}"
            )

        with st.form("email_code_form"):
            code = st.text_input("Code", max_chars=6)
            verify_code = st.form_submit_button("Verify code")

        if verify_code:
            verified, message = verify_email_code(pending_email, code)
            if verified:
                st.session_state.authenticated_email = pending_email
                st.session_state.messages = []
                st.session_state.dev_verification_code = None
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.stop()


authenticated_email = render_email_auth()
usage = get_question_usage(authenticated_email)
admin_mode = is_admin_email(authenticated_email)
debug_mode = render_admin_tools() if admin_mode else False

st.caption(
    f"Signed in as {authenticated_email}. "
    f"Questions remaining: {usage['remaining']} of {usage['limit']}."
)

if st.button("Use another email"):
    st.session_state.authenticated_email = None
    st.session_state.pending_email = None
    st.session_state.dev_verification_code = None
    st.session_state.messages = []
    st.rerun()

if st.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

with st.expander("Example questions"):
    st.markdown("""
- I would like to send marketing emails. What I shall consider?
- What are controller obligations when using a subcontractor, who processes personal data?
- What are GDPR requirements for video surveillance?
- What are common GDPR violations in employment sector?
""")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if usage["remaining"] <= 0:
    st.warning("To request additional access, please contact the administrator at qwqwbu@gmail.com. Briefly describe how you plan to use the application (e.g. for personal, business, or study purposes) and give a rough estimate of how many requests you expect to make. Thank you!")
    question = None
else:
    question = st.chat_input("Ask your question about GDPR")

if question:
    if not can_ask_question(authenticated_email):
        st.warning(
            "To request access, please contact the administrator at qwqwbu@gmail.com")
        st.stop()

    chat_history_before_current_question = st.session_state.messages.copy()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching GDPR guidance and enforcement practice..."):
            result = answer_question(
                query=question,
                chat_history=chat_history_before_current_question,
                debug=debug_mode,
            )

            st.markdown(result)

    usage = record_question(authenticated_email)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result,
    })

    if usage["remaining"] <= 0:
        st.warning("To request additional access, please contact the administrator at qwqwbu@gmail.com. Briefly describe how you plan to use the application (e.g. for personal, business, or study purposes) and give a rough estimate of how many requests you expect to make. Thank you!")
