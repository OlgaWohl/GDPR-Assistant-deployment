import csv
import io
import os

import streamlit as st
import streamlit.components.v1 as components
from rag_pipeline import answer_question
from users.auth_usage import (
    ANSWER_LENGTH_OPTIONS,
    ACCESS_REQUEST_SUBMITTED_MESSAGE,
    SATISFACTION_RATINGS,
    USE_AGAIN_OPTIONS,
    USER_ROLE_OPTIONS,
    can_ask_question,
    grant_more_access,
    get_question_usage,
    list_access_requests,
    list_feedback,
    normalize_email,
    request_verification_code,
    verify_email_code,
    record_question,
    submit_access_request,
)

st.set_page_config(page_title="GDPR Assistant", page_icon="⚖️")

st.title("GDPR Assistant")

# st.subheader(
#    "Your AI assistant for questions about the EU General Data Protection Regulation (GDPR)")

st.markdown(
    """
    <p style="font-size:18px; font-weight:600; margin-bottom:0.5rem;">
    Your AI assistant for questions about the EU General Data Protection Regulation (GDPR)
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown(

    """
  The Assistant:  
   
- ✅ Is based on the GDPR  and official EDPB / Article 29 Working Party guidance
- ✅ Includes insights from 3,000+ GDPR enforcement cases across Europe
- ✅ Combines legal requirements with real-world enforcement practice
- ✅ Provides clear, practical, and structured answers
"""
)

# st.info(
#    "You can ask standalone questions or short follow-up questions. "
#    "The assistant rewrites follow-ups using recent chat context before searching."
# )

if "messages" not in st.session_state:
    st.session_state.messages = []

ADMIN_CONTACT_EMAIL = "assistant.legalai@gmail.com"
REQUEST_MORE_ACCESS_LABEL = "Request more access"
ACCESS_REQUEST_FORM_ANCHOR = "access-request-form-anchor"
ACCESS_REQUEST_CONFIRMATION_ANCHOR = "access-request-confirmation-anchor"
FEEDBACK_COMMENT_PLACEHOLDER = (
    "Tell us what you found helpful, what could be improved, or what you "
    "would like the assistant to offer in future versions. All suggestions "
    "are welcome."
)


def scroll_to_latest_message():
    components.html(
        """
        <script>
        const scrollToBottom = () => {
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;
            const scrollingElement =
                parentDocument.scrollingElement ||
                parentDocument.documentElement ||
                parentDocument.body;

            parentWindow.requestAnimationFrame(() => {
                scrollingElement.scrollTo({
                    top: scrollingElement.scrollHeight,
                    behavior: "smooth"
                });
            });
        };

        setTimeout(scrollToBottom, 100);
        setTimeout(scrollToBottom, 400);
        </script>
        """,
        height=0,
    )


def render_scroll_anchor(anchor_id):
    st.markdown(
        f'<div id="{anchor_id}" style="scroll-margin-top: 80px;"></div>',
        unsafe_allow_html=True,
    )


def scroll_to_anchor(anchor_id):
    components.html(
        f"""
        <script>
        const scrollToAnchor = () => {{
            const parentDocument = window.parent.document;
            const anchor = parentDocument.getElementById("{anchor_id}");

            if (anchor) {{
                anchor.scrollIntoView({{
                    behavior: "smooth",
                    block: "start"
                }});
            }}
        }};

        setTimeout(scrollToAnchor, 100);
        setTimeout(scrollToAnchor, 400);
        </script>
        """,
        height=0,
    )


def is_admin_email(email):
    admin_email = normalize_email(os.getenv("ADMIN_EMAIL", ""))
    return bool(admin_email) and normalize_email(email) == admin_email


def build_feedback_csv(feedback_rows):
    if not feedback_rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=feedback_rows[0].keys())
    writer.writeheader()
    writer.writerows(feedback_rows)
    return output.getvalue()


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

    st.sidebar.subheader("Access requests")
    access_requests = list_access_requests()
    if access_requests:
        st.sidebar.dataframe(
            access_requests,
            column_order=("email", "purpose", "comment",
                          "created_at", "status"),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.sidebar.caption("No access requests yet.")

    feedback_rows = list_feedback()
    st.sidebar.download_button(
        "Download feedback as CSV",
        data=build_feedback_csv(feedback_rows),
        file_name="feedback_export.csv",
        mime="text/csv",
        disabled=not feedback_rows,
    )

    return debug_enabled


def render_access_request_flow(authenticated_email):
    st.warning("You have used all available questions for this account.")

    if st.button(REQUEST_MORE_ACCESS_LABEL):
        st.session_state.show_access_request_form = True
        st.session_state.scroll_to_access_form = True
        st.session_state.access_request_confirmation = None

    if st.session_state.get("show_access_request_form"):
        render_scroll_anchor(ACCESS_REQUEST_FORM_ANCHOR)

        st.text_input("Verified email",
                      value=authenticated_email, disabled=True)
        user_role = st.selectbox(
            "What best describes you?",
            USER_ROLE_OPTIONS,
            index=None,
            placeholder="Please select an option",
            key="access_request_user_role",
        )
        custom_role = ""
        if user_role == "Other":
            custom_role = st.text_input(
                "Please specify your role or background",
                max_chars=120,
                key="access_request_custom_role",
            )

        with st.form("access_request_form"):
            answer_length_rating = st.radio(
                "How would you rate the length of the answers?",
                ANSWER_LENGTH_OPTIONS,
                index=None,
            )
            satisfaction_rating = st.radio(
                "Overall, how satisfied were you with the answers?",
                SATISFACTION_RATINGS,
                format_func=lambda value: (
                    "1 = Not satisfied at all"
                    if value == 1
                    else "5 = Completely satisfied"
                    if value == 5
                    else str(value)
                ),
                index=None,
            )
            use_again = st.radio(
                "Would you use this assistant again?",
                USE_AGAIN_OPTIONS,
                index=None,
            )
            comments = st.text_area(
                "Comments",
                placeholder=FEEDBACK_COMMENT_PLACEHOLDER,
                max_chars=2000,
            )
            submit_request = st.form_submit_button("Submit request")

        if submit_request:
            st.session_state.access_request_confirmation = None
            result = submit_access_request(
                authenticated_email,
                user_role,
                custom_role,
                answer_length_rating,
                satisfaction_rating,
                use_again,
                comments,
            )
            if result["ok"]:
                message = result.get(
                    "message") or ACCESS_REQUEST_SUBMITTED_MESSAGE
                message_type = "success" if result.get("created") else "info"
                st.session_state.access_request_confirmation = {
                    "message": message,
                    "type": message_type,
                }
                st.session_state.scroll_to_access_confirmation = True
            else:
                st.error(result["message"])

        if st.session_state.get("access_request_confirmation"):
            render_scroll_anchor(ACCESS_REQUEST_CONFIRMATION_ANCHOR)
            confirmation = st.session_state.access_request_confirmation
            if confirmation["type"] == "success":
                st.success(confirmation["message"])
            else:
                st.info(confirmation["message"])

        st.caption(
            f"You can also contact the administrator at {ADMIN_CONTACT_EMAIL}")

        if st.session_state.get("scroll_to_access_confirmation"):
            scroll_to_anchor(ACCESS_REQUEST_CONFIRMATION_ANCHOR)
            st.session_state.scroll_to_access_confirmation = False
        elif st.session_state.get("scroll_to_access_form"):
            scroll_to_anchor(ACCESS_REQUEST_FORM_ANCHOR)
            st.session_state.scroll_to_access_form = False


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
    st.session_state.show_access_request_form = False
    st.session_state.scroll_to_access_form = False
    st.session_state.scroll_to_access_confirmation = False
    st.session_state.access_request_confirmation = None
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
    render_access_request_flow(authenticated_email)
    question = None
else:
    question = st.chat_input("Ask your question about GDPR")

if question:
    if not can_ask_question(authenticated_email):
        render_access_request_flow(authenticated_email)
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

    scroll_to_latest_message()

    if usage["remaining"] <= 0:
        render_access_request_flow(authenticated_email)
