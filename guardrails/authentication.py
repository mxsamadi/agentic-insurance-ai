
import streamlit as st
import hmac


def require_authentication():

    if st.session_state.get("authenticated", False):

        with st.sidebar:

            if st.button("Log Out"):
                st.session_state.clear()
                st.rerun()

        return True

    st.title("🛡️ Agentic AI Insurance Platform")
    st.subheader("Secure Demo Access")

    if "DEMO_PASSWORD" not in st.secrets:
        st.error("Demo password is not configured.")
        st.stop()

    with st.form("login_form"):

        password = st.text_input(
            "Demo Access Password",
            type="password"
        )

        submitted = st.form_submit_button(
            "Access Platform",
            type="primary"
        )

    if submitted:

        if hmac.compare_digest(
            password.encode("utf-8"),
            str(st.secrets["DEMO_PASSWORD"]).encode("utf-8")
        ):

            st.session_state["authenticated"] = True
            st.rerun()

        else:
            st.error("Incorrect password.")

    return False
