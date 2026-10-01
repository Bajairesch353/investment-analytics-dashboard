import streamlit as st


def render(newsletter_path):
    st.subheader("Daily Newsletter")

    newsletter_files = sorted(
        newsletter_path.glob("newsletter_*.md"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if newsletter_files:
        selected_newsletter = st.selectbox(
            "Select newsletter",
            newsletter_files,
            format_func=lambda x: x.name
        )

        newsletter_text = selected_newsletter.read_text(encoding="utf-8")
        st.markdown(newsletter_text)

        html_version = selected_newsletter.with_suffix(".html")

        if html_version.exists():
            st.download_button(
                "Download HTML version",
                html_version.read_text(encoding="utf-8"),
                file_name=html_version.name,
                mime="text/html"
            )
    else:
        st.info(f"No newsletter files found in: {newsletter_path}")
