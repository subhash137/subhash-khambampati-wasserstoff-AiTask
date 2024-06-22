import streamlit as st
import requests
import base64
import json

def create_wp_connection(username, password):
    wp_connection = f"{username}:{password}"
    token = base64.b64encode(wp_connection.encode()).decode('utf-8')
    return {
        'Authorization': f'Basic {token}'
    }

import json

def get_posts(url, headers):
    api_url = f"{url}"
    try:
        response = requests.get(api_url, headers=headers)
        st.write(f"Status Code: {response.status_code}")
        st.write("Response Headers:")
        st.json(dict(response.headers))
        
        st.write("Response Content:")
        st.code(response.text)
        
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            st.error("The response is not JSON. Check the URL and API configuration.")
            return None

    except requests.exceptions.RequestException as err:
        st.error(f"An error occurred: {err}")
    return None

def main():
    st.title("WordPress Posts Viewer")

    # User input
    url = st.text_input("Enter your WordPress REST API URL (e.g., https://www.example.com/wp-json/wp/v2)")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")

    if st.button("Fetch Posts"):
        if url and username and password:
            with st.spinner("Fetching posts..."):
                headers = create_wp_connection(username, password)
                posts = get_posts(url, headers)

                if posts:
                    if len(posts) > 0:
                        st.success(f"Found {len(posts)} posts")
                        for post in posts:
                            st.subheader(post['title']['rendered'])
                            st.write(post['content']['rendered'], unsafe_allow_html=True)
                            st.write("---")
                    else:
                        st.info("No posts found.")
        else:
            st.warning("Please fill in all fields.")

    st.markdown("""
    ### Troubleshooting Tips:
    1. Ensure your WordPress site has the REST API enabled.
    2. Verify that you're using the correct REST API URL. It should end with `/wp-json/wp/v2`.
    3. Check if you have the necessary permissions to access the API.
    4. Try accessing the API endpoint directly in your browser to see if it's publicly accessible.
    """)

if __name__ == "__main__":
    main()