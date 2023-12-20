import streamlit as st
import time
import chatbot2 as chatbot

# Run the training first
@st.cache_data
def init_training():
    return chatbot.train()

model, input_shape, tokenizer, responses = init_training()
chatbt = chatbot.ChatBot(model, input_shape, tokenizer, responses)

st.title("🏖️ Ocean Bay")
st.header("Caribbean Seafood - Restaurant")
st.subheader('Support Assistant', divider=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbt.predict_class(prompt)

    if response == "Sorry! I didn't catch that.":
        print(f'FAILURE QUESTION: {prompt}')

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty()
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        response = response.format(name='Hello')
        message_placeholder.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
