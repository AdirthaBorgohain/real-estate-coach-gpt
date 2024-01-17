import os
import streamlit as st
from streaming import StreamHandler
from analytics import analyze_conversation
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

if "analytics" not in st.session_state:
    st.session_state.analytics = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="Cold-Calling Simulation Bot for Scenario 1", page_icon="⭐")


def clear_chat_history():
    st.session_state.messages = []
    st.session_state.analytics = {}
    st.cache_resource.clear()


def generate_analytics():
    messages = st.session_state.messages
    conversation_text = ""
    for message in messages:
        if message['role'] == 'user':
            speaker = 'Agent'
        else:
            speaker = 'Home-owner'
        conversation_text += f"{speaker}: {message['content']}\n"
    st.session_state.messages = []
    response = analyze_conversation(conversation_text)
    st.session_state.analytics = response


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button('Start New Chat', on_click=clear_chat_history)

with col4:
    if not st.session_state.analytics:
        st.button('Get Analytics', on_click=generate_analytics)

st.header('Cold-Calling Simulation Bot for Scenario 1')
st.write('Talk with a real-estate owner to discuss details on his house and set-up an appointment')

if st.session_state.analytics:
    data = st.session_state.analytics
    st.markdown("## Conversation Analytics:")

    st.markdown("#### Was the Agent successful in setting up an appointment?")
    st.markdown(f"**{'Yes' if data['successful'] else 'No'}**")

    st.markdown("#### Understanding of Customer Needs:")
    st.markdown(f"{data['understanding_of_customer_needs']}")

    st.markdown("#### Proficiency in Real Estate Concepts: ")
    st.markdown(f"{data['proficiency_in_real_estate_concepts']}")

    st.markdown("#### Negotiation Skills: ")
    st.markdown(f"{data['negotiation_skills']}")

    st.markdown("#### Recommendations and Suggestions to improve: \n{}".format(
        '\n'.join('- {}'.format(recommendation) for recommendation in data['suggestions'])))

    st.markdown(f"#### Key Shift in the Conversation:")
    st.markdown(f"{data['key_shift']}")

with st.sidebar:
    st.title('💬 Chat Parameters')

    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        value=st.session_state['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in st.session_state else os.environ.get(
            'OPENAI_API_KEY', ''),
        placeholder="sk-..."
    )
    if openai_api_key:
        st.session_state['OPENAI_API_KEY'] = openai_api_key
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from this link: https://platform.openai.com/account/api-keys")
        st.stop()

    owner_name = st.text_input(
        label="Name of owner",
        placeholder="John Doe"
    )
    owner_nature = st.selectbox(
        label="Nature of Owner",
        options=[
            None,
            "Friendly and Welcoming",
            "Demanding or Perfectionist",
            "Reserved or Private",
            "Assertive or Business-Oriented",
            "Anxious or Nervous"])
    owner_description = st.selectbox(
        label="Description of the Owner",
        options=[
            None,
            "Approaches the sale with empathy, ensuring a positive experience for all parties involved.",
            "Takes a cautious approach to negotiations, seeking additional information and assurances.",
            "Is highly motivated and responsive, eager to engage in negotiations and excited about the process.",
            "May need more time for decision-making, seeking guidance and support during negotiations.",
            "Has a relaxed approach, not rushing negotiations or being overly proactive.",
            "Are meticulous, ensuring all transaction details are well-defined and met.",
            "Prioritizes trust and confidentiality, sharing property information selectively.",
            "Has a strong emotional ties to his property, influencing the negotiation process.",
            "Is confident with clear, non-negotiable terms, expecting an efficient process.",
            "Is reluctant to engage in negotiations and may be unresponsive, causing delays in the process and "
            "frustrating potential buyers and agents."
        ])
    negotiations_difficulty = st.selectbox(
        label="Negotiations Difficulty",
        options=[
            None,
            "Easy",
            "Moderate",
            "Tough"],
    )
    house_description = st.text_area("Description of the property", height=None)
    asking_price = st.number_input(label="Asking price of the property in USD", step=1, min_value=0, value=12000)
    asking_price_type = st.selectbox(
        "Asking price type",
        options=[
            None,
            "Cheap",
            "Below Average",
            "Average",
            "Above Average",
            "Exorbitant"])

if not st.session_state.analytics:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def generate_system_prompt():
    print("Generating new system prompt...")
    return (
        f"You are to role-play as {owner_name}, the owner of {house_description.lower()}. The property has a "
        f"partially open listing. You have set a asking price of {asking_price} USD that is deemed "
        f"{asking_price_type.lower()} considering the location of the property.\n"
        f"You are characterized as someone who is {owner_nature.lower()} and someone who {owner_description.lower()}."
        "You frequently find yourself approached by realtors who want to represent you, showcase your property, and "
        "arrange a sale with prospective buyers.\n"
        "You will be contacted by a real-estate agent over call to gather information about the property and may "
        "propose a viewing based on how the discussions between you and the agent go. If such a scenario arises, you "
        "are to decide whether you want to accept the proposal or not. You must always maintain a tough negotiating "
        f"stance. Usually negotiations with you are {negotiations_difficulty}. It is important that you do not "
        "anticipate calls from agents beforehand and do not share any information about your property unless "
        "specifically asked and only do so if you feel comfortable sharing based on the agent's behaviour and approach.\n"
        "If the agent manages to successfully schedule a tour and suggests a listing agreement, ensure you fully "
        "understand the process before giving consent. While maintaining your persona, be firm and persuasive when "
        "required but remember that the agent is trying to convince you, not the other way around. Always look to "
        "steer the negotiations to a direction where it benefits you the most.\n"
        "Your initial responses should be unenthusiastic and give an impression of disinterest.Display a cold attitude "
        "towards the caller initially, allowing the agent to make the first move. Always adhere to this instruction."
    )


def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)


@st.cache_resource
def setup_chain():
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(generate_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name='gpt-4', temperature=0, streaming=True)  # make model_name dynamic
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
    return chain


def main():
    chain = setup_chain()
    if not st.session_state.analytics:
        user_query = st.chat_input(placeholder=f"Talk to {owner_name}!")
        if user_query:
            display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = chain.run({"human_input": user_query}, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})


if all([owner_name, owner_nature, owner_description, negotiations_difficulty, house_description, asking_price,
        asking_price_type]):
    main()
elif not st.session_state.analytics:
    st.error("Please fill in all parameters in the sidebar to begin the chat.")
