import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

grammar_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a language teacher. The user is learning {target_language}. "
               "First, check if the sentence has any grammar errors. "
               "If the sentence is correct, say 'Your sentence is grammatically correct!' and give one brief compliment. "
               "If there are errors, identify each one, explain briefly, and provide the corrected version. "
               "Do NOT invent errors that do not exist.\n\n"
               "Past errors this session: {error_history}\n"
               "If the current error matches a past one, start your response with "
               "'You made this mistake before: [error]. Let's fix it again!'"),
    ("human", "{transcript}")
])

vocabulary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a language teacher. The user is learning {target_language}. "
               "Suggest 2-3 better or more natural word choices for their sentence. "
               "Keep suggestions concise.\n\n"
               "Past errors this session: {error_history}\n"
               "If the user is repeating a weak vocabulary pattern from before, point it out gently."),
    ("human", "{transcript}")
])

def grammar_node(state: dict) -> dict:
    error_history = state.get("error_history", [])
    chain = grammar_prompt | llm
    result = chain.invoke({
        "transcript": state["transcript"],
        "target_language": state.get("target_language", "English"),
        "error_history": ", ".join(error_history) if error_history else "none yet"
    })
    feedback = result.content
    # extract new errors into history
    if "grammatically correct" not in feedback.lower():
        error_history = error_history + [state["transcript"]]
    return {**state, "result": feedback, "task": "grammar", "error_history": error_history}

def vocabulary_node(state: dict) -> dict:
    error_history = state.get("error_history", [])
    chain = vocabulary_prompt | llm
    result = chain.invoke({
        "transcript": state["transcript"],
        "target_language": state.get("target_language", "English"),
        "error_history": ", ".join(error_history) if error_history else "none yet"
    })
    return {**state, "result": result.content, "task": "vocabulary", "error_history": error_history}

def pronunciation_node(state: dict) -> dict:
    from agent.pronunciation import assess_pronunciation, format_pronunciation_feedback
    audio_path = state.get("audio_path", "")
    transcript = state.get("transcript", "")
    lang_map = {"English": "en-US", "German": "de-DE", "Japanese": "ja-JP", "Mandarin Chinese": "zh-CN"}
    lang_code = lang_map.get(state.get("target_language", "English"), "en-US")
    if not audio_path:
        return {**state, "result": "No audio file available for pronunciation assessment.", "task": "pronunciation"}
    scores = assess_pronunciation(audio_path, transcript, lang_code)
    feedback = format_pronunciation_feedback(scores)
    return {**state, "result": feedback, "task": "pronunciation"}

def rag_node(state: dict) -> dict:
    from agent.rag import answer_grammar_question
    answer = answer_grammar_question(state["transcript"])
    return {**state, "result": answer, "task": "knowledge_query"}

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are classifying a language learner's sentence. "
               "Decide what kind of feedback would be MOST useful:\n"
               "- grammar: the sentence has a clear grammatical error\n"
               "- vocabulary: the sentence is correct but word choices are unnatural or repetitive\n"
               "- pronunciation: the user is asking about how to pronounce something\n"
               "- knowledge_query: the user is asking a grammar or language question (e.g. 'when do I use...', 'what is...', 'how do I...')\n"
               "If the sentence is correct and natural, reply 'vocabulary'.\n"
               "Reply with only one word: grammar, vocabulary, pronunciation, or knowledge_query."),
    ("human", "{transcript}")
])

def router_node(state: dict) -> dict:
    chain = router_prompt | llm
    result = chain.invoke({"transcript": state["transcript"]})
    task = result.content.strip().lower()
    if task not in ("grammar", "vocabulary", "pronunciation", "knowledge_query"):
        task = "grammar"
    return {**state, "task": task}
