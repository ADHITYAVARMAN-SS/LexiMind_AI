import os
import random
import time
import streamlit as st
import pandas as pd

from database import (
    init_db,
    load_words_from_csv,
    get_connection,
    record_attempt,
    get_due_words,
    update_schedule,
    get_total_correct,
    get_total_attempts,
    reset_database,
    get_daily_attempts,
    get_hard_words,
    get_average_response_time,
    get_mastered_words_count,
    get_semantic_distractors,
    generate_and_store_embeddings
)

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="LexiMind AI",
    layout="centered"
)

# ---------------------------
# LOAD DATA (CACHED)
# ---------------------------

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "..", "data", "vocab.csv")
    file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path)

    valid_df = df[
        df["word"].notna() &
        df["meaning"].notna() &
        (df["word"].astype(str).str.strip() != "") &
        (df["meaning"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    return valid_df


# ---------------------------
# INITIALIZATION
# ---------------------------

init_db()

valid_df = load_data()
load_words_from_csv(valid_df)

# Generate embeddings only once per session
if "embeddings_generated" not in st.session_state:
    generate_and_store_embeddings()
    st.session_state.embeddings_generated = True


# ---------------------------
# SESSION STATE
# ---------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "question_pool" not in st.session_state:
    st.session_state.question_pool = []

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "start_time" not in st.session_state:
    st.session_state.start_time = None


# ---------------------------
# HELPERS
# ---------------------------

def get_mistake_words():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT w.id, w.word, w.meaning
        FROM attempts a
        JOIN words w ON a.word_id = w.id
        WHERE a.correct = 0
        GROUP BY w.id
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def generate_question():
    if not st.session_state.question_pool:
        st.success("🎉 Session Completed!")
        st.session_state.current_question = None
        return

    word_id, word, meaning = st.session_state.question_pool.pop()

    # Semantic distractors
    wrong_options = get_semantic_distractors(word_id, limit=3)

    # Fallback
    if len(wrong_options) < 3:
        fallback_words = get_due_words()
        random_options = [w[2] for w in fallback_words if w[0] != word_id]
        random.shuffle(random_options)
        wrong_options += random_options[:3 - len(wrong_options)]

    options = wrong_options + [meaning]
    random.shuffle(options)

    st.session_state.current_question = {
        "word_id": word_id,
        "word": word,
        "correct": meaning,
        "options": options
    }

    st.session_state.start_time = time.time()


# =====================================================
# HOME PAGE
# =====================================================

if st.session_state.page == "home":

    st.title("🚀 LexiMind AI — Adaptive Vocabulary System")

    total_correct = get_total_correct()
    total_attempts = get_total_attempts()

    accuracy = round((total_correct / total_attempts) * 100, 2) if total_attempts > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("Total Correct", total_correct)
    col2.metric("Accuracy (%)", accuracy)

    st.markdown("---")

    if st.button("🧠 Practice Due Words"):
        words = get_due_words()

        if not words:
            st.info("🎉 No words due for review today!")
        else:
            random.shuffle(words)
            st.session_state.question_pool = words
            st.session_state.page = "practice"
            generate_question()
            st.rerun()

    if st.button("❌ Review Mistakes"):
        words = get_mistake_words()

        if not words:
            st.info("No mistakes recorded yet.")
        else:
            random.shuffle(words)
            st.session_state.question_pool = words
            st.session_state.page = "practice"
            generate_question()
            st.rerun()

    if st.button("📊 View Analytics"):
        st.session_state.page = "analytics"
        st.rerun()

    st.markdown("---")

    if st.button("🔄 Reset System"):
        reset_database()
        st.success("System reset successfully.")
        st.rerun()


# =====================================================
# PRACTICE PAGE
# =====================================================

elif st.session_state.page == "practice":

    if not st.session_state.current_question:
        generate_question()

    q = st.session_state.current_question

    if not q:
        if st.button("🏠 Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        st.stop()

    st.title("🧠 Practice Mode")
    st.markdown(f"## {q['word']}")

    selected_option = st.radio(
        "Choose the correct meaning:",
        q["options"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit"):

            response_time = time.time() - st.session_state.start_time
            correct = selected_option == q["correct"]

            record_attempt(q["word_id"], correct, response_time)
            update_schedule(q["word_id"], correct)

            if correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Correct Answer: {q['correct']}")

            generate_question()
            st.rerun()

    with col2:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.session_state.current_question = None
            st.rerun()


# =====================================================
# ANALYTICS PAGE
# =====================================================

elif st.session_state.page == "analytics":

    st.title("📊 Learning Analytics Dashboard")

    total_correct = get_total_correct()
    total_attempts = get_total_attempts()
    mastered = get_mastered_words_count()
    avg_time = get_average_response_time()

    accuracy = round((total_correct / total_attempts) * 100, 2) if total_attempts > 0 else 0

    col1, col2 = st.columns(2)

    col1.metric("Total Attempts", total_attempts)
    col2.metric("Accuracy (%)", accuracy)

    col1.metric("Mastered Words (≥3 reps)", mastered)
    col2.metric("Avg Response Time (sec)", avg_time)

    st.markdown("---")

    st.subheader("📈 Daily Activity")

    daily_data = get_daily_attempts()

    if daily_data:
        df = pd.DataFrame(daily_data, columns=["Date", "Attempts"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        st.line_chart(df)
    else:
        st.info("No activity data yet.")

    st.markdown("---")
    st.subheader("🔥 Hardest Words")

    hard_words = get_hard_words()

    if hard_words:
        for word, wrong_count in hard_words:
            st.write(f"• {word} — {wrong_count} mistakes")
    else:
        st.info("No mistakes recorded yet.")

    st.markdown("---")

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()