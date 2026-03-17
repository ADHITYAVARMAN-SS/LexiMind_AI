import os
import random
import time
import streamlit as st
import pandas as pd

from database import (
    init_db,
    load_words_from_csv,
    record_attempt,
    get_due_words,
    update_schedule,
    update_difficulty,
    get_home_page_data,
    get_analytics_page_data,
    get_all_time_stats,
    update_all_time_stats,
    get_random_words,
    get_mistake_words,
    get_word_by_id,
    search_words,
    get_word_history,
    reset_database,
    get_semantic_distractors,
    generate_and_store_embeddings,
)

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(page_title="LexiMind AI", layout="centered")

# ---------------------------
# LOAD DATA  (cached, with CSV row-repair)
# ---------------------------

@st.cache_data
def load_data():
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "vocab.csv"))
    df        = pd.read_csv(file_path)

    # Repair split rows: some meanings overflow onto the next CSV line
    # (second row has NaN/blank word, continuation text in meaning column)
    fixed = []
    for _, row in df.iterrows():
        word    = str(row.get("word",    "")).strip()
        meaning = str(row.get("meaning", "")).strip()

        # Continuation row — blank word, meaning is the rest of the previous entry
        if word in ("", "nan") and fixed:
            if meaning not in ("", "nan"):
                fixed[-1]["meaning"] = fixed[-1]["meaning"].rstrip() + " " + meaning
        else:
            if word not in ("", "nan") and meaning not in ("", "nan"):
                fixed.append({"word": word, "meaning": meaning})

    return pd.DataFrame(fixed)


# ---------------------------
# INITIALIZATION
# ---------------------------

init_db()
valid_df = load_data()
load_words_from_csv(valid_df)

if "embeddings_generated" not in st.session_state:
    with st.spinner("⚙️ Loading AI model for smarter distractors… (once per session, ~30s)"):
        generate_and_store_embeddings()
    st.session_state.embeddings_generated = True


# ---------------------------
# SESSION STATE DEFAULTS
# ---------------------------

_defaults = {
    "page":             "home",
    "question_pool":    [],
    "current_question": None,
    "start_time":       None,
    # Live HUD
    "current_streak":   0,
    "best_streak":      0,
    "session_score":    0,
    # Post-session summary
    "session_results":  [],
    "session_attempted": 0,
    "session_correct":  0,
    # "practice" | "test" | "mistakes"
    "session_mode":     "practice",
    # Two-phase answer flow
    "feedback_state":   None,
    # Confirmation guard for Reset System
    "confirm_reset":    False,
    # Guard so update_all_time_stats fires only once per summary render
    "stats_saved":      False,
    # Word lookup page — stores selected word_id for detail view
    "lookup_word_id":   None,
}

for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------
# HELPERS
# ---------------------------

def difficulty_label(d):
    if   d < 0.6: return "🟢 Easy"
    elif d < 1.4: return "🟡 Medium"
    elif d < 2.2: return "🟠 Hard"
    else:         return "🔴 Very Hard"


def calculate_points(correct, response_time, streak):
    if not correct:
        return 0
    return 10 + (5 if response_time < 4.0 else 0) + min(streak * 2, 20)


def reset_session_stats():
    st.session_state.current_streak    = 0
    st.session_state.best_streak       = 0
    st.session_state.session_score     = 0
    st.session_state.session_results   = []
    st.session_state.session_attempted = 0
    st.session_state.session_correct   = 0
    st.session_state.stats_saved       = False
    st.session_state.feedback_state    = None
    st.session_state.current_question  = None   # prevent stale question flash



def generate_question():
    if not st.session_state.question_pool:
        st.session_state.current_question = None
        st.session_state.page             = "summary"
        return

    word_id, word, meaning, difficulty = st.session_state.question_pool.pop()

    wrong_options = get_semantic_distractors(word_id, limit=3)
    if len(wrong_options) < 3:
        # Use full vocabulary as fallback pool so test/mistakes modes
        # don't accidentally pull from the due-words schedule
        fallback_pool = get_random_words(50)
        fallback = [w[2] for w in fallback_pool if w[0] != word_id]
        random.shuffle(fallback)
        wrong_options += fallback[:3 - len(wrong_options)]

    options = wrong_options + [meaning]
    random.shuffle(options)

    st.session_state.current_question = {
        "word_id":    word_id,
        "word":       word,
        "correct":    meaning,
        "options":    options,
        "difficulty": difficulty,   # already in pool tuple — no extra DB call
    }
    st.session_state.start_time = time.time()


# =====================================================
# HOME PAGE
# =====================================================

if st.session_state.page == "home":

    st.title("🚀 LexiMind AI — Adaptive Vocabulary System")

    (
        total_attempts, total_correct, total_wrong,
        all_time_streak, all_time_score,
        total_vocab, due_words,
    ) = get_home_page_data()

    accuracy  = round((total_correct / total_attempts) * 100, 2) if total_attempts else 0
    due_count = len(due_words)

    # --- All-time records row ---
    r1, r2 = st.columns(2)
    r1.metric("🏅 All-time Best Streak", all_time_streak)
    r2.metric("🥇 All-time Best Score",  all_time_score)

    st.markdown("---")

    # --- Session stats row ---
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("📚 Vocabulary",  total_vocab)
    s2.metric("📝 Practiced",   total_attempts)
    s3.metric("✅ Correct",     total_correct)
    s4.metric("❌ Wrong",       total_wrong)
    s5.metric("🎯 Accuracy",    f"{accuracy}%")

    st.markdown("---")

    practice_label = f"🧠 Practice Due Words ({due_count} due today)"
    if st.button(practice_label):
        words = due_words   # already fetched above — no second DB call
        if not words:
            st.info("🎉 No words due for review today!")
        else:
            random.shuffle(words)
            st.session_state.question_pool  = words
            st.session_state.session_mode   = "practice"
            st.session_state.page           = "practice"
            reset_session_stats()
            generate_question()
            st.rerun()

    if st.button("🎯 Quick Test  (10 random questions)"):
        words = get_random_words(10)
        if not words:
            st.info("No words in the database yet.")
        else:
            st.session_state.question_pool  = list(words)
            st.session_state.session_mode   = "test"
            st.session_state.page           = "practice"
            reset_session_stats()
            generate_question()
            st.rerun()

    if st.button("❌ Review Mistakes"):
        words = get_mistake_words()
        if not words:
            st.info("No mistakes recorded yet.")
        else:
            random.shuffle(words)
            st.session_state.question_pool  = words
            st.session_state.session_mode   = "mistakes"
            st.session_state.page           = "practice"
            reset_session_stats()
            generate_question()
            st.rerun()

    if st.button("🔍 Word Lookup"):
        st.session_state.page          = "lookup"
        st.session_state.lookup_word_id = None
        st.rerun()

    if st.button("📊 View Analytics"):
        st.session_state.page = "analytics"
        st.rerun()

    st.markdown("---")

    if not st.session_state.confirm_reset:
        if st.button("🔄 Reset System"):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("⚠️ This will erase all your attempts, streaks, and review history. Are you sure?")
        yes_col, no_col = st.columns(2)
        with yes_col:
            if st.button("✅ Yes, reset everything"):
                reset_database()
                st.session_state.confirm_reset = False
                st.success("System reset successfully.")
                st.rerun()
        with no_col:
            if st.button("❌ Cancel"):
                st.session_state.confirm_reset = False
                st.rerun()


# =====================================================
# PRACTICE PAGE
# =====================================================

elif st.session_state.page == "practice":

    if not st.session_state.current_question:
        generate_question()

    q = st.session_state.current_question

    if not q:
        st.session_state.page = "summary"
        st.rerun()
        st.stop()

    mode = st.session_state.session_mode
    if mode == "test":
        remaining = len(st.session_state.question_pool)
        answered  = st.session_state.session_attempted
        total_q   = answered + remaining + 1
        st.title(f"🎯 Quick Test — Q{answered + 1} / {total_q}")
        st.progress((answered) / total_q)
    elif mode == "mistakes":
        st.title("❌ Mistake Review")
    else:
        st.title("🧠 Practice Mode")

    # Live HUD
    hud1, hud2, hud3 = st.columns(3)
    hud1.metric("🏆 Score",  st.session_state.session_score)
    hud2.metric("🔥 Streak", st.session_state.current_streak)
    hud3.metric("⚡ Best",   st.session_state.best_streak)

    st.markdown("---")
    st.caption(difficulty_label(q["difficulty"]))
    st.markdown(f"## {q['word']}")

    fb = st.session_state.feedback_state

    # ── PHASE 1: unanswered ──────────────────────────────────────────
    if fb is None:
        selected_option = st.radio("Choose the correct meaning:", q["options"])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit"):
                response_time = time.time() - st.session_state.start_time
                correct       = selected_option == q["correct"]

                record_attempt(q["word_id"], correct, response_time)

                # Test mode: record for analytics but don't touch SM-2 schedule
                if mode != "test":
                    update_schedule(q["word_id"], correct)
                    update_difficulty(q["word_id"], correct, response_time)

                if correct:
                    st.session_state.current_streak += 1
                    st.session_state.best_streak = max(
                        st.session_state.best_streak,
                        st.session_state.current_streak
                    )
                else:
                    st.session_state.current_streak = 0

                points = calculate_points(correct, response_time, st.session_state.current_streak)
                st.session_state.session_score     += points
                st.session_state.session_attempted += 1
                if correct:
                    st.session_state.session_correct += 1

                st.session_state.session_results.append({
                    "word":          q["word"],
                    "correct":       correct,
                    "response_time": round(response_time, 2),
                    "points":        points,
                })

                st.session_state.feedback_state = {
                    "correct":      correct,
                    "points":       points,
                    "chosen":       selected_option,
                    "right_answer": q["correct"],
                }
                st.rerun()

        with col2:
            if st.button("🏠 Home"):
                st.session_state.page             = "home"
                st.session_state.current_question = None
                st.session_state.feedback_state   = None
                st.rerun()

    # ── PHASE 2: answered — show feedback + Next ──────────────────────
    else:
        st.radio(
            "Choose the correct meaning:",
            q["options"],
            index=q["options"].index(fb["chosen"]),
            disabled=True,
        )

        if fb["correct"]:
            st.success(f"✅ Correct! +{fb['points']} pts")
        else:
            st.error(f"❌ Wrong — the correct answer is: **{fb['right_answer']}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Next →"):
                st.session_state.feedback_state = None
                generate_question()
                st.rerun()
        with col2:
            if st.button("🏠 Home"):
                st.session_state.page             = "home"
                st.session_state.current_question = None
                st.session_state.feedback_state   = None
                st.rerun()


# =====================================================
# SUMMARY PAGE
# =====================================================

elif st.session_state.page == "summary":

    # Persist all-time records exactly once when the summary first renders
    if not st.session_state.stats_saved:
        update_all_time_stats(
            st.session_state.session_score,
            st.session_state.best_streak
        )
        st.session_state.stats_saved = True

    mode        = st.session_state.session_mode
    attempted   = st.session_state.session_attempted
    correct     = st.session_state.session_correct
    wrong       = attempted - correct
    score       = st.session_state.session_score
    best_streak = st.session_state.best_streak
    accuracy    = round((correct / attempted) * 100) if attempted else 0

    if mode == "test":
        st.title("🎯 Test Complete!")
        grade = "A" if accuracy >= 90 else "B" if accuracy >= 80 else \
                "C" if accuracy >= 70 else "D" if accuracy >= 60 else "F"
        st.markdown(f"### Your grade: **{grade}**  ({accuracy}%)")
    else:
        st.title("🎉 Session Complete!")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Score",       score)
    m2.metric("✅ Correct",     correct)
    m3.metric("❌ Wrong",       wrong)
    m4.metric("⚡ Best Streak", best_streak)

    st.markdown("---")

    if   accuracy >= 90: st.success("🌟 Outstanding! You're truly mastering these words!")
    elif accuracy >= 70: st.info("👍 Solid session! A bit more practice and you'll nail these.")
    elif accuracy >= 50: st.warning("📚 Getting there! Consistent review will make a big difference.")
    else:                st.error("💪 Tough session — but that means lots of room to grow. Keep going!")

    if st.session_state.session_results:
        st.markdown("### 📋 Word-by-Word Breakdown")
        df = pd.DataFrame(st.session_state.session_results)
        df["Result"]   = df["correct"].map({True: "✅ Correct", False: "❌ Wrong"})
        df["Time (s)"] = df["response_time"]
        df["Points"]   = df["points"]
        # Wrong answers sorted to top so failures are immediately visible
        df = df.sort_values("correct", ascending=True).reset_index(drop=True)
        st.dataframe(
            df[["word", "Result", "Time (s)", "Points"]].rename(columns={"word": "Word"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        # Button label and action match the mode that was just completed
        again_label = "🎯 Take Another Test" if mode == "test" else "🔁 Practice Again"
        if st.button(again_label):
            if mode == "test":
                words = get_random_words(10)
                if not words:
                    st.info("No words available.")
                else:
                    st.session_state.question_pool = list(words)
                    st.session_state.session_mode  = "test"
                    st.session_state.page          = "practice"
                    reset_session_stats()
                    generate_question()
                    st.rerun()
            else:
                words = get_due_words()
                if not words:
                    st.info("🎉 No words due for review right now!")
                else:
                    random.shuffle(words)
                    st.session_state.question_pool = words
                    st.session_state.session_mode  = "practice"
                    st.session_state.page          = "practice"
                    reset_session_stats()
                    generate_question()
                    st.rerun()

    with c2:
        if st.button("🏠 Back to Home"):
            st.session_state.page = "home"
            st.rerun()



# =====================================================
# WORD LOOKUP PAGE
# =====================================================

elif st.session_state.page == "lookup":

    st.title("🔍 Word Lookup")

    # ── Detail view: user clicked a word from results ────────────────
    if st.session_state.lookup_word_id is not None:
        word_id = st.session_state.lookup_word_id

        # Fetch word info via proper DB function
        row = get_word_by_id(word_id)

        if row:
            word, meaning, difficulty = row
            total, correct, wrong, avg_time, history, next_review, reps = get_word_history(word_id)
            accuracy = round((correct / total) * 100) if total else 0

            st.markdown(f"## {word}")
            st.markdown(f"**Meaning:** {meaning}")
            st.caption(difficulty_label(difficulty))

            st.markdown("---")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📝 Attempts",  total)
            c2.metric("✅ Correct",   correct)
            c3.metric("❌ Wrong",     wrong)
            c4.metric("🎯 Accuracy",  f"{accuracy}%")

            c5, c6 = st.columns(2)
            c5.metric("⏱️ Avg Time (s)",   avg_time)
            c6.metric("📅 Next Review",    next_review)

            if history:
                st.markdown("---")
                st.subheader("📋 Recent Attempts (last 10)")
                df_hist = pd.DataFrame(history, columns=["correct", "response_time", "date"])
                df_hist["Result"]   = df_hist["correct"].map({1: "✅ Correct", 0: "❌ Wrong"})
                df_hist["Time (s)"] = df_hist["response_time"]
                df_hist["Date"]     = df_hist["date"]
                df_hist = df_hist.sort_values("correct", ascending=True)
                st.dataframe(
                    df_hist[["Date", "Result", "Time (s)"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No attempts recorded for this word yet.")

        st.markdown("---")
        if st.button("← Back to Search"):
            st.session_state.lookup_word_id = None
            st.rerun()

    # ── Search view ──────────────────────────────────────────────────
    else:
        query = st.text_input("Search by word or meaning:", placeholder="e.g. abstract, to study carefully…")

        if query.strip():
            results = search_words(query)

            if results:
                st.markdown(f"**{len(results)} result{'s' if len(results) != 1 else ''} found**")
                st.markdown("---")

                for word_id, word, meaning, difficulty in results:
                    col_word, col_diff, col_btn = st.columns([3, 2, 1])
                    snippet = meaning[:80] + ("…" if len(meaning) > 80 else "")
                    col_word.markdown(f"**{word}**")
                    col_word.caption(snippet)
                    col_diff.caption(difficulty_label(difficulty))
                    with col_btn:
                        if st.button("View", key=f"lookup_{word_id}"):
                            st.session_state.lookup_word_id = word_id
                            st.rerun()
            else:
                st.info("No words matched your search.")
        else:
            st.caption("Type a word or part of a meaning to search the full vocabulary.")

    st.markdown("---")
    if st.button("🏠 Back to Home"):
        st.session_state.page           = "home"
        st.session_state.lookup_word_id = None
        st.rerun()


# =====================================================
# ANALYTICS PAGE
# =====================================================

elif st.session_state.page == "analytics":

    st.title("📊 Learning Analytics Dashboard")

    (
        total_attempts, total_correct, accuracy,
        mastered, avg_time,
        all_time_streak, all_time_score,
        daily_attempts, daily_accuracy,
        difficulty_dist, hard_words,
    ) = get_analytics_page_data()

    # Top metrics — two rows of 3
    c1, c2, c3 = st.columns(3)
    c1.metric("📝 Total Attempts",     total_attempts)
    c2.metric("🎯 Overall Accuracy",   f"{accuracy}%")
    c3.metric("⭐ Mastered Words",     mastered)

    c4, c5, c6 = st.columns(3)
    c4.metric("🏅 Best Streak (ever)", all_time_streak)
    c5.metric("🥇 Best Score (ever)",  all_time_score)
    c6.metric("⏱️ Avg Response (sec)", avg_time)

    st.markdown("---")

    # ── Chart 1: Daily attempt count ────────────────────────────────
    st.subheader("📈 Daily Activity")
    if daily_attempts:
        df_daily = pd.DataFrame(daily_attempts, columns=["Date", "Attempts"])
        df_daily["Date"] = pd.to_datetime(df_daily["Date"])
        st.line_chart(df_daily.set_index("Date"))
    else:
        st.info("No activity data yet.")

    st.markdown("---")

    # ── Chart 2: Accuracy over time ──────────────────────────────────
    st.subheader("🎯 Accuracy Over Time")
    if daily_accuracy:
        df_acc = pd.DataFrame(daily_accuracy, columns=["Date", "Accuracy (%)"])
        df_acc["Date"] = pd.to_datetime(df_acc["Date"])
        st.line_chart(df_acc.set_index("Date"))
    else:
        st.info("No accuracy data yet.")

    st.markdown("---")

    # ── Chart 3: Difficulty distribution ────────────────────────────
    st.subheader("📊 Vocabulary Difficulty Distribution")
    total_words = sum(difficulty_dist.values())
    if total_words > 0:
        df_dist = pd.DataFrame({
            "Difficulty": list(difficulty_dist.keys()),
            "Words":      list(difficulty_dist.values()),
        })
        st.bar_chart(df_dist.set_index("Difficulty"))
        for label, count in difficulty_dist.items():
            pct = round(count / total_words * 100)
            st.caption(f"{label}: {count} words ({pct}%)")
    else:
        st.info("No word difficulty data yet.")

    st.markdown("---")

    # ── Hardest words ────────────────────────────────────────────────
    st.subheader("🔥 Hardest Words")
    if hard_words:
        for word, wrong_count in hard_words:
            st.write(f"• **{word}** — {wrong_count} mistakes")
    else:
        st.info("No mistakes recorded yet.")

    st.markdown("---")

    if st.button("🏠 Back to Home"):
        st.session_state.page = "home"
        st.rerun()