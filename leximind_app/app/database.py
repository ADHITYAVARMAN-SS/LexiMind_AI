import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

DB_FILE = "vocab.db"

# ---------------------------
# IN-MEMORY VECTOR CACHE
# ---------------------------

embedding_matrix   = None
embedding_word_ids = None
embedding_meanings = None
id_to_index        = None

model = None


# ---------------------------
# DATABASE CONNECTION
# timeout=10 prevents silent hangs when two browser tabs write simultaneously
# ---------------------------

def get_connection():
    return sqlite3.connect(DB_FILE, timeout=10)


# ---------------------------
# INITIALIZATION
# ---------------------------

def init_db():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            word       TEXT UNIQUE,
            meaning    TEXT,
            difficulty REAL DEFAULT 1.0,
            embedding  TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id       INTEGER,
            correct       INTEGER,
            response_time REAL,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_schedule (
            word_id     INTEGER PRIMARY KEY,
            ease_factor REAL    DEFAULT 2.5,
            interval    INTEGER DEFAULT 1,
            repetitions INTEGER DEFAULT 0,
            next_review TEXT,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )
    """)

    # Persistent all-time records — single row (id = 1)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            id          INTEGER PRIMARY KEY DEFAULT 1,
            best_streak INTEGER DEFAULT 0,
            best_score  INTEGER DEFAULT 0
        )
    """)
    cursor.execute(
        "INSERT OR IGNORE INTO stats (id, best_streak, best_score) VALUES (1, 0, 0)"
    )

    conn.commit()
    conn.close()


# ---------------------------
# LOAD WORDS
# ---------------------------

def load_words_from_csv(df):
    """
    Inserts words into DB. Skips entirely when DB already has >= len(df) words
    to avoid iterating hundreds of rows on every cold start.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM words")
    if cursor.fetchone()[0] >= len(df):
        conn.close()
        return

    for _, row in df.iterrows():
        cursor.execute(
            "INSERT OR IGNORE INTO words (word, meaning) VALUES (?, ?)",
            (str(row["word"]).strip(), str(row["meaning"]).strip())
        )

    conn.commit()
    conn.close()


# ---------------------------
# RECORD ATTEMPTS
# ---------------------------

def record_attempt(word_id, correct, response_time):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO attempts (word_id, correct, response_time) VALUES (?, ?, ?)",
        (word_id, int(correct), response_time)
    )
    conn.commit()
    conn.close()


# ---------------------------
# SM-2 SPACED REPETITION
# ---------------------------

def update_schedule(word_id, correct):
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT ease_factor, interval, repetitions FROM review_schedule WHERE word_id = ?",
        (word_id,)
    )
    row = cursor.fetchone()
    ease_factor, interval, repetitions = row if row else (2.5, 1, 0)

    if correct:
        repetitions += 1
        if   repetitions == 1: interval = 1
        elif repetitions == 2: interval = 6
        else:                  interval = int(interval * ease_factor)
        ease_factor += 0.1
    else:
        repetitions = 0
        interval    = 1
        ease_factor = max(1.3, ease_factor - 0.2)

    next_review = (datetime.now() + timedelta(days=interval)).strftime("%Y-%m-%d")
    cursor.execute("""
        INSERT OR REPLACE INTO review_schedule
            (word_id, ease_factor, interval, repetitions, next_review)
        VALUES (?, ?, ?, ?, ?)
    """, (word_id, ease_factor, interval, repetitions, next_review))

    conn.commit()
    conn.close()


# ---------------------------
# DIFFICULTY PROGRESSION
# ---------------------------

def update_difficulty(word_id, correct, response_time):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT difficulty FROM words WHERE id = ?", (word_id,))
    row        = cursor.fetchone()
    difficulty = row[0] if row else 1.0

    if correct:
        difficulty = max(0.1, difficulty - (0.15 if response_time < 4.0 else 0.05))
    else:
        difficulty = min(3.0, difficulty + 0.25)

    cursor.execute("UPDATE words SET difficulty = ? WHERE id = ?", (difficulty, word_id))
    conn.commit()
    conn.close()


def get_word_difficulty(word_id):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT difficulty FROM words WHERE id = ?", (word_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 1.0


# ---------------------------
# GET DUE WORDS
# ---------------------------

def get_due_words():
    conn   = get_connection()
    cursor = conn.cursor()
    today  = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT w.id, w.word, w.meaning
        FROM words w
        LEFT JOIN review_schedule r ON w.id = r.word_id
        WHERE r.next_review IS NULL OR r.next_review <= ?
        ORDER BY w.difficulty DESC
    """, (today,))
    rows = cursor.fetchall()
    conn.close()
    return rows



# ---------------------------
# HOME STATS  (single round-trip)
# ---------------------------

def get_home_stats():
    """Returns (total_attempts, total_correct, total_wrong)."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*),
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END),
            SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END)
        FROM attempts
    """)
    row = cursor.fetchone()
    conn.close()
    total   = row[0] or 0
    correct = row[1] or 0
    wrong   = row[2] or 0
    return total, correct, wrong


# ---------------------------
# ALL-TIME RECORDS
# ---------------------------

def get_all_time_stats():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT best_streak, best_score FROM stats WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    return (row[0], row[1]) if row else (0, 0)


def update_all_time_stats(session_score, session_best_streak):
    """Only writes when the session beat existing records."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT best_streak, best_score FROM stats WHERE id = 1")
    row            = cursor.fetchone()
    cur_streak     = row[0] if row else 0
    cur_score      = row[1] if row else 0
    cursor.execute(
        "UPDATE stats SET best_streak = ?, best_score = ? WHERE id = 1",
        (max(cur_streak, session_best_streak), max(cur_score, session_score))
    )
    conn.commit()
    conn.close()


# ---------------------------
# ANALYTICS
# ---------------------------

def get_daily_attempts():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(timestamp), COUNT(*)
        FROM attempts GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_daily_accuracy():
    """Returns (date, accuracy_pct) for the accuracy-over-time chart."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            DATE(timestamp),
            ROUND(100.0 * SUM(correct) / COUNT(*), 1)
        FROM attempts
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_difficulty_distribution():
    """Word counts per difficulty bracket."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            SUM(CASE WHEN difficulty <  0.6 THEN 1 ELSE 0 END),
            SUM(CASE WHEN difficulty >= 0.6 AND difficulty < 1.4 THEN 1 ELSE 0 END),
            SUM(CASE WHEN difficulty >= 1.4 AND difficulty < 2.2 THEN 1 ELSE 0 END),
            SUM(CASE WHEN difficulty >= 2.2 THEN 1 ELSE 0 END)
        FROM words
    """)
    row = cursor.fetchone()
    conn.close()
    return {
        "🟢 Easy":      row[0] or 0,
        "🟡 Medium":    row[1] or 0,
        "🟠 Hard":      row[2] or 0,
        "🔴 Very Hard": row[3] or 0,
    }


def get_hard_words(limit=5):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT w.word,
               SUM(CASE WHEN a.correct = 0 THEN 1 ELSE 0 END) AS wrong_count
        FROM attempts a JOIN words w ON a.word_id = w.id
        GROUP BY w.id ORDER BY wrong_count DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_average_response_time():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(response_time) FROM attempts")
    result = cursor.fetchone()[0]
    conn.close()
    return round(result, 2) if result else 0


def get_mastered_words_count():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM review_schedule WHERE repetitions >= 3")
    result = cursor.fetchone()[0]
    conn.close()
    return result


# Legacy single-stat helpers kept for backward compatibility
def get_total_correct():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attempts WHERE correct = 1")
    result = cursor.fetchone()[0]
    conn.close()
    return result

def get_total_wrong():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attempts WHERE correct = 0")
    result = cursor.fetchone()[0]
    conn.close()
    return result

def get_total_attempts():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attempts")
    result = cursor.fetchone()[0]
    conn.close()
    return result

def get_random_words(n=10):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, word, meaning FROM words ORDER BY RANDOM() LIMIT ?", (n,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows



def get_mistake_words():
    """Words the user has answered incorrectly at least once."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT w.id, w.word, w.meaning
        FROM attempts a JOIN words w ON a.word_id = w.id
        WHERE a.correct = 0
        GROUP BY w.id
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_total_words():
    """Total number of words loaded in the vocabulary."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM words")
    result = cursor.fetchone()[0]
    conn.close()
    return result


def reset_database():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attempts")
    cursor.execute("DELETE FROM review_schedule")
    cursor.execute("UPDATE words SET difficulty = 1.0")
    cursor.execute("UPDATE stats SET best_streak = 0, best_score = 0 WHERE id = 1")
    conn.commit()
    conn.close()


# ---------------------------
# EMBEDDING GENERATION
# ---------------------------

def generate_and_store_embeddings(batch_size=128):
    global model
    global embedding_matrix, embedding_word_ids, embedding_meanings, id_to_index

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, meaning FROM words WHERE embedding IS NULL")
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        load_embedding_cache()
        return

    word_ids   = [r[0] for r in rows]
    meanings   = [r[1] for r in rows]
    embeddings = model.encode(meanings, batch_size=batch_size, show_progress_bar=False)

    for word_id, emb in zip(word_ids, embeddings):
        cursor.execute("UPDATE words SET embedding = ? WHERE id = ?",
                       (json.dumps(emb.tolist()), word_id))

    conn.commit()
    conn.close()
    embedding_matrix = embedding_word_ids = embedding_meanings = id_to_index = None
    load_embedding_cache()


# ---------------------------
# EMBEDDING CACHE
# ---------------------------

def load_embedding_cache():
    global embedding_matrix, embedding_word_ids, embedding_meanings, id_to_index

    if embedding_matrix is not None:
        return

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, meaning, embedding FROM words WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return

    embedding_word_ids = []
    embedding_meanings = []
    vectors = []

    for word_id, meaning, emb_json in rows:
        embedding_word_ids.append(word_id)
        embedding_meanings.append(meaning)
        vectors.append(json.loads(emb_json))

    embedding_matrix = np.array(vectors)
    norms            = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / norms
    id_to_index      = {wid: idx for idx, wid in enumerate(embedding_word_ids)}


# ---------------------------
# SEMANTIC DISTRACTORS
# ---------------------------

def get_semantic_distractors(word_id, limit=3):
    global embedding_matrix, id_to_index, embedding_meanings

    load_embedding_cache()

    if embedding_matrix is None or word_id not in id_to_index:
        return []

    idx          = id_to_index[word_id]
    similarities = embedding_matrix @ embedding_matrix[idx]
    similarities[idx] = -1

    sweet = np.where((similarities >= 0.35) & (similarities <= 0.78))[0]

    if len(sweet) >= limit:
        top = sweet[np.argsort(similarities[sweet])[::-1]][:limit]
    else:
        filtered = similarities.copy()
        filtered[similarities > 0.90] = -1
        top = np.argsort(filtered)[-limit:][::-1]

    return [embedding_meanings[i] for i in top]