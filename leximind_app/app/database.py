import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

DB_FILE = "vocab.db"

# ---------------------------
# IN-MEMORY VECTOR CACHE
# ---------------------------

embedding_matrix = None
embedding_word_ids = None
embedding_meanings = None
id_to_index = None

# Lazy-loaded model
model = None


# ---------------------------
# DATABASE CONNECTION
# ---------------------------

def get_connection():
    return sqlite3.connect(DB_FILE)


# ---------------------------
# INITIALIZATION
# ---------------------------

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            meaning TEXT,
            difficulty REAL DEFAULT 1.0,
            embedding TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word_id INTEGER,
            correct INTEGER,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_schedule (
            word_id INTEGER PRIMARY KEY,
            ease_factor REAL DEFAULT 2.5,
            interval INTEGER DEFAULT 1,
            repetitions INTEGER DEFAULT 0,
            next_review TEXT,
            FOREIGN KEY(word_id) REFERENCES words(id)
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------
# LOAD WORDS
# ---------------------------

def load_words_from_csv(df):
    conn = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO words (word, meaning)
            VALUES (?, ?)
        """, (row["word"], row["meaning"]))

    conn.commit()
    conn.close()


# ---------------------------
# RECORD ATTEMPTS
# ---------------------------

def record_attempt(word_id, correct, response_time):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO attempts (word_id, correct, response_time)
        VALUES (?, ?, ?)
    """, (word_id, int(correct), response_time))

    conn.commit()
    conn.close()


# ---------------------------
# SM-2 SPACED REPETITION
# ---------------------------

def update_schedule(word_id, correct):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ease_factor, interval, repetitions
        FROM review_schedule
        WHERE word_id = ?
    """, (word_id,))

    row = cursor.fetchone()

    if row:
        ease_factor, interval, repetitions = row
    else:
        ease_factor, interval, repetitions = 2.5, 1, 0

    if correct:
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = int(interval * ease_factor)

        ease_factor += 0.1
    else:
        repetitions = 0
        interval = 1
        ease_factor = max(1.3, ease_factor - 0.2)

    next_review = datetime.now() + timedelta(days=interval)

    cursor.execute("""
        INSERT OR REPLACE INTO review_schedule
        (word_id, ease_factor, interval, repetitions, next_review)
        VALUES (?, ?, ?, ?, ?)
    """, (
        word_id,
        ease_factor,
        interval,
        repetitions,
        next_review.strftime("%Y-%m-%d")
    ))

    conn.commit()
    conn.close()


# ---------------------------
# GET DUE WORDS
# ---------------------------

def get_due_words():
    conn = get_connection()
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT w.id, w.word, w.meaning
        FROM words w
        LEFT JOIN review_schedule r
        ON w.id = r.word_id
        WHERE r.next_review IS NULL
        OR r.next_review <= ?
    """, (today,))

    rows = cursor.fetchall()
    conn.close()
    return rows


# ---------------------------
# ANALYTICS
# ---------------------------

def get_total_correct():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attempts WHERE correct = 1")
    result = cursor.fetchone()[0]
    conn.close()
    return result


def get_total_attempts():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM attempts")
    result = cursor.fetchone()[0]
    conn.close()
    return result


def get_daily_attempts():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DATE(timestamp), COUNT(*)
        FROM attempts
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_hard_words(limit=5):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT w.word,
               SUM(CASE WHEN a.correct = 0 THEN 1 ELSE 0 END) as wrong_count
        FROM attempts a
        JOIN words w ON a.word_id = w.id
        GROUP BY w.id
        ORDER BY wrong_count DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_average_response_time():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT AVG(response_time) FROM attempts")
    result = cursor.fetchone()[0]
    conn.close()
    return round(result, 2) if result else 0


def get_mastered_words_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM review_schedule
        WHERE repetitions >= 3
    """)
    result = cursor.fetchone()[0]
    conn.close()
    return result


def reset_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attempts")
    cursor.execute("DELETE FROM review_schedule")
    conn.commit()
    conn.close()


# ---------------------------
# EMBEDDING GENERATION
# ---------------------------

def generate_and_store_embeddings(batch_size=128):
    global model
    global embedding_matrix, embedding_word_ids
    global embedding_meanings, id_to_index

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, meaning FROM words WHERE embedding IS NULL")
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return

    word_ids = [row[0] for row in rows]
    meanings = [row[1] for row in rows]

    embeddings = model.encode(
        meanings,
        batch_size=batch_size,
        show_progress_bar=True
    )

    for word_id, embedding in zip(word_ids, embeddings):
        cursor.execute("""
            UPDATE words
            SET embedding = ?
            WHERE id = ?
        """, (json.dumps(embedding.tolist()), word_id))

    conn.commit()
    conn.close()

    # Reset cache
    embedding_matrix = None
    embedding_word_ids = None
    embedding_meanings = None
    id_to_index = None


# ---------------------------
# EMBEDDING CACHE LOADING
# ---------------------------

def load_embedding_cache():
    global embedding_matrix, embedding_word_ids
    global embedding_meanings, id_to_index

    if embedding_matrix is not None:
        return

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, meaning, embedding
        FROM words
        WHERE embedding IS NOT NULL
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return

    embedding_word_ids = []
    embedding_meanings = []
    vectors = []

    for word_id, meaning, embedding_json in rows:
        embedding_word_ids.append(word_id)
        embedding_meanings.append(meaning)
        vectors.append(json.loads(embedding_json))

    embedding_matrix = np.array(vectors)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    embedding_matrix = embedding_matrix / norms

    id_to_index = {
        word_id: idx
        for idx, word_id in enumerate(embedding_word_ids)
    }


# ---------------------------
# SEMANTIC DISTRACTORS
# ---------------------------

def get_semantic_distractors(word_id, limit=3):
    global embedding_matrix, id_to_index, embedding_meanings

    load_embedding_cache()

    if embedding_matrix is None:
        return []

    if word_id not in id_to_index:
        return []

    target_index = id_to_index[word_id]
    target_vector = embedding_matrix[target_index]

    similarities = embedding_matrix @ target_vector
    similarities[target_index] = -1

    top_indices = np.argsort(similarities)[-limit:][::-1]

    return [embedding_meanings[i] for i in top_indices]