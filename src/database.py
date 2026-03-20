"""
database.py
===========
SQLite-based persistence layer for the Research Assistant.

Two responsibilities:
  1. User accounts — email/password auth, per-user search history
  2. Search cache — store fetched+processed results, expire after 7 days

Database file is stored at:
  Windows: C:/Users/<user>/research_assistant.db
  Mac/Linux: ~/research_assistant.db

No server needed. SQLite is built into Python.
"""

import os
import sqlite3
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# ── Database location ────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.expanduser("~"), "research_assistant.db")
CACHE_EXPIRY_DAYS = 7


# ─────────────────────────────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    try:
        c = conn.cursor()

        # ── Users table ──────────────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT    UNIQUE NOT NULL,
                name        TEXT    NOT NULL,
                password_hash TEXT  NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── Search history ───────────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id),
                query       TEXT    NOT NULL,
                paper_count INTEGER,
                cluster_count INTEGER,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── Paper cache ──────────────────────────────────────────────
        # Stores fetched+processed papers keyed by query hash.
        # papers_json  = full list of paper dicts (with summaries)
        # clusters_json = cluster assignment dict
        # expires_at   = datetime string, checked before serving cache
        c.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash   TEXT    UNIQUE NOT NULL,
                query_text   TEXT    NOT NULL,
                papers_json  TEXT    NOT NULL,
                clusters_json TEXT,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                expires_at   TEXT    NOT NULL
            )
        """)

        # ── Saved papers (user bookmarks) ────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS saved_papers (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                paper_json TEXT    NOT NULL,
                query      TEXT,
                note       TEXT,
                saved_at   TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        conn.commit()
        logger.info(f"[DB] Database initialised at {DB_PATH}")
    except Exception as e:
        logger.error(f"[DB] Init failed: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# USER AUTH
# ─────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """SHA-256 hash with salt. Simple and dependency-free."""
    salt = "research_assistant_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def create_user(email: str, name: str, password: str) -> Dict:
    """
    Create a new user. Returns {'success': True, 'user_id': int}
    or {'success': False, 'error': str}.
    """
    email = email.strip().lower()
    if not email or not password or not name:
        return {'success': False, 'error': 'All fields are required'}
    if len(password) < 6:
        return {'success': False, 'error': 'Password must be at least 6 characters'}

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
            (email, name.strip(), _hash_password(password))
        )
        conn.commit()
        user_id = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()['id']
        return {'success': True, 'user_id': user_id, 'name': name.strip(), 'email': email}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': 'An account with this email already exists'}
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def login_user(email: str, password: str) -> Dict:
    """
    Verify credentials. Returns {'success': True, 'user': {...}}
    or {'success': False, 'error': str}.
    """
    email = email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        if not row:
            return {'success': False, 'error': 'No account found with this email'}
        if row['password_hash'] != _hash_password(password):
            return {'success': False, 'error': 'Incorrect password'}
        return {
            'success': True,
            'user': {
                'id':    row['id'],
                'name':  row['name'],
                'email': row['email'],
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# SEARCH HISTORY
# ─────────────────────────────────────────────────────────────────────

def save_search(user_id: int, query: str, paper_count: int, cluster_count: int):
    """Record a search in the user's history."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO searches (user_id, query, paper_count, cluster_count)
               VALUES (?, ?, ?, ?)""",
            (user_id, query, paper_count, cluster_count)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[DB] Could not save search: {e}")
    finally:
        conn.close()


def get_search_history(user_id: int, limit: int = 20) -> List[Dict]:
    """Return last N searches for a user, most recent first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT query, paper_count, cluster_count, created_at
               FROM searches
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[DB] Could not get history: {e}")
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# SEARCH CACHE
# ─────────────────────────────────────────────────────────────────────

def _query_hash(query: str) -> str:
    """Normalise query and hash it for cache key lookup."""
    normalised = ' '.join(query.strip().lower().split())
    return hashlib.md5(normalised.encode()).hexdigest()


def get_cached_search(query: str) -> Optional[Dict]:
    """
    Return cached papers+clusters if they exist and haven't expired.
    Returns None if cache miss or expired.
    """
    qhash = _query_hash(query)
    conn  = get_connection()
    try:
        row = conn.execute(
            "SELECT papers_json, clusters_json, expires_at FROM search_cache WHERE query_hash = ?",
            (qhash,)
        ).fetchone()
        if not row:
            return None
        # Check expiry
        expires_at = datetime.fromisoformat(row['expires_at'])
        if datetime.now() > expires_at:
            # Expired — delete it
            conn.execute("DELETE FROM search_cache WHERE query_hash = ?", (qhash,))
            conn.commit()
            return None
        return {
            'papers':   json.loads(row['papers_json']),
            'clusters': json.loads(row['clusters_json']) if row['clusters_json'] else {},
        }
    except Exception as e:
        logger.warning(f"[DB] Cache read failed: {e}")
        return None
    finally:
        conn.close()


def save_to_cache(query: str, papers: List[Dict], clusters: Dict):
    """Store search results in cache with 7-day expiry."""
    qhash      = _query_hash(query)
    expires_at = (datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)).isoformat()
    conn       = get_connection()
    try:
        # Serialize — strip large extracted_content to keep DB size manageable
        slim_papers = []
        for p in papers:
            sp = {k: v for k, v in p.items() if k != 'extracted_content'}
            slim_papers.append(sp)

        conn.execute(
            """INSERT OR REPLACE INTO search_cache
               (query_hash, query_text, papers_json, clusters_json, expires_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                qhash,
                query,
                json.dumps(slim_papers, default=str),
                json.dumps(clusters,    default=str),
                expires_at
            )
        )
        conn.commit()
        logger.info(f"[DB] Cached {len(papers)} papers for query: {query[:40]}")
    except Exception as e:
        logger.warning(f"[DB] Cache write failed: {e}")
    finally:
        conn.close()


def clear_expired_cache():
    """Delete all expired cache entries. Call periodically."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM search_cache WHERE expires_at < ?",
            (datetime.now().isoformat(),)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"[DB] Cache cleanup failed: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# SAVED PAPERS
# ─────────────────────────────────────────────────────────────────────

def save_paper(user_id: int, paper: Dict, query: str = "", note: str = "") -> bool:
    """Bookmark a paper for a user."""
    conn = get_connection()
    try:
        slim = {k: v for k, v in paper.items() if k != 'extracted_content'}
        conn.execute(
            "INSERT INTO saved_papers (user_id, paper_json, query, note) VALUES (?, ?, ?, ?)",
            (user_id, json.dumps(slim, default=str), query, note)
        )
        conn.commit()
        return True
    except Exception as e:
        logger.warning(f"[DB] Save paper failed: {e}")
        return False
    finally:
        conn.close()


def get_saved_papers(user_id: int) -> List[Dict]:
    """Return all bookmarked papers for a user."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT paper_json, query, note, saved_at FROM saved_papers WHERE user_id = ? ORDER BY saved_at DESC",
            (user_id,)
        ).fetchall()
        result = []
        for row in rows:
            paper = json.loads(row['paper_json'])
            paper['_saved_query'] = row['query']
            paper['_note']        = row['note']
            paper['_saved_at']    = row['saved_at']
            result.append(paper)
        return result
    except Exception as e:
        logger.warning(f"[DB] Get saved papers failed: {e}")
        return []
    finally:
        conn.close()