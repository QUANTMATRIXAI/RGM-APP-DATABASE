import streamlit as st
import hashlib, os, sqlite3
from datetime import datetime

# ───────────────────────── PAGE CONFIG ─────────────────────────
st.set_page_config(page_title="QM Trinity | Login", page_icon="🔒", layout="wide")

# ───────────────────────── CUSTOM CSS ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
:root{
  --qm-yellow:#FFBD59;--qm-green:#41C185;--qm-blue:#458EE2;--qm-dark:#333;
}
*{box-sizing:border-box;margin:0;padding:0;}
.stApp{font-family:'Inter',sans-serif;color:var(--qm-dark);}
body::before{content:"";position:fixed;inset:0;z-index:-2;
  background:linear-gradient(135deg,#FFF7E8 0%,#F5F5F5 35%,#ECF3FF 100%);}
.stApp,.block-container,[data-testid="stHeader"],[data-testid="stSidebar"]{background:transparent !important;}

/* ───── Cards Grid ───── */
.projects-grid{display:flex;flex-wrap:wrap;gap:1.25rem;}
.card{flex:1 1 280px;max-width:310px;min-height:180px;display:flex;
  flex-direction:column;justify-content:space-between;
  border:1px solid #ddd;border-radius:12px;padding:1rem;background:#fff;
  box-shadow:0 2px 5px rgba(0,0,0,.07);transition:transform .2s;}
.card:hover{transform:translateY(-4px);}
.card h3{font-size:1.25rem;margin:0 0 .35rem;}
.card p{font-size:.9rem;color:#555;margin:0 0 1rem;}
.card .stButton > button{width:100%;padding:.45rem 0;border-radius:30px;
  background:var(--qm-blue);color:#fff;font-weight:600;border:none;}
.card .stButton > button:hover{background:#3571c1;}

/* “Create new” card */
.new-card{border:2px dashed var(--qm-yellow);color:#777;text-align:center;}
.new-card:hover{background:rgba(255,189,89,.07);transform:none;}
.new-card h3{color:var(--qm-yellow);}
.new-card input{margin-top:.75rem;width:100%;padding:.7rem .9rem;border:1px solid #ccc;
  border-radius:30px;font-size:.95rem;background:#ffffffee;}
.new-card .stButton > button{margin-top:.65rem;background:var(--qm-yellow);color:#333;}
.new-card .stButton > button:hover{background:#ffd37e;}

/* Current project header */
.proj-header{display:flex;align-items:center;margin-bottom:1.25rem;gap:.75rem;}
.proj-header h2{margin:0;font-size:1.8rem;font-weight:700;}
.back-btn{background:var(--qm-blue);color:#fff;font-size:.9rem;border:none;
  border-radius:50px;padding:.4rem 1.2rem;cursor:pointer;}
.back-btn:hover{background:#3571c1;}
</style>

<style>
/* ───── Saved‑Project Card ───── */
.proj-card{
  border:1px solid #e0e0e0; border-radius:16px;
  background:#fff; height:200px; padding:1.1rem;
  display:flex; flex-direction:column; justify-content:space-between;
  box-shadow:0 2px 6px rgba(0,0,0,.08);
  transition:transform .25s, box-shadow .25s;
}
.proj-card:hover{
  transform:translateY(-6px);
  box-shadow:0 8px 16px rgba(0,0,0,.15);
}

.proj-card h4{
  margin:0; font-size:1.25rem; font-weight:600; color:var(--qm-dark);
}
.proj-card .date{
  font-size:.85rem; color:#666; margin:0 0 .8rem;
}

/* primary & ghost buttons that sit inside cards */
.btn-primary>button{
  width:100%; padding:.5rem 0; border:none; border-radius:30px;
  background:var(--qm-blue); color:#fff; font-weight:600; transition:background .25s;
}
.btn-primary>button:hover{background:#3571c1;}

.btn-ghost>button{
  width:100%; padding:.45rem 0; border:2px solid var(--qm-yellow);
  border-radius:30px; background:transparent; color:var(--qm-yellow);
  font-weight:600; transition:background .25s, color .25s;
}
.btn-ghost>button:hover{
  background:var(--qm-yellow); color:var(--qm-dark);
}

/* “＋ New project” card */
.new-card{
  border:2px dashed var(--qm-yellow); border-radius:16px;
  color:#777; height:200px; display:flex; flex-direction:column;
  justify-content:center; text-align:center; transition:background .25s;
}
.new-card:hover{background:rgba(255,189,89,.07);}
.new-card input{
  margin-top:.9rem; width:100%; padding:.75rem .95rem;
  border:1px solid #ccc; border-radius:30px; font-size:.95rem; background:#ffffffe8;
}
.new-card .btn-primary>button{
  margin-top:.6rem; background:var(--qm-yellow); color:#333;
}
.new-card .btn-primary>button:hover{background:#ffd37e;}
</style>

""", unsafe_allow_html=True)


st.markdown("""
<style>
    /* Input fields with light blue border */
    .stTextInput > div > div > input {
        background-color: white;
        border: 1px solid #c0d6f9 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Textarea with light blue border */
    .stTextArea > div > div > textarea {
        background-color: white;
        border: 1px solid #c0d6f9 !important;
        border-radius: 6px !important;
        padding: 10px !important;
        transition: all 0.3s ease;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Form container with light blue border */
    .stForm {
        background-color: #f8f9fa;
        border: 1px solid #c0d6f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────── DATABASE SETUP ─────────────────────────
# ───────────────────────── DATABASE SETUP ─────────────────────────
DB_PATH = "users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    """Create both tables when the DB file does not yet exist."""
    db = get_db()
    # full project schema, including description
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            username      TEXT PRIMARY KEY,
            salt          BLOB NOT NULL,
            password_hash BLOB NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT,
            name        TEXT,
            client      TEXT,
            study       TEXT,
            phase       TEXT,
            description TEXT,
            type        TEXT,
            created_at  TEXT
        );
        """
    )

    # seed demo users if first launch
    for u, p in [("alice", "password1"), ("bob", "password2")]:
        if not db.execute("SELECT 1 FROM users WHERE username=?", (u,)).fetchone():
            salt = os.urandom(32)
            hsh = hashlib.pbkdf2_hmac("sha256", p.encode(), salt, 100_000)
            db.execute("INSERT INTO users VALUES (?,?,?)", (u, salt, hsh))

    db.commit()


def ensure_cols():
    """Add new columns to an old DB file (runs every start‑up, harmless if up‑to‑date)."""
    db = get_db()
    existing = {info[1] for info in db.execute("PRAGMA table_info(projects)")}
    for col in ("client", "study", "phase", "description","type"):
        if col not in existing:
            db.execute(f"ALTER TABLE projects ADD COLUMN {col} TEXT;")
    db.commit()


# initialise database (fresh file) and then ensure all columns exist
if not os.path.exists(DB_PATH):
    init_db()
ensure_cols()


# ── password helpers (unchanged) ──────────────────────────────
def hash_pw(pw, salt=None):
    salt = salt or os.urandom(32)
    hsh = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)
    return salt, hsh


def verify_pw(user, pw):
    row = get_db().execute(
        "SELECT salt, password_hash FROM users WHERE username=?", (user,)
    ).fetchone()
    return bool(row) and hashlib.pbkdf2_hmac("sha256", pw.encode(), row[0], 100_000) == row[1]

# ───────────────────────── SESSION STATE ─────────────────────────
st.session_state.setdefault("logged",False)
st.session_state.setdefault("user",None)
st.session_state.setdefault("current_project",None)

# ───────────────────────── AUTH FLOW ─────────────────────────
if not st.session_state.logged:
    # small vertical spacer
    st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)

    mid = st.columns([1, 3, 1])[1]          # centre the auth box
    with mid:
        st.image("logo.jpg", width=84)      # floating logo
        st.markdown(
            "<h1 style='text-align:center;margin-top:.25rem;'>QM Trinity</h1>",
            unsafe_allow_html=True
        )

        tabs = st.tabs(["Login", "Register"])

        # ─── LOGIN ────────────────────────────────────────────────
        with tabs[0]:
            with st.form("login_form"):
                st.markdown('<div class="login-wrap">', unsafe_allow_html=True)

                # username field
                st.markdown('<div class="icon">👤</div>', unsafe_allow_html=True)
                user = st.text_input("Username", label_visibility="collapsed")

                # password field
                st.markdown('<div class="icon">🔒</div>', unsafe_allow_html=True)
                pwd = st.text_input("Password", type="password",
                                    label_visibility="collapsed")

                # submit
                if st.form_submit_button("Sign In"):
                    if verify_pw(user, pwd):
                        st.session_state.logged = True
                        st.session_state.user   = user
                        st.rerun()
                    else:
                        st.error("Invalid credentials", icon="⚠️")

                st.markdown("</div>", unsafe_allow_html=True)   # /login‑wrap

        # ─── REGISTER ────────────────────────────────────────────
        with tabs[1]:
            with st.form("register_form"):
                ru = st.text_input("Choose a username")
                p1 = st.text_input("Choose a password",  type="password")
                p2 = st.text_input("Confirm password",  type="password")

                if st.form_submit_button("Create Account"):
                    db = get_db()
                    if not ru or db.execute(
                        "SELECT 1 FROM users WHERE username=?",
                        (ru,)
                    ).fetchone():
                        st.error("Username invalid or already taken.")
                    elif p1 != p2:
                        st.error("Passwords do not match.")
                    else:
                        s, h = hash_pw(p1)
                        db.execute("INSERT INTO users VALUES (?,?,?)", (ru, s, h))
                        db.commit()
                        st.success("Account created — you can log in now! 😊")

# ───────────────────────── MAIN APP ─────────────────────────
else:
    import time                          #  needed for the 1‑second pause
    db, user = get_db(), st.session_state.user

    # ── SIDEBAR: user badge + logout pill ──────────────────────────
    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center;line-height:1.4">
                <span style="font-size:1.8rem;">👤</span><br>
                <strong>{user.title()}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("----")          # thin rule

        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state.logged = False
            st.session_state.user = None
            st.session_state.current_project = None
            st.rerun()
    # ───────────────────────────────────────────────────────────────

    # ‣ DASHBOARD (cards) ────────────────────────────────────────
    st.subheader(f"Welcome, {user.title()}!")

    projs = db.execute(
        "SELECT id, name, client, study, phase, type, created_at "  # Added 'type' field
        "FROM projects WHERE username=? ORDER BY created_at DESC",
        (user,)
    ).fetchall()

    rows, idx = (len(projs) + 2) // 3, 0
    for _ in range(rows):
        cols = st.columns(3)
        for col in cols:
            if idx < len(projs):
                pid, name, client, study, phase, proj_type, created = projs[idx]  # Include proj_type
                with col:
                    st.markdown(
                        f"""
                        <div class="proj-card">
                            <h4>{name}</h4>
                            <p class="meta"><strong>Client:</strong> {client or '-'}</p>
                            <p class="meta"><strong>Study:</strong> {study or '-'}</p>
                            <p class="meta"><strong>Phase:</strong> {phase or '-'}</p>
                            <p class="meta"><strong>Type:</strong> {proj_type or '-'}</p>
                            <p class="date">Created on {created}</p>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Buttons (two equal columns inside the card)
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button("🔍 View", key=f"view_{pid}", use_container_width=True):
                            st.session_state.current_project = name
                            st.rerun()
                    with bcol2:
                        if st.button("🗑 Remove", key=f"remove_{pid}", use_container_width=True):
                            db.execute("DELETE FROM projects WHERE id=?", (pid,))
                            db.commit()
                            st.rerun()

                    st.markdown("</div>", unsafe_allow_html=True)  # close proj‑card
                idx += 1
            else:
                col.empty()





    # ─────────────────────────  ADD A NEW PROJECT  ─────────────────────────
    st.markdown("---")
    st.markdown("### ➕ Add a New Project")

    # Extra CSS for form styling with light blue borders
    st.markdown("""
    <style>
        /* Form container styling */
        .stForm {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            border: 1px solid #c0d6f9;
        }
        
        /* Input field styling with light blue border */
        .stTextInput > div > div > input {
            background-color: white;
            border: 1px solid #c0d6f9 !important;
            border-radius: 6px !important;
            padding: 10px !important;
            transition: all 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Textarea styling with light blue border */
        .stTextArea > div > div > textarea {
            background-color: white;
            border: 1px solid #c0d6f9 !important;
            border-radius: 6px !important;
            padding: 10px !important;
            transition: all 0.3s ease;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Selectbox styling with light blue border */
        div[data-baseweb="select"] > div {
            background-color: white;
            border: 1px solid #c0d6f9 !important;
            border-radius: 6px !important;
            transition: all 0.3s ease;
        }
        div[data-baseweb="select"]:focus-within > div {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Submit button styling */
        .stButton button {
            background-color: #3b82f6 !important;
            color: white !important;
            border-radius: 6px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton button:hover {
            background-color: #2563eb !important;
        }
        
            /* card wrapper */
    .proj-card{
      border:1px solid #e0e0e0;border-radius:16px;background:#fff;
      padding:1rem;height:230px;display:flex;flex-direction:column;
      justify-content:space-between;box-shadow:0 2px 6px rgba(0,0,0,.08);
      transition:transform .25s,box-shadow .25s;
    }
    .proj-card:hover{transform:translateY(-6px);box-shadow:0 8px 16px rgba(0,0,0,.15);}
    .proj-card h4{margin:0;font-size:1.25rem;font-weight:600;color:var(--qm-dark);}
    .proj-card .meta{margin:0;font-size:.8rem;color:#555;}
        
    </style>
    """, unsafe_allow_html=True)

    with st.form(key="new_project_form"):
        c1, c2 = st.columns(2)
        with c1:
            new_name = st.text_input("Project title *", key="np_name")
            new_study = st.text_input("Study name *", key="np_study")
        with c2:
            new_client = st.text_input("Client name *", key="np_client")
            new_phase = st.text_input("Phase *", key="np_phase")

        # Project type dropdown (preserved)
        proj_type = st.selectbox(
            "Project type *",
            ("Promo Analysis", "Category Forecasting", "MMM"),
            key="np_type"
        )

        new_desc = st.text_area(
            "Description (optional)",
            key="np_desc",
            placeholder="Brief project description…",
            height=90
        )

        submitted = st.form_submit_button("Create Project")

        if submitted:
            # validation
            missing = None
            if not new_name.strip():
                missing = "Project title"
            elif not new_client.strip():
                missing = "Client name"
            elif not new_study.strip():
                missing = "Study name"
            elif not new_phase.strip():
                missing = "Phase"
            if missing:
                st.error(f"❌ {missing} is required")
            else:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                db.execute(
                    """INSERT INTO projects
                    (username, name, client, study, phase,
                        description, type, created_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        user,
                        new_name.strip(),
                        new_client.strip(),
                        new_study.strip(),
                        new_phase.strip(),
                        new_desc.strip(),
                        proj_type,
                        ts,
                    ),
                )
                db.commit()
                st.success("✅ Project created!")
                st.rerun()


    # ───── Extra CSS (buttons & inputs) ─────
    st.markdown(
        """
        <style>
        /* card wrapper */
        .proj-card{
          border:1px solid #e0e0e0;border-radius:16px;background:#fff;
          padding:1rem;height:230px;display:flex;flex-direction:column;
          justify-content:space-between;box-shadow:0 2px 6px rgba(0,0,0,.08);
          transition:transform .25s,box-shadow .25s;
        }
        .proj-card:hover{transform:translateY(-6px);box-shadow:0 8px 16px rgba(0,0,0,.15);}
        .proj-card h4{margin:0;font-size:1.25rem;font-weight:600;color:var(--qm-dark);}
        .proj-card .meta{margin:0;font-size:.8rem;color:#555;}
        .proj-card .date{margin:.4rem 0 .8rem;font-size:.78rem;color:#666;}

        /* pill buttons in cards */
        .proj-card .stButton>button{
          width:100%;padding:.52rem 0;border:none;border-radius:30px;
          font-weight:600;color:#fff;background:var(--qm-blue);
          transition:background .25s;
        }
        .proj-card .stButton>button:hover{background:#3571c1;}

        /* remove button second in DOM ⤵ */
        .proj-card .stButton:nth-of-type(2)>button{
          background:transparent;color:var(--qm-yellow);border:2px solid var(--qm-yellow);
        }
        .proj-card .stButton:nth-of-type(2)>button:hover{
          background:var(--qm-yellow);color:var(--qm-dark);
        }

        /* dashed new‑project card */
        .new-card{
          border:2px dashed var(--qm-yellow);border-radius:16px;
          padding:1rem;text-align:center;
          display:flex;flex-direction:column;justify-content:center;
          transition:background .25s;
        }
        .new-card:hover{background:rgba(255,189,89,.07);}
        .form-desc{color:#666;font-size:.9rem;margin-bottom:15px;}

        .new-card input,.new-card textarea{
          background:#fafafa!important;border:1px solid #e0e0e0!important;
          transition:border-color .3s,box-shadow .3s!important;
        }
        .new-card input:focus,.new-card textarea:focus{
          border-color:var(--qm-yellow)!important;box-shadow:0 0 0 1px rgba(255,189,89,.3)!important;
        }
        .new-card .stButton>button{
          background:linear-gradient(90deg,var(--qm-yellow),#ffcb7c);
          box-shadow:0 2px 5px rgba(255,189,89,.4);
          font-weight:600;transition:all .3s;
        }
        .new-card .stButton>button:hover{
          box-shadow:0 4px 8px rgba(255,189,89,.5);transform:translateY(-2px);
        }
        
        /* input focus accent */
.stTextInput input:focus, .stTextArea textarea:focus{
  border:2px solid var(--qm-yellow)!important; box-shadow:none!important;
}

/* create‑project pill */
button[kind="secondary"]:has(span:contains('Create Project')){
  padding:.55rem 0;border:none;border-radius:30px;
  background:var(--qm-yellow);color:#333;font-weight:600;
  box-shadow:0 2px 5px rgba(255,189,89,.4);transition:background .25s,transform .2s;
}
button[kind="secondary"]:has(span:contains('Create Project')):hover{
  background:#ffd37e;transform:translateY(-2px);
}
        </style>
        """,
        unsafe_allow_html=True,
    )
