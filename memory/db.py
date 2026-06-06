# memory.py
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict
from logger import get_logger

logger = get_logger("memory.db")

DB_PATH = "parenting_memory.db"

def init_db():
    """初始化数据库表（首次自动创建）"""
    logger.info("初始化数据库...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 宝宝信息表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS babies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            birth_date TEXT,           -- 格式 YYYY-MM-DD
            allergy TEXT,              -- 过敏史，如 "牛奶蛋白"
            notes TEXT,                -- 额外备注
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")

def find_baby_by_name(name: str) -> Optional[Dict]:
    """根据名字查找宝宝（模糊匹配）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM babies WHERE name LIKE ?", (f"%{name}%",))
    row = cursor.fetchone()
    conn.close()
    result = dict(row) if row else None
    logger.debug(f"查找宝宝: name={name}, 找到={result is not None}")
    return result

def upsert_baby(name: str, birth_date: Optional[str] = None,
                allergy: Optional[str] = None, notes: Optional[str] = None) -> int:
    """如果宝宝存在则更新，不存在则创建，返回 baby_id"""
    existing = find_baby_by_name(name)
    if existing:
        update_baby(existing['id'], birth_date=birth_date, allergy=allergy, notes=notes)
        logger.debug(f"更新现有宝宝: name={name}, id={existing['id']}")
        return existing['id']
    else:
        baby_id = create_baby(name, birth_date, allergy, notes)
        logger.debug(f"创建新宝宝: name={name}, id={baby_id}")
        return baby_id

def create_baby(name: str, birth_date: Optional[str] = None,
                allergy: Optional[str] = None, notes: Optional[str] = None) -> int:
    """创建宝宝档案，返回 baby_id"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO babies (name, birth_date, allergy, notes) VALUES (?, ?, ?, ?)",
        (name, birth_date, allergy, notes)
    )
    baby_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"创建宝宝档案: id={baby_id}, name={name}")
    return baby_id

def get_baby(baby_id: int) -> Optional[Dict]:
    """根据 ID 获取宝宝信息"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM babies WHERE id = ?", (baby_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    logger.warning(f"未找到宝宝: baby_id={baby_id}")
    return None

def update_baby(baby_id: int, **kwargs) -> bool:
    """更新宝宝信息，如 update_baby(1, allergy="牛奶蛋白", notes="已添加辅食")"""
    allowed_fields = ["name", "birth_date", "allergy", "notes"]
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return False

    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [baby_id]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        f"UPDATE babies SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
        values
    )
    conn.commit()
    conn.close()
    logger.info(f"更新宝宝档案: id={baby_id}, 字段={list(updates.keys())}")
    return True

def get_all_babies() -> list:
    """获取所有宝宝列表"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, birth_date FROM babies ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    result = [dict(row) for row in rows]
    logger.debug(f"获取所有宝宝: 共 {len(result)} 个")
    return result

# 模块加载时自动初始化
init_db()
