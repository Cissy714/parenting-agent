"""删除名为'宝宝'的错误档案（需确认）"""
import sqlite3

DB_PATH = "parenting_memory.db"

confirm = input("将删除名为'宝宝'的错误档案，确认？(y/n): ")
if confirm.lower() != "y":
    print("已取消。")
    exit(0)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("DELETE FROM babies WHERE name = '宝宝'")
conn.commit()
conn.close()
print(f"已删除 {cursor.rowcount} 条记录。")
