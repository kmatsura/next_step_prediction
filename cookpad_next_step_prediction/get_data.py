import os
import pandas as pd
import mysql.connector as mydb
from dotenv import load_dotenv


def get_title_and_category(conn):
    """
    cookpad_dataからレシピ名とそれに対応する手順を持ってくる。
    """
    cur = conn.cursor()
    sql = "SELECT r.title, st.memo, st.position FROM recipes r INNER JOIN steps st ON r.id = st.recipe_id;"  # レシピのタイトルと対応する手順をもってくる。
    result = cur.execute(sql)
    rows = cur.fetchall()
    return result, rows


def get_data():
    """
    sqlで生データを取ってきて、必要な箇所をcsvの形で保存。
    """
    load_dotenv(verbose=True)
    conn = mydb.connect(
        host=os.environ.get("HOST_NAME"),
        user=os.environ.get("USER_NAME"),
        password=os.environ.get("PASSWORD"),
        database=os.environ.get("DB_NAME")
    )
    conn.ping(reconnect=True)
    assert conn.is_connected(), "connection error"
    result, rows = get_title_and_category(conn)
    if result == 0:
        print("No Data")
    tmp_dict = {}
    for i, row in enumerate(rows):
        title, memo, pos = row
        tmp_dict[i] = title, memo, pos  # df.append()は遅いので、dictを使う。
    datasets = pd.DataFrame.from_dict(
        tmp_dict, orient='index', columns=["title", "memo", "pos"])
    BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUTDIR = os.path.join(BASEDIR, os.environ.get("DATADIR"))
    FILENAME = "raw_data.csv"
    if not os.path.exists(OUTPUTDIR):
        os.makedirs(OUTPUTDIR)
    datasets.to_csv(os.path.join(OUTPUTDIR, FILENAME))


def main():
    get_data()


if __name__ == "__main__":
    main()
