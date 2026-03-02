"""Ad-hoc SQL exploration of the data lake using DuckDB.

Usage:
  python src/explore.py "SELECT count(*) FROM delta_scan('s3://taxi-lake/silver/trips')"
  python src/explore.py  # interactive mode
"""

import os
import sys

import duckdb


def get_connection():
    """Create DuckDB connection with S3/MinIO credentials."""
    conn = duckdb.connect()
    conn.execute("INSTALL delta; LOAD delta;")
    conn.execute(f"""
        SET s3_endpoint='{os.environ["MLFLOW_S3_ENDPOINT_URL"].replace("http://", "")}';
        SET s3_access_key_id='{os.environ["AWS_ACCESS_KEY_ID"]}';
        SET s3_secret_access_key='{os.environ["AWS_SECRET_ACCESS_KEY"]}';
        SET s3_use_ssl=false;
        SET s3_url_style='path';
    """)
    return conn


def main():
    conn = get_connection()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(conn.sql(query))
    else:
        print("DuckDB connected to taxi-lake. Tables:")
        print("  delta_scan('s3://taxi-lake/bronze/raw')")
        print("  delta_scan('s3://taxi-lake/bronze/streaming')")
        print("  delta_scan('s3://taxi-lake/silver/trips')")
        print("  delta_scan('s3://taxi-lake/silver/demand')")
        print("  delta_scan('s3://taxi-lake/gold/zone_stats')")
        print()
        print("Example: SELECT count(*) FROM delta_scan('s3://taxi-lake/silver/trips')")
        print()
        while True:
            try:
                query = input("sql> ")
                if query.strip().lower() in ("exit", "quit", "\\q"):
                    break
                if query.strip():
                    print(conn.sql(query))
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
