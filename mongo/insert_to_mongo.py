import argparse
import os
import pandas as pd
import numpy as np
from typing import Optional
from pymongo import MongoClient
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def build_review(row: pd.Series) -> str:
    pos = row.get("Positive_Review", "")
    neg = row.get("Negative_Review", "")

    if isinstance(pos, str) and pos.strip().lower() == "no positive":
        pos = ""
    if isinstance(neg, str) and neg.strip().lower() == "no negative":
        neg = ""

    if pos and neg:
        return f"{pos} {neg}"
    return pos or neg or ""


def process_chunk(df: pd.DataFrame, score_threshold: float) -> pd.DataFrame:
    # Ensure numeric types for scores used elsewhere in the project
    if "Reviewer_Score" in df.columns:
        df["Reviewer_Score"] = pd.to_numeric(df["Reviewer_Score"], errors="coerce")
    if "Average_Score" in df.columns:
        df["Average_Score"] = pd.to_numeric(df["Average_Score"], errors="coerce")

    # Create a unified numeric score field and derive positive
    if "Reviewer_Score" in df.columns:
        df["score"] = df["Reviewer_Score"]
    elif "Average_Score" in df.columns:
        df["score"] = df["Average_Score"]
    else:
        df["score"] = np.nan

    # Derive a boolean sentiment label used by the app/models
    df["positive"] = pd.to_numeric(df["score"], errors="coerce").fillna(0) >= score_threshold

    # Create a single free-text field for ML scripts
    df["review"] = df.apply(build_review, axis=1)

    return df


def insert(
    csv_path: str,
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    chunksize: int,
    score_threshold: float,
    drop: bool,
    limit: Optional[int] = None,
) -> int:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection_name]

    if drop:
        coll.drop()

    total_inserted = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk = process_chunk(chunk, score_threshold)

        # Replace NaN with None to keep BSON clean
        records = chunk.replace({np.nan: None}).to_dict("records")

        if limit is not None and limit > 0:
            remaining = limit - total_inserted
            if remaining <= 0:
                break
            records = records[:remaining]

        if records:
            coll.insert_many(records, ordered=False)
            total_inserted += len(records)
            print(f"Inserted {total_inserted} records...", flush=True)

        if limit is not None and total_inserted >= limit:
            break

    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Insert hotel reviews CSV into MongoDB")
    parser.add_argument(
        "--csv",
        default="data/Hotel_Reviews.csv",
        help="Path to Hotel_Reviews.csv",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=int(os.getenv("INSERT_CHUNKSIZE")),
        help="CSV chunk size for streaming inserts",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=float(os.getenv("SCORE_THRESHOLD")),
        help="Threshold for positive sentiment based on Reviewer_Score",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the target collection before inserting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Insert only the first N rows (0 = all)",
    )

    args = parser.parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None

    total = insert(
        csv_path=args.csv,
        mongo_uri=os.getenv("MONGO_URI"),
        db_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        chunksize=args.chunksize,
        score_threshold=args.score_threshold,
        drop=args.drop,
        limit=limit,
    )

    print(f"Done. Inserted {total} documents into {args.db}.{args.collection}")


if __name__ == "__main__":
    main()
