# graph_builder.py
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import ast
import time
#from config import NEO4J_URI, NEO4J_USER, 

NEO4J_URI = "bolt://localhost:7687" 
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ibm_password" # <--- Ensure this is your correct password

# ============================================================
# TOPIC & METAPOLICY DEFINITIONS
# ============================================================

TOPIC_ASSIGNMENTS = {
    1: "Politics & Governance",
    2: "Law & Justice",
    3: "Law & Justice",
    4: "Health & Ethics",
    5: "Economy & Society",
    6: "Politics & Governance",
    7: "Social & Cultural Issues",
    8: "Economy & Society",
    9: "Social & Cultural Issues"
}

METAPOLICY_NAMES = {
    1: "Political Governance",
    2: "Criminal Justice & Law",
    3: "Civil Liberties & Rights",
    4: "Bioethics & Personal Autonomy",
    5: "Economic & Labor Policy",
    6: "Religion & Secularism",
    7: "Gender, Family & Social Norms",
    8: "Education, Media & Technology",
    9: "Animal Welfare & Environment"
}


class DebateGraphBuilder:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # ----------------------------------------------------------
    # CONSTRAINTS
    # ----------------------------------------------------------
    def create_constraints(self):
        print("Creating constraints...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MetaPolicy) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Policy) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Argument) REQUIRE a.arg_id IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                session.run(c)
        print("  Constraints created ✅")

    def create_vector_index(self):
        print("\nCreating vector index for Argument embeddings...")

        query = """
        CREATE VECTOR INDEX argument_embedding_index IF NOT EXISTS
        FOR (a:Argument)
        ON (a.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }
        }
        """

        with self.driver.session() as session:
            session.run(query)

        print("  Vector index created ✅")    

    # ----------------------------------------------------------
    # PHASE 1 — Topic nodes (5 nodes, runs instantly)
    # ----------------------------------------------------------
    def ingest_topics(self):
        print("\nIngesting Topic nodes...")
        topics = list(set(TOPIC_ASSIGNMENTS.values()))
        with self.driver.session() as session:
            for topic_name in topics:
                session.run(
                    "MERGE (t:Topic {name: $name})",
                    name=topic_name
                )
        print(f"  {len(topics)} Topic nodes created ✅")

    # ----------------------------------------------------------
    # PHASE 2 — MetaPolicy nodes + :BELONGS_TO Topic
    # ----------------------------------------------------------
    def ingest_metapolicies(self):
        print("\nIngesting MetaPolicy nodes...")
        with self.driver.session() as session:
            for mp_id, mp_name in METAPOLICY_NAMES.items():
                topic_name = TOPIC_ASSIGNMENTS[mp_id]
                session.run("""
                    MERGE (m:MetaPolicy {id: $id})
                    SET m.name = $name
                    WITH m
                    MATCH (t:Topic {name: $topic_name})
                    MERGE (m)-[:BELONGS_TO]->(t)
                """, id=mp_id, name=mp_name, topic_name=topic_name)
        print(f"  {len(METAPOLICY_NAMES)} MetaPolicy nodes created ✅")

    # ----------------------------------------------------------
    # PHASE 3 — Policy nodes + :BELONGS_TO MetaPolicy
    # ----------------------------------------------------------
    def ingest_policies(self, policy_df):
        print("\nIngesting Policy nodes...")
        with self.driver.session() as session:
            for _, row in policy_df.iterrows():
                session.run("""
                    MERGE (p:Policy {name: $name})
                    WITH p
                    MATCH (m:MetaPolicy {id: $metapolicy_id})
                    MERGE (p)-[:BELONGS_TO]->(m)
                """, name=row['policy'],
                     metapolicy_id=int(row['metapolicy_id']))
        print(f"  {len(policy_df)} Policy nodes created ✅")

    # ----------------------------------------------------------
    # PHASE 4 — Argument nodes + :SUPPORTS/:ATTACKS Policy
    # ----------------------------------------------------------
    def ingest_arguments(self, df, embeddings, batch_size=500):
        print(f"\nIngesting {len(df)} Argument nodes in batches of {batch_size}...")

        records = []
        for idx, row in df.iterrows():
            records.append({
                'arg_id':     row['arg_id'],
                'text':       row['argument'],
                'stance':     row['stance'],
                'confidence': float(row['stance_confidence']),
                'policy':     row['policy'],
                'embedding':  embeddings[idx].tolist()
            })

        query_supports = """
        UNWIND $data AS row
        MERGE (a:Argument {arg_id: row.arg_id})
        SET a.text       = row.text,
            a.stance     = row.stance,
            a.embedding  = row.embedding
        WITH a, row
        MATCH (p:Policy {name: row.policy})
        MERGE (a)-[:SUPPORTS {confidence: row.confidence}]->(p)
        """

        query_attacks = """
        UNWIND $data AS row
        MERGE (a:Argument {arg_id: row.arg_id})
        SET a.text       = row.text,
            a.stance     = row.stance,
            a.embedding  = row.embedding
        WITH a, row
        MATCH (p:Policy {name: row.policy})
        MERGE (a)-[:ATTACKS {confidence: row.confidence}]->(p)
        """

        pro_records = [r for r in records if r['stance'] == 1]
        con_records = [r for r in records if r['stance'] == -1]

        start = time.time()

        with self.driver.session() as session:
            # Ingest PRO arguments
            print(f"  Ingesting {len(pro_records)} PRO arguments...")
            for i in range(0, len(pro_records), batch_size):
                batch = pro_records[i:i + batch_size]
                session.run(query_supports, data=batch)
                print(f"    PRO: {min(i + batch_size, len(pro_records))}"
                      f"/{len(pro_records)}")

            # Ingest CON arguments
            print(f"  Ingesting {len(con_records)} CON arguments...")
            for i in range(0, len(con_records), batch_size):
                batch = con_records[i:i + batch_size]
                session.run(query_attacks, data=batch)
                print(f"    CON: {min(i + batch_size, len(con_records))}"
                      f"/{len(con_records)}")

        elapsed = round(time.time() - start, 2)
        print(f"  Arguments ingested in {elapsed}s ✅")

    # ----------------------------------------------------------
    # VERIFICATION
    # ----------------------------------------------------------
    def verify_graph(self):
        print("\nVerifying graph...")
        checks = [
            ("Argument nodes",  "MATCH (a:Argument) RETURN count(a) AS c"),
            ("Policy nodes",    "MATCH (p:Policy) RETURN count(p) AS c"),
            ("MetaPolicy nodes","MATCH (m:MetaPolicy) RETURN count(m) AS c"),
            ("Topic nodes",     "MATCH (t:Topic) RETURN count(t) AS c"),
            (":SUPPORTS edges", "MATCH ()-[r:SUPPORTS]->() RETURN count(r) AS c"),
            (":ATTACKS edges",  "MATCH ()-[r:ATTACKS]->() RETURN count(r) AS c"),
            ("Policy→MetaPolicy","MATCH ()-[r:BELONGS_TO]->(:MetaPolicy) RETURN count(r) AS c"),
            ("MetaPolicy→Topic","MATCH ()-[r:BELONGS_TO]->(:Topic) RETURN count(r) AS c"),
        ]

        orphan_checks = [
            ("Orphan policies",
             "MATCH (p:Policy) WHERE NOT (p)-[:BELONGS_TO]->() RETURN p.name"),
            ("Orphan arguments",
             "MATCH (a:Argument) WHERE NOT (a)-[:SUPPORTS|ATTACKS]->() "
             "RETURN count(a) AS c"),
            ("Orphan MetaPolicies",
             "MATCH (m:MetaPolicy) WHERE NOT (m)-[:BELONGS_TO]->() RETURN m.name"),
        ]

        with self.driver.session() as session:
            print("\n  Node & relationship counts:")
            for label, query in checks:
                result = session.run(query).single()
                print(f"    {label:<25} {result['c']}")

            print("\n  Orphan checks:")
            for label, query in orphan_checks:
                results = list(session.run(query))
                if 'c' in results[0].keys():
                    val = results[0]['c']
                    status = "✅" if val == 0 else "⚠️"
                    print(f"    {label:<25} {val} {status}")
                else:
                    names = [r[0] for r in results]
                    status = "✅" if not names else "⚠️"
                    print(f"    {label:<25} {names or 'None'} {status}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # --- Load data ---
    print("Loading data...")
    df = pd.read_csv('arguments_clean.csv')
    policy_df = pd.read_csv('policies.csv')
    embeddings = np.load('argument_embeddings.npy')

    print(f"  Arguments:  {len(df)}")
    print(f"  Policies:   {len(policy_df)}")
    print(f"  Embeddings: {embeddings.shape}")

    # --- Verify alignment ---
    assert len(df) == embeddings.shape[0], \
        "Mismatch between arguments CSV and embeddings matrix"
    assert embeddings.shape[1] == 768, \
        "Unexpected embedding dimension"
    print("  Alignment check passed ✅")

    # --- Build graph ---
    builder = DebateGraphBuilder()

    try:
        builder.create_constraints()
        builder.create_vector_index()   
        builder.ingest_topics()
        builder.ingest_metapolicies()
        builder.ingest_policies(policy_df)
        builder.ingest_arguments(df, embeddings, batch_size=500)
        builder.verify_graph()
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise
    finally:
        builder.close()

    print("\nGraph build complete ✅")