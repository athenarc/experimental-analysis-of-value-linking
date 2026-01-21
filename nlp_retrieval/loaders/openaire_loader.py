import psycopg2
from tqdm import tqdm
from nlp_retrieval.core.models import SearchableItem
from nlp_retrieval.loaders.loader_abc import BaseLoader

class OpenAireLoader(BaseLoader):
    def __init__(self, db_config, max_values=10000):
        self.db_config = db_config
        self.max_values = max_values
        self.schema_map = {} 
        self._discover_text_columns()

    def _discover_text_columns(self):
        print("--- Initializing OpenAIRE Loader ---")
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            query_cols = """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND data_type IN ('character varying', 'text', 'character', 'char');
            """
            cursor.execute(query_cols)
            rows = cursor.fetchall()
            for table, col in rows:
                if table not in self.schema_map:
                    self.schema_map[table] = []
                self.schema_map[table].append(col)
            conn.close()
            print(f"Found {sum(len(cols) for cols in self.schema_map.values())} text columns.")
        except Exception as e:
            print(f"Error discovering columns: {e}")

    def load(self):
        """
        Yields SearchableItem objects one by one (Generator).
        """
        total_yielded = 0
        
        seen_hashes = set()
        
        print(f"Streaming up to {self.max_values} distinct values...")

        for table, columns in self.schema_map.items():
            if total_yielded >= self.max_values:
                break

            for col_name in columns:
                if total_yielded >= self.max_values:
                    break
                
                remaining = self.max_values - total_yielded
                
                try:
                    with psycopg2.connect(**self.db_config) as conn:
                        with conn.cursor(name='server_side_stream') as cursor:
                            
                            query = f'SELECT "{col_name}" FROM public."{table}" WHERE "{col_name}" IS NOT NULL LIMIT {remaining * 2};'
                            
                            cursor.execute(query)
                            
                            while True:
                                rows = cursor.fetchmany(10000)
                                if not rows:
                                    break
                                
                                for row in rows:
                                    if total_yielded >= self.max_values:
                                        break

                                    val = row[0]
                                    if val and isinstance(val, str):
                                        clean_val = val.strip()
                                        
                                        if 0 < len(clean_val) < 1000:
                                            val_hash = hash(clean_val)
                                            
                                            if val_hash not in seen_hashes:
                                                seen_hashes.add(val_hash)
                                                
                                                yield SearchableItem(
                                                    item_id=f"{table}.{col_name}.{total_yielded}",
                                                    content=clean_val,
                                                    metadata={"table": table, "column": col_name}
                                                )
                                                total_yielded += 1
                                                
                                                if total_yielded % 100000 == 0:
                                                    print(f"Loaded {total_yielded} items...", end='\r')

                                if total_yielded >= self.max_values:
                                    break
                except Exception as e:
                    print(f"\nSkipping {table}.{col_name} due to error: {e}")
                    continue

        # Clear memory
        del seen_hashes
        print(f"\nFinished streaming. Total items: {total_yielded}")