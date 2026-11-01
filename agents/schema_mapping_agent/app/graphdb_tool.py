from graphdb import GraphDatabase
import json
from typing import Dict, Any

class GraphDBAgent:
    def __init__(self, uri: str, user: str, password: str, database: str = "graphdb"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
    
    def clear_database(self):
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_indexes(self):
        with self.driver.session(database=self.database) as session:
            queries = [
                "CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)",
                "CREATE INDEX type_name IF NOT EXISTS FOR (t:Type) ON (t.name)",
                "CREATE INDEX field_name IF NOT EXISTS FOR (f:Field) ON (f.name)",
                "CREATE INDEX field_full_name IF NOT EXISTS FOR (f:Field) ON (f.full_name)"
            ]
            for query in queries:
                session.run(query)
    
    def add_product(self, product_name: str):
        with self.driver.session(database=self.database) as session:
            session.run("MERGE (p:Product {name: $name})", name=product_name)
    
    def add_type(self, product_name: str, type_name: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (p:Product {name: $product_name})
                MERGE (t:Type {name: $type_name, product: $product_name})
                MERGE (p)-[:HAS_TYPE]->(t)
                """,
                product_name=product_name,
                type_name=type_name
            )
    
    def add_field(self, product_name: str, type_name: str, field_name: str):
        full_field_name = f"{type_name}.{field_name}"
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (t:Type {name: $type_name, product: $product_name})
                MERGE (f:Field {name: $field_name, full_name: $full_field_name, type: $type_name, product: $product_name})
                MERGE (t)-[:HAS_FIELD]->(f)
                """,
                product_name=product_name,
                type_name=type_name,
                field_name=field_name,
                full_field_name=full_field_name
            )
    
    def add_bidirectional_field_mapping(self, src_product: str, src_field_full: str, 
                                      tgt_product: str, tgt_field_full: str, justification: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (sf:Field {full_name: $src_field, product: $src_product})
                MATCH (tf:Field {full_name: $tgt_field, product: $tgt_product})
                MERGE (sf)-[:FIELD_MAPPING {justification: $justification}]->(tf)
                MERGE (tf)-[:FIELD_MAPPING {justification: $justification}]->(sf)
                """,
                src_product=src_product,
                tgt_product=tgt_product,
                src_field=src_field_full,
                tgt_field=tgt_field_full,
                justification=justification
            )
    
    def add_type_mapping_once(self, src_product: str, src_type: str, 
                            tgt_product: str, tgt_type: str, justification: str):
        key = tuple(sorted([(src_product, src_type), (tgt_product, tgt_type)]))
        if key in self.type_mappings_added:
            return
        self.type_mappings_added.add(key)
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (st:Type {name: $src_type, product: $src_product})
                MATCH (tt:Type {name: $tgt_type, product: $tgt_product})
                MERGE (st)-[:TYPE_MAPPING {justification: $justification}]->(tt)
                """,
                src_product=src_product,
                src_type=src_type,
                tgt_product=tgt_product,
                tgt_type=tgt_type,
                justification=justification
            )

    def process_mapping_data(self, mapping_data: Dict[str, Any]):
        product_name = mapping_data["name"].split("_")[0]
        self.add_product(product_name)

        self.type_mappings_added = set()

        properties = mapping_data["properties"]

        for src_field_full, mappings in properties.items():
            src_type, src_field = src_field_full.split(".", 1)
            self.add_type(product_name, src_type)
            self.add_field(product_name, src_type, src_field)

            for mapping in mappings:
                tgt_schema = mapping["mapTable"]
                tgt_product = tgt_schema.split("_")[0]
                tgt_type, tgt_field = mapping["mapProperty"].split(".", 1)
                justification = mapping["justification"]

                self.add_product(tgt_product)
                self.add_type(tgt_product, tgt_type)
                self.add_field(tgt_product, tgt_type, tgt_field)

                self.add_bidirectional_field_mapping(
                    product_name, src_field_full,
                    tgt_product, f"{tgt_type}.{tgt_field}",
                    justification
                )

                self.add_type_mapping_once(
                    product_name, src_type,
                    tgt_product, tgt_type,
                    f"Types related through at least one field mapping"
                )

def populate_graphdb_graph(mapping_data: Dict,
                         uri: str = "bolt://internal-host:7687",
                         user: str = "graphdb",
                         password: str = "newpassword123") -> str:
    """
    Populate GraphDB graph database with mapping data from a parsed JSON object.
    
    Args:
        mapping_data: A dictionary containing the mapping information
        uri: GraphDB database URI (optional)
        user: GraphDB username (optional)
        password: GraphDB password (optional)
    
    Returns:
        Status message with database statistics
    """
    graphdb_agent = None
    try:
        # Connect to GraphDB
        graphdb_agent = GraphDBAgent(uri, user, password)
        
        # Clear database and create indexes
        # graphdb_agent.clear_database()
        # graphdb_agent.create_indexes()
                
        graphdb_agent.process_mapping_data(mapping_data)
        
        # Get statistics
        with graphdb_agent.driver.session(database="graphdb") as session:
            node_stats = {}
            for record in session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count"):
                node_stats[str(record['labels'])] = record['count']
            
            rel_stats = {}
            for record in session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count"):
                rel_stats[record['rel_type']] = record['count']
        
        # Format result
        result = f"Successfully populated GraphDB graph\n\n"
        result += "Node counts:\n"
        for label, count in node_stats.items():
            result += f"  {label}: {count}\n"
        result += "\nRelationship counts:\n"
        for rel_type, count in rel_stats.items():
            result += f"  {rel_type}: {count}\n"
        
        return result
        
    except Exception as e:
        return f"Error populating GraphDB: {str(e)}"
    finally:
        if graphdb_agent:
            graphdb_agent.close()


if __name__ == "__main__":
    print(populate_graphdb_graph("~/mydata/resilience_application-optimizer_entity.json"))
