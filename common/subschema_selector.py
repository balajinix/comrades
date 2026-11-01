import re
import sys
import os
import json
from tqdm import tqdm
sys.path.append(os.path.abspath("./agents"))
from utils.map import index_chroma_subschema, query_chroma_subschema

def extract_field_type(field_type: str):
    t = field_type.replace("[","").replace("]","").replace("!","")
    return t


# Function to parse GraphQL schema and create dictionary of types and their field types/enums/inputs.
def parse_graphql_schema(schema):
    # Regular expressions to match types, enums, and inputs.
    scalar_regex = re.compile(r'scalar\s+(\w+)')
    scalar_regex = re.compile(r'scalar\s+(\w+)')
    type_regex = re.compile(r'type\s+(\w+)(\s+implements\s+\w+)?\s*{')
    input_regex = re.compile(r'input\s+(\w+)\s*{')
    enum_regex = re.compile(r'enum\s+(\w+)\s*{')
    field_regex = re.compile(r'\s*(\w+)\s*:\s*([\w\[\]!]+)')
    ret_type_regex = re.compile(r"\s*\):\s*(?:\[(\w+)\]|(\w+))")
    
    # Initialize dictionary to store types and their field types/enums/inputs.
    type_dict = {}
    code_dict = {}
    
    # Parse schema to build dictionary.
    lines = schema.split('\n')
    current_type_enum_input = None
    
    for line in lines:
        scalar_match = scalar_regex.match(line)
        type_match = type_regex.match(line)
        input_match = input_regex.match(line)
        enum_match = enum_regex.match(line)
        ret_type_match = ret_type_regex.match(line)
        if scalar_match:
            current_type_enum_input = scalar_match.group(1)
            code_dict[current_type_enum_input] = line + "\n"
            if current_type_enum_input not in type_dict:
                type_dict[current_type_enum_input] = []
            current_type_enum_input = None
        elif type_match:
            current_type_enum_input = type_match.group(1)
            code_dict[current_type_enum_input] = line + "\n"
        elif input_match:
            current_type_enum_input = input_match.group(1)
            code_dict[current_type_enum_input] = line + "\n"
        elif enum_match:
            current_type_enum_input = enum_match.group(1)
            code_dict[current_type_enum_input] = line + "\n"
        elif current_type_enum_input and current_type_enum_input not in ["Query", "Mutation", ""]:
            if current_type_enum_input not in type_dict:
                type_dict[current_type_enum_input] = []
            field_match = field_regex.match(line)
            if field_match:
                field_type = field_match.group(2).strip('[]!')
                if field_type not in ['String', 'DateTime', 'Int', 'Float', 'Boolean', 'ID', 'JSON', '']:
                    type_dict[current_type_enum_input].append(field_type)
            if ret_type_match:
                ret_type = ret_type_match.group(1) or ret_type_match.group(2)
                if ret_type not in ['String', 'DateTime', 'Int', 'Float', 'Boolean', 'ID', 'JSON', '']:
                    type_dict[current_type_enum_input].append(ret_type)

            code_dict[current_type_enum_input] += line + "\n"
            if line.startswith("}"):
                current_type_enum_input = None

        elif current_type_enum_input in ["Query", "Mutation"]:
            code_dict[current_type_enum_input] += line + "\n"
            if line.startswith("}"):
                current_type_enum_input = None


    
    type_dependency_dict = {}
    for x in type_dict:
        if x not in ["Query", "Mutation", ""]:
            type_dependency_dict[x] = []
            for f in type_dict[x]:
                if f in type_dict:
                    type_dependency_dict[x].append(f)

    return type_dependency_dict, code_dict

# Function to generate connected components from the given dictionary
def generate_connected_components(type_dict):
    # Create a graph from the type_dict
    graph = {}
    for key, values in type_dict.items():
        if key not in graph:
            graph[key] = []
        for value in values:
            graph[key].append(value)
            if value not in graph:
                graph[value] = []
            graph[value].append(key)
    
    # Function to perform DFS and find all nodes in a connected component
    def dfs(node, visited, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, component)
    
    # Find all connected components
    visited = set()
    components = []
    for node in graph:
        if node == '':
            print("hello")
            continue
        if node not in visited:
            component = []
            dfs(node, visited, component)
            components.append(component)
    
    return components


def sanitize_query_block(schema: str):
    clean_schema = ""
    for line in schema.splitlines():
        stripped = line.strip()

        if stripped != '"""' and stripped.startswith('"') and stripped.endswith('"'):
            continue
        clean_schema += line + "\n"
    return clean_schema
    

def split_queries(graphql_schema):
    # Split the schema into individual queries using regex
    queries = re.findall(r'(\w+\s*(\(.*?\))?\s*:\s*\[?\w+\]?\s*@graphql\s*\(.*?\))', graphql_schema, re.DOTALL)
    return queries

def extract_parameter_types(parameter_block):
    # Use regular expression to find all parameter types
    parameter_types = re.findall(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)', parameter_block)
    return parameter_types

def extract_query_return_type_and_dependencies(query_block: str, param_block: str, type_dependency_dict: dict):
    d = type_dependency_dict.copy()
    if len(param_block)>0:
        part1 = query_block.split(param_block)[0]
        words = re.findall(r"\w+", part1)
        query_name = words[-1]
        part2 = query_block.split(param_block)[1]
        ret_type = part2.split("\n",1)[0]
        ret_type = ret_type.replace(":","").replace("[","").replace("]","").strip()
        param_types = extract_parameter_types(param_block)
        for p in param_types:
            if p in type_dependency_dict and p not in type_dependency_dict[ret_type]:
                d[ret_type].append(p)
                d[p].append(ret_type)
            
    else:   
        words = re.findall(r"\w+", query_block)
        query_name, ret_type = words[0].replace(":",""), words[1]

    return query_name, ret_type, d


def split_schema(schema):
    schema_parts = {}
    type_dependency_dict, code_dict = parse_graphql_schema(schema)
    query_schema = code_dict["Query"]
    query_schema_text = sanitize_query_block(query_schema)
    query_ret_types = {}
    query_blocks = {}
    for query in split_queries(query_schema_text):
        query_block = query[0]
        param_block = query[1]
        query_name, ret_type, type_dependency_dict = extract_query_return_type_and_dependencies(query_block, param_block, type_dependency_dict)
        query_ret_types[query_name] = ret_type
        query_blocks[query_name] = query_block
        
    components = generate_connected_components(type_dependency_dict)
    for query_name in query_ret_types:
        ret_type = query_ret_types[query_name]
        query_block = query_blocks[query_name]
        comp = []
        for c in components:
            if ret_type in c:
                comp = c

        s = ""
        for x in comp:
            s +=  code_dict[x] + "\n"
        
        s += "type Query {\n  " + query_block + "\n}"
        schema_parts[query_name] = s

    return schema_parts, type_dependency_dict, components

def index_subschema(schema_parts):
    collection_name = "subschema"
    persist_directory = "chroma_dir"
    index_chroma_subschema(collection_name, persist_directory, schema_parts)

def select_subschema(nl_query, schema_parts):
    collection_name = "subschema"
    persist_directory = "chroma_dir"
    query_texts = [nl_query]
    query_class_relevance = query_chroma_subschema(collection_name, persist_directory, query_texts)
    query_name = query_class_relevance[0][0]
    return schema_parts[query_name]

def select_subschema_manual(query_name, schema_parts):
    return schema_parts[query_name]

