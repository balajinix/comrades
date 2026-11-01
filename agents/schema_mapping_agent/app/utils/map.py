import json
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import scipy
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def split_camel_case(name: str):
    split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    return " ".join(split)

def index_chroma(collection_name, persist_directory, data):
    embed_model = "all-MiniLM-L6-v2"
    client = chromadb.PersistentClient(path=persist_directory)
    documents_list, metadatas_list, ids_list = [],[],[]
    for i, op in enumerate(data):
        op_name = split_camel_case(op)
        op_desc = data[op]["description"]
        s = op_name + ": " + op_desc
        if "fields" in data[op]:
            for x in data[op]["fields"]:
                s += " " + split_camel_case(x) + ": " + data[op]["fields"][x]["description"]
        # s = op_desc
        doc = s
        metadata = {"source": collection_name, "class": op}
        documents_list.append(doc)
        metadatas_list.append(metadata)
        ids_list.append("doc_id_"+str(i))

    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    try:
        client.delete_collection(name=collection_name)
        print("Deleted the existing collection :",collection_name)  
    except:
        pass
    print("Creating collection:", collection_name)
    collection = client.create_collection(name=collection_name, embedding_function=embed_model_function, metadata={"hnsw:space": "cosine"})
    collection.add(
            documents= documents_list, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            metadatas= metadatas_list, # filter on these!
            ids=ids_list, # unique for each doc 
            )
    print("Documents indexed: ", len(documents_list))

def index_chroma_graphql(collection_name, persist_directory, data):
    embed_model = "all-MiniLM-L6-v2"
    client = chromadb.PersistentClient(path=persist_directory)
    documents_list, metadatas_list, ids_list = [],[],[]
    i = 0
    for db in data:
        for t in data[db]:
            type_desc = data[db][t]["description"]
            s = db + ": " + t + ": " + type_desc
            if "fields" in data[db][t]:
                for x in data[db][t]["fields"]:
                    s += " " + split_camel_case(x) + ": " + data[db][t]["fields"][x]["description"]
            # s = op_desc
            doc = s
            metadata = {"source": collection_name, "db_name": db, "type_name": t}
            documents_list.append(doc)
            metadatas_list.append(metadata)
            ids_list.append("doc_id_"+str(i))
            i += 1

    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    try:
        client.delete_collection(name=collection_name)
        print("Deleted the existing collection :",collection_name)  
    except Exception as e:
        print(str(e))
        pass
    print("Creating collection:", collection_name)
    collection = client.create_collection(name=collection_name, embedding_function=embed_model_function, metadata={"hnsw:space": "cosine"})
    collection.add(
            documents= documents_list, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            metadatas= metadatas_list, # filter on these!
            ids=ids_list, # unique for each doc 
            )
    print("Documents indexed: ", len(documents_list))

def query_chroma(collection_name, persist_directory, query_texts, recall_limit=10):
    embed_model = "all-MiniLM-L6-v2"
    # print("Querying chroma db collection:", collection_name)
    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name, embedding_function=embed_model_function)
    output = collection.query(
        query_texts=query_texts,
        n_results=recall_limit,
    )
    results = output.get('metadatas')
    query_class_relevance = []
    for res in results:
        rel_classes = [r.get('class') for r in res]
        query_class_relevance.append(rel_classes)
    # print("Query count :", len(query_class_relevance))
    return query_class_relevance

def query_chroma_graphql(collection_name, persist_directory, query_texts, recall_limit=10):
    embed_model = "all-MiniLM-L6-v2"
    # print("Querying chroma db collection:", collection_name)
    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name, embedding_function=embed_model_function)
    output = collection.query(
        query_texts=query_texts,
        n_results=recall_limit,
    )
    results = output.get('metadatas')
    query_class_relevance = []
    for res in results:
        rel_classes = [r.get("db_name")+"###"+r.get("type_name") for r in res]
        query_class_relevance.append(rel_classes)
    # print("Query count :", len(query_class_relevance))
    return query_class_relevance

def get_relevant_ops(src_data, op, target_collection, persist_directory, dst_graph):
    op_name = split_camel_case(op)
    op_desc = src_data[op]["description"]
    s = op_name + ": " + op_desc
    if "fields" in src_data[op]:
        for x in src_data[op]["fields"]:
            s += " " + split_camel_case(x) + ": " + src_data[op]["fields"][x]["description"]

    # s = op_desc

    matching_ops = query_chroma(target_collection, persist_directory, [s])[0]
    related_ops = []
    for op in matching_ops:
        related_ops += [x["related_op"] for x in dst_graph[op]]
    for op in related_ops:
        if op not in matching_ops:
            matching_ops.append(op)
    return matching_ops

def get_relevant_types(src_data, db_name, type_name, target_collection, persist_directory):
    type_desc = src_data[db_name][type_name]["description"]
    s = db_name + ": " + type_name + ": " + type_desc
    if "fields" in src_data[db_name][type_name]:
        for x in src_data[db_name][type_name]["fields"]:
            s += " " + split_camel_case(x) + ": " + src_data[db_name][type_name]["fields"][x]["description"]


    matching_types = query_chroma_graphql(target_collection, persist_directory, [s])[0]
    return matching_types


def get_relevant_fields(src_data, src_op, target_data, target_op):
    threshold = 0.3
    src_field_embeddings = {}
    if "fields" in src_data[src_op]:
        for field in src_data[src_op]["fields"]:
            src_field_embeddings[field] = embedding_model.encode(split_camel_case(field)+": "+src_data[src_op]["fields"][field]["description"])
            if "items" in src_data[src_op]["fields"] and "fields" in src_data[src_op]["fields"]["items"]:
                for item_field in src_data[src_op]["fields"]["items"]["fields"]:
                    src_field_embeddings[item_field] = embedding_model.encode(split_camel_case(item_field)+": "+src_data[src_op]["fields"]["items"]["fields"][item_field]["description"])
    target_field_embeddings = {}
    if "fields" in target_data[target_op]:
        for field in target_data[target_op]["fields"]:
            target_field_embeddings[field] = embedding_model.encode(split_camel_case(field)+": "+target_data[target_op]["fields"][field]["description"])
            if "items" in target_data[target_op]["fields"] and "fields" in target_data[target_op]["fields"]["items"]:
                for item_field in target_data[target_op]["fields"]["items"]["fields"]:
                    target_field_embeddings[item_field] = embedding_model.encode(split_camel_case(item_field)+": "+target_data[target_op]["fields"]["items"]["fields"][item_field]["description"])

    field_matches = {} 
    for src_field in src_field_embeddings:
        sim_scores = [(x, 1-scipy.spatial.distance.cosine(src_field_embeddings[src_field], target_field_embeddings[x])) for x in target_field_embeddings]
        max_score = 0
        match = 0
        for x in sim_scores:
            if x[1]>max_score:
                max_score = x[1]
                match = x[0]
        # print(max_score)
        if max_score > threshold:
            target_field = match
        else:
            target_field = ""
        field_matches[src_field] = target_field

    return field_matches

def index_chroma_search(collection_name, persist_directory, index_data):
    embed_model = "all-MiniLM-L6-v2"
    client = chromadb.PersistentClient(path=persist_directory)
    documents_list, metadatas_list, ids_list = [],[],[]
    for i, term in enumerate(index_data):
        metadata = {"class": json.dumps(index_data[term], indent=4)}
        documents_list.append(term)
        metadatas_list.append(metadata)
        ids_list.append("doc_id_"+str(i))

    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    try:
        client.delete_collection(name=collection_name)
        print("Deleted the existing collection :",collection_name)  
    except:
        pass
    print("Creating collection:", collection_name)
    i = 0
    collection = client.create_collection(name=collection_name, embedding_function=embed_model_function, metadata={"hnsw:space": "cosine"})
    collection.add(
            documents= documents_list, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            metadatas= metadatas_list, # filter on these!
            ids=ids_list, # unique for each doc 
            )
    print("Documents indexed: ", len(documents_list))


def index_chroma_subschema(collection_name, persist_directory, schema_parts):
    embed_model = "all-MiniLM-L6-v2"
    client = chromadb.PersistentClient(path=persist_directory)
    documents_list, metadatas_list, ids_list = [],[],[]
    i = 0
    for query in schema_parts:
        doc = schema_parts[query]
        metadata = {"source": collection_name, "query_name": query}
        documents_list.append(doc)
        metadatas_list.append(metadata)
        ids_list.append("doc_id_"+str(i))
        i += 1

    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    try:
        client.delete_collection(name=collection_name)
        print("Deleted the existing collection :",collection_name)  
    except Exception as e:
        print(str(e))
        pass
    print("Creating collection:", collection_name)
    collection = client.create_collection(name=collection_name, embedding_function=embed_model_function, metadata={"hnsw:space": "cosine"})
    collection.add(
            documents= documents_list, # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            metadatas= metadatas_list, # filter on these!
            ids=ids_list, # unique for each doc 
            )
    print("Documents indexed: ", len(documents_list))


def query_chroma_subschema(collection_name, persist_directory, query_texts, recall_limit=10):
    embed_model = "all-MiniLM-L6-v2"
    # print("Querying chroma db collection:", collection_name)
    # Create collection. get_collection, get_or_create_collection, delete_collection also available!
    if embed_model == 'default':
        embed_model_function = embedding_functions.DefaultEmbeddingFunction()
    else:
        embed_model_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name, embedding_function=embed_model_function)
    output = collection.query(
        query_texts=query_texts,
        n_results=recall_limit,
    )
    results = output.get('metadatas')
    query_class_relevance = []
    for res in results:
        rel_classes = [r.get("query_name") for r in res]
        query_class_relevance.append(rel_classes)
    # print("Query count :", len(query_class_relevance))
    return query_class_relevance
