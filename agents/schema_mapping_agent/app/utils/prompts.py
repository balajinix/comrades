from utils.map import *

OPERATION_MAPPING_TEMPLATE = """
You are given two applications with various operations and their descriptions. Your task is to match the operations in application 1 with the operations in application 2 and provide reasoning for the same.
If application 1 has two operations, operation1 and operation2 which respectively map to two operations, operation3 and operation4 in application 2, your output should be formatted in JSON format as follows:

{
    "operation1": {
        "match": "operation3",
        "reasoning": "..."
    },
    "operation2": 
        "match": "operation4",
        "reasoning": "..."
    }
}

If there are no matching operations, return an empty JSON.
Do not generate any non-existent operation in the response.
If the operation name contains the "/" character, then the full operation name mmust be retained.
Limit your response to the top 10 matches.

Here are the details of application 1:

Operations:
{operations1}

Here are the details of application 2:

Operations:
{operations2}

Top 10 Matching Operations along with reasoning (in JSON format):
"""

OPERATION_MAPPING_TEMPLATE_WITH_HISTORY = """
You are given two applications with various operations and their descriptions. Your task is to match the operations in application 1 with the operations in application 2 and provide reasoning for the same.
If application 1 has two operations, operation1 and operation2 which respectively map to two operations, operation3 and operation4 in application 2, your output should be formatted in JSON format as follows:

{
    "operation1": {
        "match": "operation3",
        "reasoning": "..."
    },
    "operation2": 
        "match": "operation4",
        "reasoning": "..."
    }
}

In addition, you are given the following history of previous operation matches:

{history}

Given the above history, you should try to avoid mapping multiple source operations to the same target operation as far as possible.

If there are no matching operations, return an empty JSON.
Do not generate any non-existent operation in the response.
If the operation name contains the "/" character, then the full operation name must be retained.

Here are the details of application 1:

Name: {name1}
Operations:
{operations1}

Here are the details of application 2:

Name: {name2}
Operations:
{operations2}

Matching Operations along with reasoning (in JSON format):
"""


def construct_operation_mapping_prompt(app1: str, data1: dict, ops1: list, app2: str, data2: dict, dst_graph: dict):
    name1 = app1
    operations1 = ""
    relevant_ops = []
    for e in ops1:
        operations1 += "- " + e + ": " + data1[e]["description"] + "\n"
        relevant_ops += get_relevant_ops(data1, e, app2, dst_graph)

    name2 = app2
    operations2 = ""
    relevant_ops = list(set(relevant_ops))
    # print(relevant_ops)
    for e in relevant_ops:
        operations2 += "- " + e + ": " + data2[e]["description"] + "\n"
        # if "fields" in data2[e]:
        #     operations2 += ". It has the following fields: "
        #     for f in data2[e]["fields"]:
        #         operations2 += " " + f + " (" + data2[e]["fields"][f]["description"] + ")"

    prompt = OPERATION_MAPPING_TEMPLATE.replace("{operations1}",operations1) \
                .replace("{operations2}",operations2) 
                
    return prompt

def construct_operation_mapping_prompt_with_history(app1: str, data1: dict, ops1: list, app2: str, data2: dict, history: dict, dst_graph: dict):
    name1 = app1
    operations1 = ""
    relevant_ops = []
    for e in ops1:
        operations1 += "- " + e + ": " + data1[e]["description"] + "\n"
        relevant_ops += get_relevant_ops(data1, e, app2, dst_graph)

    name2 = app2
    operations2 = ""
    relevant_ops = list(set(relevant_ops))
    # print(relevant_ops)
    for e in relevant_ops:
        operations2 += "- " + e + ": " + data2[e]["description"] + "\n"
        # if "fields" in data2[e]:
        #     operations2 += ". It has the following fields: "
        #     for f in data2[e]["fields"]:
        #         operations2 += " " + f + " (" + data2[e]["fields"][f]["description"] + ")"

    history_str = json.dumps(history, indent=4)

    prompt = OPERATION_MAPPING_TEMPLATE_WITH_HISTORY.replace("{name1}",name1) \
                .replace("{name2}",name2) \
                .replace("{operations1}",operations1) \
                .replace("{operations2}",operations2) \
                .replace("{history}", history_str)
                
    return prompt

TYPE_MAPPING_TEMPLATE = """
You are given two applications with some GraphQL schemas with various GraphQL types and their descriptions. Your task is to match the types in application 1 with the types in application 2 and provide reasoning for the same.
If application 1 has two operations, schema1/type1 and schema1/type2 which respectively map to two types, schema2/type3 and schema2/type4 in application 2, your output should be formatted in JSON format as follows:

{
    "schema1/type1": {
        "match": "schema2/type3",
        "reasoning": "..."
    },
    "schema1/type2": 
        "match": "schema2/type4",
        "reasoning": "..."
    }
}

If there are no matching types, return an empty JSON.
Do not generate any non-existent type in the response.
Limit your response to the top 10 matches.

Here are the details of application 1:

Schemas and Types:
{types1}

Here are the details of application 2:

Schemas and Types:
{types2}

Top 10 Matching Operations along with reasoning (in JSON format):
"""

def construct_type_mapping_prompt(app1: str, data1: dict, dbs1: list, app2: str, data2: dict, chroma_directory: str):
    name1 = app1
    types1 = ""
    relevant_types = []
    for db_name in dbs1:
        for type_name in data1[db_name]:
            field_desc = "It contains the following fields:"
            if "fields" in data1[db_name][type_name]:
                for x in data1[db_name][type_name]["fields"]:
                    field_desc += " " + split_camel_case(x) + " (" + data1[db_name][type_name]["fields"][x]["description"]+")"
            types1 += f"- {db_name}/{type_name}: " + data1[db_name][type_name]["description"] + field_desc + "\n"
            relevant_types += get_relevant_types(data1, db_name, type_name, app2+"_graphql", chroma_directory)

    name2 = app2
    types2 = ""
    relevant_types = list(set(relevant_types))
    # print(relevant_types, data2.keys())
    for t in relevant_types:
        db_name, type_name = t.split("###")
        field_desc = "It contains the following fields:"
        if "fields" in data2[db_name][type_name]:
            for x in data2[db_name][type_name]["fields"]:
                field_desc += " " + split_camel_case(x) + " (" + data2[db_name][type_name]["fields"][x]["description"]+")"
        types2 += f"- {db_name}/{type_name}: " + data2[db_name][type_name]["description"] + field_desc + "\n"

    prompt = TYPE_MAPPING_TEMPLATE.replace("{name1}",name1) \
                .replace("{name2}",name2) \
                .replace("{types1}",types1) \
                .replace("{types2}",types2) 
                
    # print(prompt)
    return prompt

OPERATION_MAPPING_REFINEMENT_TEMPLATE = """
You are given two applications and their operations. Some matching operations have been found with some reasoning. Your job is to criticize the matches and discover flaws in the reasoning, if any. You will provide a new set of matches in JSON format and give an updated reasoning in the JSON incorporating criticism of the previous reasoning by adding a "criticism" field in the provided JSON.

If the original JSON was

{
    "operation1": {
        "match": "operation3",
        "reasoning": "..."
    },
    "operation2": 
        "match": "operation4",
        "reasoning": "..."
    }
}

then your response should be

{
    "operation1": {
        "match": "operation3",
        "reasoning": "...",
        "criticism": "..."
    },
    "operation2": {
        "match": "operation4",
        "reasoning": "...",
        "criticism": "..."
    }
}

Take care to only alter the Reason part and replace it with a Criticism. Do not modify the JSON including the operation names in any other way.

If the matches are correct, you may repeat the original matches.

Here are the details of application 1:

Operations:
{operations1}

Here are the details of application 2:

Operations:
{operations2}

Matching Operations along with reasoning (in JSON format):
{op_matches}

New matching operations along with updated reasoning and criticism (in JSON format):
"""


def construct_operation_mapping_refinement_prompt(app1: str, data1: dict, ops1: list, app2: str, data2: dict, op_matches: str, dst_graph: dict):
    name1 = app1
    operations1 = ""
    relevant_ops = []
    for e in ops1:
        operations1 += "- " + e + ": " + data1[e]["description"] + "\n"
        relevant_ops += get_relevant_ops(data1, e, app2, dst_graph)

    name2 = app2
    operations2 = ""
    relevant_ops = list(set(relevant_ops))
    # print(relevant_ops)
    for e in relevant_ops:
        operations2 += "- " + e + ": " + data2[e]["description"] + "\n"
        # if "fields" in data2[e]:
        #     operations2 += ". It has the following fields: "
        #     for f in data2[e]["fields"]:
        #         operations2 += " " + f + " (" + data2[e]["fields"][f]["description"] + ")"

    prompt = OPERATION_MAPPING_REFINEMENT_TEMPLATE.replace("{operations1}",operations1) \
                .replace("{operations2}",operations2) \
                .replace("{op_matches}",op_matches)
                
    return prompt


TYPE_MAPPING_REFINEMENT_TEMPLATE = """
You are given two applications which have some GraphQL schemas and their GraphQL types. Some matching types between the applications have been found with some reasoning. Your job is to criticize the matches and discover flaws in the reasoning, if any. You will provide a new set of matches in JSON format and give an updated reasoning in the JSON incorporating criticism of the previous reasoning by adding a "criticism" field in the provided JSON.

The criticism should be as follows:
If a mapping is reasonbly good, say it is acceptable.
If a mapping is not good, always try to suggest an alternative mapping.
If no alternative mapping can be found, then say that no mapping is possible.


If the original JSON was in the format

{
    "schema1/type1": {
        "match": "schema2/type3",
        "reasoning": "..."
    },
    "schema1/type2": 
        "match": "schema2/type4",
        "reasoning": "..."
    }
    ...
}

then your response should be in the format

{
    "schema1/type1": {
        "match": "schema2/type3",
        "reasoning": "...",
        "criticism": "..."
    },
    "schema1/type2": 
        "match": "schema2/type4",
        "reasoning": "...",
        "criticism": "..."
    }
    ...
}

Replace schema1, schema2, type 1 and type2 with the actual schema and type names in the provided JSON.
If the matches are correct, you may repeat the original matches.

Here are the details of application 1:

Schemas and Types:
{types1}

Here are the details of application 2:

Schemas and Types:
{types2}

Matching types along with reasoning (in JSON format):
{type_matches}

New matching types along with criticism (in JSON format):
"""

def construct_type_mapping_refinement_prompt(app1: str, data1: dict, dbs1: list, app2: str, data2: dict, type_matches: str, chroma_directory: str):
    name1 = app1
    types1 = ""
    relevant_types = []
    for db_name in dbs1:
        for type_name in data1[db_name]:
            field_desc = "It contains the following fields:"
            if "fields" in data1[db_name][type_name]:
                for x in data1[db_name][type_name]["fields"]:
                    field_desc += " " + split_camel_case(x) + " (" + data1[db_name][type_name]["fields"][x]["description"]+")"
            types1 += f"- {db_name}/{type_name}: " + data1[db_name][type_name]["description"] + field_desc + "\n"
            relevant_types += get_relevant_types(data1, db_name, type_name, app2+"_graphql", chroma_directory)

    name2 = app2
    types2 = ""
    relevant_types = list(set(relevant_types))
    # print(relevant_types, data2.keys())
    for t in relevant_types:
        db_name, type_name = t.split("###")
        field_desc = "It contains the following fields:"
        if "fields" in data2[db_name][type_name]:
            for x in data2[db_name][type_name]["fields"]:
                field_desc += " " + split_camel_case(x) + " (" + data2[db_name][type_name]["fields"][x]["description"]+")"
        types2 += f"- {db_name}/{type_name}: " + data2[db_name][type_name]["description"] + field_desc + "\n"

    prompt = TYPE_MAPPING_REFINEMENT_TEMPLATE.replace("{types1}",types1) \
                .replace("{types2}",types2) \
                .replace("{type_matches}",type_matches)
                
    # print(prompt)
    return prompt

TYPE_MAPPING_CORRECTION_TEMPLATE = """
You are given type mappings for two GraphQL schemas along with expert criticism of the same.

- If the criticism says that the mapping is acceptable, retain the mapping. 
- If the criticism says that the mapping is not good, drop the mapping by replacing the "match" field with an empty string. 
- If the criticism suggests a different mapping, update the mapping accordingly.

Your response should contain the corrected type mappings in the same JSON format as you have received. Please retain the "reasoning" and "criticism" fields.

Matching types along with criticism (in JSON format with keys as <schema_name>/<type_name>):
{type_matches}

Corrected matching types based on the above criticism (in the same JSON format):
"""

def construct_type_mapping_correction_prompt(type_matches: str):
    
    prompt = TYPE_MAPPING_CORRECTION_TEMPLATE.replace("{type_matches}",type_matches)
                
    return prompt

FIELD_MAPPING_TEMPLATE = """
You are given two APIs, API 1 and API 2 with various output fields and their descriptions (it is possible that one or both of them has no fields). Your task is to match the fields in API 1, if any, with suitable fields in API 2, if any, and provide valid reasoning behind the matches.
If API 1 has two fields, field1 and field2 which respectively map to two fields field3 and field4 in API 2, your output should be formatted in JSON format as follows:

{
    "field1": {
        "match": "field3",
        "reasoning": "..."
    },
    "field2": 
        "match": "field4",
        "reasoning": "..."
    }
}

If any one of API 1 and API 2 has no fields, always return an empty JSON and do not generate any non-existent field of API 1 or API 2 in the response.
If there are no suitable matches, return an empty JSON.
Always copy the full field name in the response and do not leave out any part.
Limit your response to the top 10 matches.

Here are the details of API 1:

Name: {name1}
Description: {desc1}
Fields:
{fields1}

Here are the details of API 2:

Name: {name2}
Description: {desc2}
Fields:
{fields2}

Top 10 Matching fields along with reasoning (in JSON format):
"""

def construct_field_mapping_prompt(data1: dict, op1: str, data2: dict, op2: str):
    name1 = op1
    desc1 = data1[op1]["description"]
    fields1 = ""
    if "fields" in data1[op1]:
        for f in data1[op1]["fields"]:
            fields1 += op1 + "/" + f + ": " + data1[op1]["fields"][f]["description"] + "\n"
            if "fields" in data1[op1]["fields"][f]:
                for f1 in data1[op1]["fields"][f]["fields"]:
                    fields1 += "  - " + op1 + "/" + f + "/" + f1 + ": " + data1[op1]["fields"][f]["fields"][f1]["description"] + "\n"
    name2 = op2
    desc2 = data2[op2]["description"]
    fields2 = ""
    if "fields" in data2[op2]:
        for f in data2[op2]["fields"]:
            fields2 += op2 + "/" + f + ": " + data2[op2]["fields"][f]["description"] + "\n"
            if "fields" in data2[op2]["fields"][f]:
                for f1 in data2[op2]["fields"][f]["fields"]:
                    fields2 += "  - " + op2 + "/" + f + "/" + f1 + ": " + data2[op2]["fields"][f]["fields"][f1]["description"] + "\n"
    
    prompt = FIELD_MAPPING_TEMPLATE.replace("{name1}",name1) \
                .replace("{name2}",name2) \
                .replace("{desc1}",desc1) \
                .replace("{desc2}",desc2) \
                .replace("{fields1}",fields1) \
                .replace("{fields2}",fields2)
                
    return prompt


def construct_toolbench_field_mapping_prompt(data1: dict, op1: str, data2: dict, op2: str):
    name1 = op1
    desc1 = data1[op1]["api_description"]
    fields1 = ""
    if "fields" in data1[op1]:
        for f in data1[op1]["fields"]:
            fields1 += op1 + "/" + f + ": " + data1[op1]["fields"][f]["api_description"] + "\n"
            if "fields" in data1[op1]["fields"][f]:
                for f1 in data1[op1]["fields"][f]["fields"]:
                    fields1 += "  - " + op1 + "/" + f + "/" + f1 + ": " + data1[op1]["fields"][f]["fields"][f1]["api_description"] + "\n"
    name2 = op2
    desc2 = data2[op2]["api_description"]
    fields2 = ""
    if "fields" in data2[op2]:
        for f in data2[op2]["fields"]:
            fields2 += op2 + "/" + f + ": " + data2[op2]["fields"][f]["api_description"] + "\n"
            if "fields" in data2[op2]["fields"][f]:
                for f1 in data2[op2]["fields"][f]["fields"]:
                    fields2 += "  - " + op2 + "/" + f + "/" + f1 + ": " + data2[op2]["fields"][f]["fields"][f1]["api_description"] + "\n"

    prompt = FIELD_MAPPING_TEMPLATE.replace("{name1}",name1) \
                .replace("{name2}",name2) \
                .replace("{desc1}",desc1) \
                .replace("{desc2}",desc2) \
                .replace("{fields1}",fields1) \
                .replace("{fields2}",fields2)

    return prompt

FIELD_MAPPING_REFINEMENT_TEMPLATE = """
You are given two APIs, API 1 and API 2 with various output fields and their descriptions (it is possible that one or both of them has no fields). Some matching fields have been found with some reasoning. Your job is to criticize the matches and discover flaws in the reasoning, if any. You will provide a new set of matches and give an updated reasoning incorporating criticism of the previous reasoning by adding a "criticism" field in the provided JSON.

The criticism should be as follows:
If a mapping is reasonbly good, say it is acceptable.
If a mapping is not good, always try to suggest an alternative mapping.
If no alternative mapping can be found, then say that no mapping is possible.

If the original JSON was

{
    "field1": {
        "match": "field3",
        "reasoning": "..."
    },
    "field2": 
        "match": "field4",
        "reasoning": "..."
    }
}

then your response should be

{
    "field1": {
        "match": "field3",
        "reasoning": "...",
        "criticism": "..."
    },
    "field2": 
        "match": "field4",
        "reasoning": "...",
        "criticism": "..."
    }
}

If the matches are correct, you may repeat the original matches.

Here are the details of API 1:

Name: {name1}
Description: {desc1}
Fields:
{fields1}

Here are the details of API 2:

Name: {name2}
Description: {desc2}
Fields:
{fields2}

Matching fields along with reasoning (in JSON format):
{field_matches}

New matching fields along with updated reasoning and criticism (in JSON format):
"""

def construct_field_mapping_refinement_prompt(data1: dict, op1: str, data2: dict, op2: str, field_matches: dict):
    name1 = op1
    desc1 = data1[op1]["description"]
    fields1 = ""
    if "fields" in data1[op1]:
        for f in data1[op1]["fields"]:
            fields1 += f + ": " + data1[op1]["fields"][f]["description"] + "\n"
            if "fields" in data1[op1]["fields"][f]:
                for f1 in data1[op1]["fields"][f]["fields"]:
                    fields1 += "  - " + f1 + ": " + data1[op1]["fields"][f]["fields"][f1]["description"] + "\n"
    name2 = op2
    desc2 = data2[op2]["description"]
    fields2 = ""
    if "fields" in data2[op2]:
        for f in data2[op2]["fields"]:
            fields2 += f + ": " + data2[op2]["fields"][f]["description"] + "\n"
            if "fields" in data2[op2]["fields"][f]:
                for f1 in data2[op2]["fields"][f]["fields"]:
                    fields2 += "  - " + f1 + ": " + data2[op2]["fields"][f]["fields"][f1]["description"] + "\n"
    
    prompt = FIELD_MAPPING_REFINEMENT_TEMPLATE.replace("{name1}",name1) \
                .replace("{name2}",name2) \
                .replace("{desc1}",desc1) \
                .replace("{desc2}",desc2) \
                .replace("{fields1}",fields1) \
                .replace("{fields2}",fields2) \
                .replace("{field_matches}",json.dumps(field_matches, indent=4))
                
    return prompt

FIELD_MAPPING_CORRECTION_TEMPLATE = """
You are given field mappings for two APIs along with expert criticism of the same.

- If the criticism says that the mapping is acceptable, retain the mapping. 
- If the criticism says that the mapping is not good, drop the mapping by replacing the "match" field with an empty string. 
- If the criticism suggests a different mapping, update the mapping accordingly.

Your response should contain the corrected field mappings in the same JSON format as you have received. Please retain the "reasoning" and "criticism" fields.

Matching fields along with criticism (in JSON format):
{field_matches}

Corrected matching fields based on the above criticism (in JSON format):
"""

def construct_field_mapping_correction_prompt(field_matches: str):
    
    prompt = FIELD_MAPPING_CORRECTION_TEMPLATE.replace("{field_matches}",field_matches)
                
    return prompt


ANNOTATION_TEMPLATE = """
You are given an API endpoint along with its fields and corresponding description in JSON format.
Your job is to insert a "title" key in the fields of the API which provides a set of at least two phrases, each of which gives a succinct summary of the field and its purpose.

Examples are provided below:

EXAMPLE 1

INPUT: 

{
    "get-instances": {
        "description": "This endpoint returns a list containing a summary of each of your Aura instances. To find out more about a specific instance, retrieve the details using the GET /instances/{instanceId} endpoint.\n\nYou can filter instances in a particular tenant using GET /instances?tenantId={tenantId}. If the tenant query parameter is not specified, this endpoints lists all instances a user has access to across all tenants.",
        "path": [
            "/instances",
            "get"
        ],
        "fields": {
            "data": {
                "description": "",
                "fields": {
                    "id": {
                        "description": "The id of the instance",
                    },
                    "name": {
                        "description": "The name of the instance"
                    },
                    "tenant_id": {
                        "description": "The project to which the instance belongs to"
                    },
                    "cloud_provider": {
                        "description": "The cloud provider of the instance"
                    },
                    "created_at": {
                        "description": "The timestamp when the instance was created"
                    }
                }
            }
        }
    }
}

OUTPUT:

{
    "get-instances": {
        "description": "This endpoint returns a list containing a summary of each of your Aura instances. To find out more about a specific instance, retrieve the details using the GET /instances/{instanceId} endpoint.\n\nYou can filter instances in a particular tenant using GET /instances?tenantId={tenantId}. If the tenant query parameter is not specified, this endpoints lists all instances a user has access to across all tenants.",
        "path": [
            "/instances",
            "get"
        ],
        "fields": {
            "data": {
                "description": "",
                "fields": {
                    "id": {
                        "description": "The id of the instance",
                        "title": ["Instance ID", "Aura instance ID"]
                    },
                    "name": {
                        "description": "The name of the instance",
                        "title": ["Instance name", "Aura instance name"]
                    },
                    "tenant_id": {
                        "description": "The project to which the instance belongs to",
                        "title": ["Project ID", "Aura project"]
                    },
                    "cloud_provider": {
                        "description": "The cloud provider of the instance",
                        "title": ["Cloud provider", "Cloud service provider"]
                    },
                    "created_at": {
                        "description": "The timestamp when the instance was created",
                        "title": ["Instance creation timestamp", "Instance creation time", "Time of instance creation"]
                    }
                }
            }
        }
    }
}

EXAMPLE 2

INPUT:

{   
    "getSystemStatus": {
        "description": "This resource returns information about the MongoDB application along with API key meta data.",
        "path": [
            "/api/atlas/v2",
            "get"
        ],
        "fields": {
            "apiKey": {
                "description": "Details contained in one API key.",
                "fields": {}
            },
            "appName": {
                "description": "Human-readable label that identifies the service from which you requested this response.",
                "fields": {}
            },
            "build": {
                "description": "Unique 40-hexadecimal digit hash that identifies the latest git commit merged for this application.",
                "fields": {}
            },
            "links": {
                "description": "List of one or more Uniform Resource Locators (URLs) that point to API sub-resources, related API resources, or both. RFC 5988 outlines these relationships.",
                "fields": {
                    "href": {
                        "description": "Uniform Resource Locator (URL) that points another API resource to which this response has some relationship. This URL often begins with `https://cloud.mongodb.com/api/atlas`."
                    },
                    "rel": {
                        "description": "Uniform Resource Locator (URL) that defines the semantic relationship between this resource and another API resource. This URL often begins with `https://cloud.mongodb.com/api/atlas`."
                    }
                }
            },
            "throttling": {
                "description": "Flag that indicates whether someone enabled throttling on this service.",
                "fields": {}
            }
        }
    }
}

OUTPUT:

{
    "getSystemStatus": {
        "description": "This resource returns information about the MongoDB application along with API key meta data.",
        "path": [
            "/api/atlas/v2",
            "get"
        ],
        "fields": {
            "apiKey": {
                "description": "Details contained in one API key.",
                "title": ["API key", "MongoDB API key"],
                "fields": {}
            },
            "appName": {
                "description": "Human-readable label that identifies the service from which you requested this response.",
                "title": ["App name", "Service name"],
                "fields": {}
            },
            "build": {
                "description": "Unique 40-hexadecimal digit hash that identifies the latest git commit merged for this application.",
                "title": ["Build number", "Git commit number", "Git commit ID", "Git commit hash"],
                "fields": {}
            },
            "links": {
                "description": "List of one or more Uniform Resource Locators (URLs) that point to API sub-resources, related API resources, or both. RFC 5988 outlines these relationships.",
                "title": ["Links to API resources", "API resource links"]
                "fields": {
                    "href": {
                        "description": "Uniform Resource Locator (URL) that points another API resource to which this response has some relationship. This URL often begins with `https://cloud.mongodb.com/api/atlas`.",
                        "title": ["Link", "URL"]
                    },
                    "rel": {
                        "description": "Uniform Resource Locator (URL) that defines the semantic relationship between this resource and another API resource. This URL often begins with `https://cloud.mongodb.com/api/atlas`.",
                        "title": ["Relationship", "Link info"]
                    }
                }
            },
            "throttling": {
                "description": "Flag that indicates whether someone enabled throttling on this service.",
                "title": ["Throttling enabled flag", "Throttling indicator"],
                "fields": {}
            }
        }
    }
}

It's your turn now!

INPUT:

{input}

OUTPUT:
"""

def construct_annotation_prompt(data: dict):
    s = json.dumps(data, indent=4)
    
    prompt = ANNOTATION_TEMPLATE.replace("{input}",s)
                
    return prompt


CATEGORY_CLASSIFICATION_TEMPLATE = """
You are given a list of categories as follows:
{categories}

Given a query, your task is to identify all possible categories to which the query might be relevant. Your output should be a comma separated list. 
E.g. if a query is relevant to two categories, category1 and category2, then your output should be as follows:

```
category1, category2
```

Query:
{query}

Categories:
"""

def construct_category_classification_prompt(categories: list, query: str):
    s = ", ".join(categories)
    prompt = CATEGORY_CLASSIFICATION_TEMPLATE.replace("{categories}",s).replace("{query}",query)                
    return prompt

TOOLBENCH_API_MAPPING_TEMPLATE = """
You are given various APIs and their descriptions. Your task is to find appropriate matches between the APIs provide reasoning for the same.
If you are given 5 APIs, (API1, API2, API3, API4 and API5), and API1 and API2 respectively map to API3 and API4, your output should be formatted in JSON format as follows:

{
    "API1 is mapped to API3": "Reason: ...",
    "API2 is mapped to API4": "Reason: ..."
}

If there are no matching APIs, return an empty JSON.
Do not generate any non-existent API in the response.
Do not map the same API to itself and always use the full name of the API i.e. always retain the ### in the API name.

Here are the API details:

{apis}

Matching APIs along with reasoning (in JSON format):
"""

TOOLBENCH_API_MAPPING_REFINEMENT_TEMPLATE = """
You are given various APIs and their descriptions.  Some matching APIs have been found with some reasoning. Your job is to criticize the matches and discover flaws in the reasoning, if any. You will provide a new set of matches and give an updated reasoning incorporating criticism of the previous reasoning. If the matches are correct, you may repeat the original matches.

Here are the API details:

{apis}

Matching APIs along with reasoning (in JSON format):
{api_matches}

New matching APIs along with reasoning (in JSON format):
"""

def construct_toolbench_api_mapping_prompt(apis: str):

    prompt = TOOLBENCH_API_MAPPING_TEMPLATE.replace("{apis}",apis)
                
    return prompt

def construct_toolbench_api_mapping_refinement_prompt(apis: str, api_matches: str):

    prompt = TOOLBENCH_API_MAPPING_REFINEMENT_TEMPLATE.replace("{apis}",apis).replace("{api_matches}",api_matches)
                
    return prompt

GRAPHQL_SCHEMA_GENERATION_TEMPLATE = """
You are an API and GraphQL expert. You are given the response structures of the following APIs in JSON format:

{apis}

Generate a unified GraphQL schema for the above APIs with no type repetitions. Your output should only contain the GraphQL schema enclosed in "```graphql" and "```".
Do not generate incomplete type definitions. If you need to generate a type for an object which has no fields, assume a default type of JSON and do not generate a separate type.

GraphQL schema:
"""

def construct_graphql_schemagen_prompt(apis: dict):

    api_str = json.dumps(apis, indent=4)
    prompt = GRAPHQL_SCHEMA_GENERATION_TEMPLATE.replace("{apis}",api_str)
                
    return prompt
