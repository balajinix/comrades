import json
import re
from collections import defaultdict
import os

def parse_graphql_schema(schema):
    type_definitions = defaultdict(dict)
    enum_definitions = []
    scalar_definitions = []
    current_type = None
    current_desc = None
    for line in schema.split("\n"):
        line = line.strip()
        enum_match = re.match(r'enum\s+(\w+)\s*{', line)
        scalar_match = re.match(r'scalar\s+(\w+)', line)
        if enum_match:
            current_enum = enum_match.group(1)
            enum_definitions.append(current_enum)
        elif scalar_match:
            current_scalar = scalar_match.group(1)
            scalar_definitions.append(current_scalar)
    for line in schema.split("\n"):
        line = line.strip()
        if line.startswith("type "):
            match = re.match(r'type\s+(\w+)(\s+implements\s+\w+)?\s*{', line)
            if match:
                current_type = match.group(1)
                type_definitions[current_type] = {}
        elif current_type and line.endswith("}"):
            current_type = None
        elif current_type and line:
            desc_match = re.match(r'".*"', line)
            if desc_match:
                current_desc = desc_match.group(0)
            field_match = re.match(r'(\w+): ([\w\[\]!]+)', line)
            if field_match:
                field_name, field_type = field_match.groups()
                type_definitions[current_type][field_name] = {"description": "", "data_type": field_type}
                if current_desc:
                    type_definitions[current_type][field_name]["description"] = current_desc
                    current_desc = None

    
    return type_definitions, enum_definitions, scalar_definitions

def build_fields(type_definitions, response_type):
    fields = {}
    if response_type in type_definitions:
        for field, field_info in type_definitions[response_type].items():
            field_description = field_info["description"]
            field_type = field_info["data_type"]
            fields[field] = {
                "description": field_description,
                "data_type": field_type,
                "fields": build_fields(type_definitions, field_type.replace("!", "").replace("[", "").replace("]", ""))
            }
    return fields

def traverse(fields, path, desc, field_type, properties, enum_definitions, scalar_definitions):
    if fields == {} and field_type != "":
        orig_field_type = field_type
        field_type = field_type.replace("!", "").replace("[", "").replace("]", "")
        schema_component = ".".join(path.split(".")[:-1])
        
        if field_type in enum_definitions:
            if "[" in orig_field_type:
                properties.append({"name": path, "description": desc, "type": "[Enum]", "schema_component": schema_component, "metadata": {}})
            else:    
                properties.append({"name": path, "description": desc, "type": "Enum", "schema_component": schema_component, "metadata": {}})
        elif field_type in scalar_definitions:
            if "[" in orig_field_type:
                properties.append({"name": path, "description": desc, "type": "[Scalar]", "schema_component": schema_component, "metadata": {}})
            else:    
                properties.append({"name": path, "description": desc, "type": "Scalar", "schema_component": schema_component, "metadata": {}})
        else:
            properties.append({"name": path, "description": desc, "type": field_type, "schema_component": schema_component, "metadata": {}})
    else:
        for f in fields:
            # print(fields[f])
            desc = desc.replace("\"","")
            fields[f]["description"] = fields[f]["description"].replace("\"","")
            if path != "" and desc != "" and fields[f]["description"] != "":
                traverse(fields[f]["fields"], path+"."+f, desc+" -> " + fields[f]["description"], fields[f]["data_type"], properties, enum_definitions, scalar_definitions)
            elif path != "" and desc != "":
                traverse(fields[f]["fields"], path+"."+f, desc, fields[f]["data_type"], properties, enum_definitions, scalar_definitions)
            elif path != "":
                traverse(fields[f]["fields"], path+"."+f, fields[f]["description"], fields[f]["data_type"], properties, enum_definitions, scalar_definitions)
            else:
                traverse(fields[f]["fields"], f, fields[f]["description"], fields[f]["data_type"], properties, enum_definitions, scalar_definitions)


def process(schema, name):
    schema_dict = {}
    schema_dict["id"] = name
    schema_dict["name"] = name
    schema_dict["description"] = ""
    schema_dict["metadata"] = {}

    schema_dict["properties"] = []

    type_definitions, enum_definitions, scalar_definitions = parse_graphql_schema(schema)
    # print(d)
    for type_name in type_definitions:
        if type_name != "Query":
            fields = build_fields(type_definitions, type_name)
            path = type_name
            desc = ""
            field_type = ""
            traverse(fields, path, desc, field_type, schema_dict["properties"], enum_definitions, scalar_definitions)

            # print(type_props)
            
    return schema_dict


if __name__ == "__main__":
    gateway_files = [f for f in os.listdir("data/graphql/gateway") if f.endswith(".graphql")]
    for f in gateway_files:
        path = "graphql/schema/gateway/"+f
        db = f.replace(".graphql","")
        with open(path) as f:
            schema = f.read()
            with open(f"data/output/{db}.json","w") as out:
                json.dump(process(schema, db), out, indent=4)



