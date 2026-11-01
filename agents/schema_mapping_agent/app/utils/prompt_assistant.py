"""
prompt_assistant.py
"""

def get_default_few_shot_prompt():
    """
    Returns a two-example "Few shot" prompt with GraphQL queries.
    """
    few_shot_prompt = (
        """question: "Get APM entities."\n"""
        """graphql query: query GetAPMEntities {\n"""
        """  entities {\n"""
        """    id\n"""
        """    label\n"""
        """    entityType\n"""
        """  }\n"""
        """}\n\n"""
        """question: "Get actions from optimizer."\n"""
        """graphql query: query GetMarketActions {\n"""
        """  actions {\n"""
        """    id\n"""
        """    name\n"""
        """    actionType\n"""
        """    relatedEntityTypes\n"""
        """    environmentType\n"""
        """    riskCategory\n"""
        """  }\n"""
        """}\n\n"""
        """question: "Get applications labeled Robot with their services for the last 5 hours."\n"""
        """graphql query: query Applications($f: apm_ApplicationFilter!, $n: Int, $a: String) {\n"""
        """  applications: apm_applications(first: $n, filter: $f, after: $a) {\n"""
        """    edges {\n"""
        """      node {\n"""
        """        id\n"""
        """        label\n"""
        """        optimizersApplication {\n"""
        """          actions {\n"""
        """            displayName\n"""
        """            risk {\n"""
        """              severity\n"""
        """              description\n"""
        """              importance\n"""
        """            }\n"""
        """          }\n"""
        """        }\n"""
        """        services {\n"""
        """          id\n"""
        """          label\n"""
        """          technologies\n"""
        """          types\n"""
        """          endpoints {\n"""
        """            id\n"""
        """            label\n"""
        """            technologies\n"""
        """            type\n"""
        """            syntheticType\n"""
        """          }\n"""
        """        }\n"""
        """      }\n"""
        """    }\n"""
        """  }\n"""
        """}\n"""
        """variables: {\n"""
        """  "f": {\n"""
        """    "label": { "eq": "Robot" },\n"""
        """    "timeFrame": {\n"""
        """      "ge": "2025-03-18T15:10:57Z",\n"""
        """      "le": "2025-03-18T20:10:57Z"\n"""
        """    }\n"""
        """  },\n"""
        """  "n": 3,\n"""
        """  "a": ""\n"""
        """}\n"""
    )
    return few_shot_prompt

def generate_few_shots_prompt(nl_query, schema_details, dataset, model_name):
    """
    Generates a few-shot prompt for GraphQL query generation.
    - For granite‑3 series models (chat interface), use chat markers with a system message,
      and include the few-shot examples.
    - For Llama models, use a specific prompt style.
    - Otherwise, use a default prompt.
    """
    # Load the default few-shot examples
    few_shot_prompt = get_default_few_shot_prompt()

    schema_name = schema_details.get("db_id", "")
    schema_text = schema_details.get("schema_content", "")
    schema_part = f"You are given the following graphql schema for {schema_name}.\n{schema_text}\n"

    if "granite-3" in model_name.lower():
        # Chat-style prompt for granite‑3 series models with a system instruction.
        prompt = (
            "<|start_of_role|>system<|end_of_role|>You are an expert in generating graphql queries. "
            "Given the graphql schema, the few-shot examples, and the utterance, generate the graphql query. "
            "If the query requires variables, generate those as well in a JSON format after the query under a 'variables:' section.<|end_of_text|>\n"
            "<|start_of_role|>user<|end_of_role|> Schema: " + schema_text + "\n\n"
            "Few shot examples:\n" + few_shot_prompt + "\n\n"
            "Utterance: " + nl_query + "<|end_of_text|>\n"
        )
    elif "lama" in model_name.lower():
        prompt = (
            "<s>[INST] Your task is to generate a graphql query for the given question.\n"
            f"<<SYS>> {schema_part}\n{few_shot_prompt} <</SYS>> \nGenerate a correct graphql query for the following question. "
            f"If variables are required, include them in a JSON format under a 'variables:' section.\n"
            f"{nl_query} [/INST]"
        )
    else:
        prompt = (
            "Your task is to generate a graphql query for the given question.\n"
            f"{schema_part}\n{few_shot_prompt}\n"
            "Generate a correct graphql query for the following question. "
            "If variables are required, include them in a JSON format under a 'variables:' section.\n"
            f"question: \"{nl_query}\"\ngraphql query:"
        )
    return prompt


def generate_zero_shot_prompt(nl_query, schema_details, dataset, model_name):
    """
    generate_zero_shot_prompt with support for variables
    """
    schema_name = schema_details.get("db_id", "")
    schema_text = schema_details.get("schema_content", "")
    schema_part = f"You are given the following graphql schema for {schema_name}.\n{schema_text}\n"

    if "lama" in model_name or model_name in ["machinists/CodeLlama-7b-Instruct-SQL", "machinists/Mistral-7B-Instruct-SQL"]:
        prompt = "<s>[INST] Your task is to generate graphql query for the given question.\n"
        prompt += f"<<SYS>> {schema_part} <</SYS>> \n Generate a correct graphql query for the following question. "
        prompt += f"If variables are required, include them in JSON format under a 'variables:' section.\n"
        prompt += f"{nl_query} [/INST]"
    else:
        prompt = f"{schema_part}\nGenerate a correct graphql query for the following question. "
        prompt += f"If variables are required, include them in JSON format under a 'variables:' section.\n"
        prompt += f"question: \"{nl_query}\"\ngraphql query:"

    return prompt


def generate_prompt(nl_query, schema_details, prompt_technique, dataset, model_name):
    """
    generate_prompt
    """
    if prompt_technique == "Zero Shot":
        return generate_zero_shot_prompt(nl_query, schema_details, dataset, model_name)
    elif prompt_technique == "Few Shot":
        return generate_few_shots_prompt(nl_query, schema_details, dataset, model_name)
    elif prompt_technique == "Chain of Thought":
        prompt = generate_few_shots_prompt(nl_query, schema_details, dataset, model_name)
        prompt += "\nThink step by step about what variables might be needed.\n"
        return prompt

    prompt = (f"question: \"{nl_query}\"\n"
              f"graphql query:")
    return prompt
