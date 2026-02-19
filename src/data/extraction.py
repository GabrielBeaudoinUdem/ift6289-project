"""
Extraction script to format the relevant information from fr-RL into JSON 
format.

"""

import json
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
import re


RAW_DATA_PATH = Path(__file__).resolve().parent / "fr-LN_V3_2" / "ls-fr-V3.2"
JSON_PATH = Path(__file__).resolve().parent / "data.json"


def extract_text(html:str) -> str:
    """
    Extracts the text from an input html code. 
    
    :param html: A string representing html code.
    :type html: str
    :return: The text corresponding to the html code, without the html tags.
    :rtype: str
    """
    if pd.isna(html):
        return None
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def extract_actants(xml_string:str) -> list[str] | None:
    """
    Extracts the semantic actants present in a definition from the XML 
    structure describing these actants.

        Example of output: `$1 is un individu`

    :param xml_string: A XML structure describing the semantic actants.
    :type xml_string: str
    :return: The list of string  describing each actants in the inputed XML 
        code, without the XML tags.
    :rtype: list[str]
    """
    if pd.isna(xml_string):
        return None
    
    root = ET.fromstring(xml_string)
    sentences = []

    for tc in root.findall(".//actantpos/tc"):
        text = "".join(tc.itertext())
        
        # clean spacing like you did for definitions
        text = (
            text.strip()
                .replace("\n", " ")
        )
        
        sentences.append(" ".join(text.split()))
    return sentences if sentences else None


def combine_examples(df: pd.DataFrame) -> list[dict]:
    """
    Combines the examples' data spread into multiple columns of the DataFrame
    into a single column.

    Takes the information present in the columns `example_text`, `occurrences`
    and `position`, and merge them into a singular list of dicitonnary. The 
    data from the original columns are lists, where the indices linked to each 
    other. For example, the information at index 0 of `example_text` is related
    to the information at index 0 of `occurrence` and `position`.

    :param df: DataFrame containing the columns `example_text`, `occurrences`
    and `position`.
    :type df: pd.DataFrame
    :return: A list of dictionnaries containing all the relevant data for one
    example
    :rtype: list[dict]
    """
    # If the cell is NaN, replace with empty list
    texts = df["example_text"] if isinstance(df["example_text"], list) else []
    occurrences = df["occurrence"] if isinstance(df["occurrence"], list) else []
    positions = df["position"] if isinstance(df["position"], list) else []

    n = max(len(texts), len(occurrences), len(positions))
    examples = []
    for i in range(n):
        examples.append({
            "text": texts[i] if i < len(texts) else None,
            "occurrence": occurrences[i] if i < len(occurrences) else None,
            "position": positions[i] if i < len(positions) else None
        })
    return examples


def extract_nodes():
    """
    Extract the data from the file `01-lsnodes.csv` and saves it
    in a DataFrame.

    :return: The DataFrame corresponding with the relevant information
    """


def extract_data():
    df = pd.DataFrame()

    # We keep overwritting ls_path and ls_df to save RAM
    

    ## 01-lsnodes.csv
    # We keep node_id, entry_id, lexnum, status, and confidence %
    # status and % are to make further cleaning if necessary. Might delete
    ls_path = RAW_DATA_PATH / "01-lsnodes.csv"
    ls_df = pd.read_csv(ls_path, sep="\t")
    tmp_df = ls_df[["id", "entry", "lexnum", "status", "%"]].rename(
        columns={
            "id": "node_id",
            "entry": "entry_id"
        }
    )
    df = pd.concat([df, tmp_df], ignore_index=True)


    ## 02-lsentries.csv
    # We keep addtoname and name (concatenate them)
    ls_path = RAW_DATA_PATH / "02-lsentries.csv"
    ls_df = pd.read_csv(ls_path, sep="\t")
    ls_df["fullname"] = ls_df["addtoname"].fillna("") + ls_df["name"].fillna("")
    ls_df = ls_df[["id", "fullname"]]

    df = df.merge(
        ls_df,
        how="left",
        left_on="entry_id",
        right_on="id"
    )
    df = df.drop(columns=["id"]).rename(columns={"fullname": "name"})

    

    ## 05-lsgramcharac-model.xml
    ls_path = RAW_DATA_PATH / "05-lsgramcharac-model.xml"
    tree = ET.parse(ls_path)
    root = tree.getroot()

    # Dictionnary of the characteristic id as keys and the names as values
    charact_type2 = { 
        char.get("id"):char.get("name") 
        for char in root.findall(".//characteristic")
    }
    

    ## 06-lsgramcharac-rel.csv
    # We use node_id to keep usagenote, usagenotevars (for testing, might 
    # delete) and POS
    # We assume that POS is only POS values... TODO: Check this
    ls_path = RAW_DATA_PATH / "06-lsgramcharac-rel.csv"
    ls_df = pd.read_csv(ls_path, sep="\t")
    ls_df = ls_df[["node", "usagenote", "usagenotevars", "POS"]]

    df = df.merge(
        ls_df,
        how="left",
        left_on="node_id",
        right_on="node"
    )
    
    df = df.drop(columns=["node"])

    # Using `charact_type2`, we replace the POS id with it's name
    df["POS"] = df["POS"].map(charact_type2)


    ## 13-lsdef.csv
    # Extract the 176 available definitions
    ls_path = RAW_DATA_PATH / "13-lsdef.csv"
    ls_df = pd.read_csv(ls_path, sep="\t")

    ls_df["definition"] = ls_df["def_HTML"].apply(extract_text)
    ls_df["actants"] = ls_df["def_XML"].apply(extract_actants)
    # Remove extra spaces and leading "Definition"
    ls_df["definition"] = (
        ls_df["definition"]
            .str.replace(r"^Définition\s+", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)

            # Fix spacing after apostrophes (both ’ and ')
            .str.replace(r"([’'])\s+", r"\1", regex=True)

            # Fix spacing after opening parenthesis
            .str.replace(r"\(\s+", "(", regex=True)

            # Fix spacing before closing parenthesis
            .str.replace(r"\s+\)", ")", regex=True)

            .str.strip()
    )
    ls_df = ls_df[["node", "definition", "actants"]]
    
    df = df.merge(
        ls_df,
        how="left",
        left_on="node_id",
        right_on="node"
    )
    df = df.drop(columns=["node"])


    ## 14-lslf-model.xml
    ls_path = RAW_DATA_PATH / "14-lslf-model.xml"
    tree = ET.parse(ls_path)
    root = tree.getroot()

    # Dictionnary of the lf id as keys and [name and linktype] as values
    lexical_function_names = { 
        char.get("id"): [ char.get("name"), char.get("linktype") ]
        for char in root.findall(".//lexicalfunction")
    }
    

    ## 15-lslf-rel.csv
    ls_path = RAW_DATA_PATH / "15-lslf-rel.csv"
    ls_df = pd.read_csv(ls_path, sep="\t")
    ls_df = ls_df[["source", "lf", "target"]]

    node_to_name = dict(zip(df["node_id"], df["name"]))
    ls_df["target"] = ls_df["target"].map(node_to_name) # Replace the target id with the actual name
    ls_df["lf"] = ls_df["lf"].map(lexical_function_names) # Replace the lf id with it's name

    # Keep only needed columns
    ls_df["lf_name"] = ls_df["lf"].apply(lambda x: x[0])  # Just the LF name
    ls_df["lf_linktype"] = ls_df["lf"].apply(lambda x: x[1])  # Just link type

    ls_df["lf_dict"] = ls_df.apply(
        lambda row: {
            "lexical_function": row["lf_name"],
            "linktype": row["lf_linktype"],
            "target": row["target"]
        },
        axis=1
    )
    ls_df_agg = ls_df.groupby("source")["lf_dict"].apply(list).reset_index()

    # Merge into df
    df = df.merge(
        ls_df_agg,
        how="left",
        left_on="node_id",
        right_on="source"
    ).drop(columns=["source"])


    ## 17-lsex.csv
    ls_path = RAW_DATA_PATH / "17-lsex.csv"
    ls_examples = pd.read_csv(ls_path, sep="\t")
    ls_examples = ls_examples[["id", "status", "content"]].copy()
    ls_examples["example_text"] = ls_examples["content"].apply(extract_text) # Remove html tags
    ls_examples.drop(columns=["content"], inplace=True)

    ## 18-lsex-rel.csv
    ls_path = RAW_DATA_PATH / "18-lsex-rel.csv"
    ls_df = pd.read_csv(ls_path, sep="\t")

    ls_df = ls_df[ls_df["position"] <= 2]

    ls_df = ls_df.merge(
        ls_examples,
        how="left",
        left_on="example",
        right_on="id"
    )

    # Aggregate per node
    ls_df_agg = ls_df.groupby("node").agg({
        "example_text": list,
        "occurrence": list,
        "position": list
    }).reset_index()

    
    ls_df_agg = ls_df_agg[["node", "example_text", "occurrence", "position"]]
    df = df.merge(
        ls_df_agg,
        how="left",
        left_on="node_id",
        right_on="node"
    )
    
    example_cols = ["example_text", "occurrence", "position"]
    

    df = df[["node_id", "entry_id", "name", "lexnum", "POS", "usagenote", "usagenotevars", "definition", 
             "actants", "lf_dict", "example_text", "occurrence", "position", "status", "%"]]
    df["examples"] = df.copy().apply(combine_examples, axis=1)
    df = df.drop(columns=example_cols)

    metadata = {
        "description": "French lexical-semantic dataset",
        "url": "https://www.ortolang.fr/market/lexicons/lexical-system-fr/v3.2",
        "dataset_version": "0.0.1",
        "fields": {
            "node_id": "Unique identifier for the lexical node",
            "entry_id": "Identifier of the dictionary entry this node belongs to",
            "name": "Full name of the lexical entry",
            "lexnum": "Lexical number (I.1, II.2, etc.)",
            "POS": "Part-of-speech of the lexical entry",
            "usagenote": "Optional usage note",
            "usagenotevars": "Optional usage variants note",
            "definition": "Definition of the lexical entry",
            "actants": "List of actants extracted from the definition",
            "lf_dict": "List of lexical functions: each item is a dict with keys 'lexical_function', 'linktype', 'target'",
            "examples": "List of example usages: each item is a dict with 'text', 'occurrence', 'position'",
            "status": "Internal status code",
            "%": "Confidence percentage"
        }
    }

    # Replace all NaN with None
    df = df.where(pd.notna(df), None)
    data_list = df.to_dict(orient="records")
    json_output = {
        "metadata": metadata,
        "data": data_list
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    
if __name__ == "__main__":
    extract_data()