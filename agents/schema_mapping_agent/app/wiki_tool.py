# wiki_tool.py 
# Main file containing type-based wiki configurations for populating the wiki using mapping results

"""
Configuration script for type-based schema-to-wiki mapping
"""

import json
import os
from pathlib import Path
from wiki_generator import TypeBasedMediaWikiGenerator
from dotenv import load_dotenv
from typing import Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# MediaWiki Configuration
WIKI_CONFIG = {
    'wiki_url':os.getenv("WIKI_URL"),
    'path': '/wiki/',
    'username':"anon", 
    'password':"YOUR_WIKI_PASSWORD"
}

# Mapping files directory (where your generated mapping JSON files are stored)
MAPPING_FILES_DIR = 'mappings_output/'

# ============================================================================
# CORE CONFIGURATION FUNCTIONS
# ============================================================================

def initialize_wiki_structure():
    """
    Initialize only the wiki products structure (run this once)
    Creates: Products -> Optimizer, APM, Resilience, FinOpsSuite
    """
    print("🏗️  Initializing products structure...")
    
    wiki_generator = TypeBasedMediaWikiGenerator(
        wiki_url=WIKI_CONFIG['wiki_url'],
        username=WIKI_CONFIG['username'],
        password=WIKI_CONFIG['password']
    )
    
    wiki_generator.initialize_products()
    
    print("✅ Products structure initialized!")
    print("📄 Created pages:")
    print("   - Products (main index)")
    print("   - Products/Optimizer, Products/APM, Products/Resilience, Products/FinOpsSuite")

def process_mapping_dict(mapping_json: Dict):
    """
    Process a mapping JSON dictionary directly
    
    Args:
        mapping_json: Mapping JSON dictionary
    """
    print("🚀 Processing mapping dictionary...")
    
    wiki_generator = TypeBasedMediaWikiGenerator(
        wiki_url=WIKI_CONFIG['wiki_url'],
        username=WIKI_CONFIG['username'],
        password=WIKI_CONFIG['password']
    )
    
    created_pages = wiki_generator.process_mapping_json(mapping_json)
    print("Success!")
    result = f"Processed mapping dictionary successfully!"
    result += f"\nCreated/updated {len(created_pages)} pages:"
    for page in created_pages:
        result += f"   - {page}"
    return result

def process_mapping_file(mapping_file_path: str):
    """
    Process a single mapping JSON file
    
    Args:
        mapping_file_path: Path to the mapping JSON file
    """
    print(f"🚀 Processing mapping file: {mapping_file_path}")
    
    if not os.path.exists(mapping_file_path):
        print(f"❌ Mapping file not found: {mapping_file_path}")
        return
    
    with open(mapping_file_path) as f:
        mapping_json = json.load(f)
    
    wiki_generator = TypeBasedMediaWikiGenerator(
        wiki_url=WIKI_CONFIG['wiki_url'],
        username=WIKI_CONFIG['username'],
        password=WIKI_CONFIG['password']
    )
    
    created_pages = wiki_generator.process_mapping_json(mapping_json)
    
    print(f"✅ Processed mapping file successfully!")
    print(f"📊 Created/updated {len(created_pages)} type pages")
    print(f"📄 Created/updated pages:")
    for page in created_pages:
        print(f"   - {wiki_generator.wiki_url}/wiki/{page}")

def process_mapping_directory(directory_path: str = None, pattern: str = "*.json"):
    """
    Process all mapping files from a directory
    
    Args:
        directory_path: Directory containing mapping files (defaults to MAPPING_FILES_DIR)
        pattern: File pattern to match (defaults to "*.json")
    """
    if directory_path is None:
        directory_path = MAPPING_FILES_DIR
    
    print(f"🔍 Looking for mapping files in: {directory_path}")
    print(f"📋 Pattern: {pattern}")
    
    mapping_files = list(Path(directory_path).glob(pattern))
    
    if not mapping_files:
        print(f"❌ No mapping files found in {directory_path} with pattern {pattern}")
        return
    
    print(f"📁 Found {len(mapping_files)} mapping files:")
    for file in mapping_files:
        print(f"   - {file}")
    
    wiki_generator = TypeBasedMediaWikiGenerator(
        wiki_url=WIKI_CONFIG['wiki_url'],
        username=WIKI_CONFIG['username'],
        password=WIKI_CONFIG['password']
    )
    
    total_pages_created = 0
    
    for mapping_file in mapping_files:
        print(f"\n{'='*60}")
        print(f"Processing: {mapping_file}")
        
        try:
            with open(mapping_file) as f:
                mapping_json = json.load(f)
            
            created_pages = wiki_generator.process_mapping_json(mapping_json)
            total_pages_created += len(created_pages)
            
        except Exception as e:
            print(f"❌ Error processing {mapping_file}: {e}")
            continue
    
    print(f"\n✅ Completed processing {len(mapping_files)} mapping files!")
    print(f"📊 Total type pages created/updated: {total_pages_created}")

def check_wiki_status():
    """
    Check the status of products structure and type pages
    """
    print("🔍 Checking products and type pages status...")
    
    wiki_generator = TypeBasedMediaWikiGenerator(
        wiki_url=WIKI_CONFIG['wiki_url'],
        username=WIKI_CONFIG['username'],
        password=WIKI_CONFIG['password']
    )
    
    return wiki_generator.check_type_pages_status()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage - uncomment the function you want to run
    
    ### 1. First time setup (run this once) - initialize structure + process all files in directory
    # print("STEP 1: Initializing Products Structure")
    # print("="*60)
    # initialize_wiki_structure()
    
    # print("\n" + "="*60)
    # print("STEP 2: Processing All Mapping Files")
    # print("="*60)
    # process_mapping_directory()
    
    # print("\n" + "="*60)
    # print("STEP 3: Checking Status")
    # print("="*60)
    # check_wiki_status()
    
    # print("\n✅ Complete setup finished!")
    
    ### 2. Process a single mapping file
    process_mapping_file('~/mydata/resilience_applications-apm_applications.json')
    #~/mydata/resilience_application-optimizer_entity.json')
    
    
    ### 3. Process all files from directory
    # process_mapping_directory()
    
    ### 4. Process a mapping dictionary directly
    # mapping_dict = {
    #     "name": "resilience_applications",
    #     "properties": {
    #         "resilience_AssociationID.id": [
    #         {
    #             "mapTable": "apm_applications",
    #             "mapProperty": "apm_Entity__Mock.id",
    #             "justification": "Both fields represent IDs of associated components, making them conceptually related."
    #         }
    #         ],
    #         "resilience_SourceRepo.associations.build_artifacts.id": [
    #         {
    #             "mapTable": "apm_applications",
    #             "mapProperty": "apm_Entity__Mock.id",
    #             "justification": "Both represent IDs of associated components, with the resilience field specifically within the context of source repository associations."
    #         }
    #         ]
    #     }
    # }

    # process_mapping_dict(mapping_dict)
    
    ### 5. Check current status
    check_wiki_status()