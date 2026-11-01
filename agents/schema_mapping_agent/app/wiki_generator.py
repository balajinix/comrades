# wiki_generator.py
# Contains all functions for creating the wiki structure and populating it with data

import mwclient
import json
import re
from typing import Dict, List, Any, Tuple, Set
from urllib.parse import quote, urlparse
from pathlib import Path
from collections import defaultdict

class TypeBasedMediaWikiGenerator:
    def __init__(self, wiki_url: str, username: str, password: str):
        """
        Initialize MediaWiki API client using mwclient
        
        Args:
            wiki_url: Base URL of MediaWiki instance (e.g., 'https://your-wiki.com')
            username: MediaWiki username
            password: MediaWiki password
        """
        self.wiki_url = wiki_url.rstrip('/')
        self.username = username
        self.password = password
        
        # Parse URL to get host and path
        parsed_url = urlparse(wiki_url)
        self.host = parsed_url.netloc
        self.path = parsed_url.path.rstrip('/') + '/'
        self.scheme = parsed_url.scheme
        
        # Initialize mwclient Site
        self.site = mwclient.Site(
            host=self.host,
            path=self.path,
            scheme=self.scheme
        )
        
        # Login
        self.login()
        
        # Track type mappings for bidirectional linking
        self.type_mappings = defaultdict(list)
        
        # Define supported products
        self.products = ['optimizer', 'apm', 'resilience', 'finopssuite']
        self.product_type_index = defaultdict(set) 

        # Load existing mappings from wiki pages
        self.load_existing_mappings()
    
    def login(self):
        """Login to MediaWiki using mwclient"""
        try:
            self.site.login(self.username, self.password)
            print("✓ Successfully logged in to MediaWiki")
        except mwclient.LoginError as e:
            raise Exception(f"Login failed: {e}")
        except Exception as e:
            raise Exception(f"Connection failed: {e}")
    
    def create_page(self, title: str, content: str, summary: str = "Created via API"):
        """
        Create or update a MediaWiki page using mwclient
        
        Args:
            title: Page title
            content: Wiki markup content
            summary: Edit summary
        """
        try:
            page = self.site.pages[title]
            page.edit(content, summary=summary)
            print(f"✓ Page '{title}' created/updated successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to create page '{title}': {e}")
            return False
    
    def page_exists(self, title: str) -> bool:
        """Check if a page exists"""
        try:
            page = self.site.pages[title]
            return page.exists
        except:
            return False
    
    def get_page_content(self, title: str) -> str:
        """Get content of an existing page"""
        try:
            page = self.site.pages[title]
            return page.text() if page.exists else ""
        except:
            return ""
    
    def parse_existing_mappings_from_page(self, page_content: str, type_key: str) -> List[Dict[str, Any]]:
        """
        Parse existing mappings from a wiki page content
        
        Args:
            page_content: The wiki page content
            type_key: The type key (e.g., 'optimizer_Action')
            
        Returns:
            List of existing mappings
        """
        mappings = []
        
        # Look for the mapping table in the content
        # Pattern to match table rows: |- followed by cells
        table_pattern = r'\{\| class="wikitable sortable".*?\|\}'
        table_match = re.search(table_pattern, page_content, re.DOTALL)
        
        if not table_match:
            return mappings
        
        table_content = table_match.group(0)
        
        # Extract table rows (lines starting with |-)
        row_pattern = r'\|-\s*\n((?:\|[^\n]*\n?)+)'
        rows = re.findall(row_pattern, table_content)
        
        for row in rows:
            # Extract cells from the row
            cells = re.findall(r'\|\s*([^|]+)', row)
            
            if len(cells) >= 4:
                # Clean up the cell content
                source_attr = re.sub(r'</?code>', '', cells[0].strip())
                # Extract target type from wiki link [[page|text]] or just the text
                target_type_match = re.search(r'\[\[[^\]]+\|([^\]]+)\]\]', cells[1])
                if target_type_match:
                    target_type = target_type_match.group(1).strip()
                else:
                    target_type = cells[1].strip()
                
                target_attr = re.sub(r'</?code>', '', cells[2].strip())
                justification = cells[3].strip()
                
                mappings.append({
                    'source_attribute': source_attr,
                    'target_type': target_type,
                    'target_attribute': target_attr,
                    'justification': justification
                })
        
        return mappings
    
    def load_existing_mappings(self):
        """Load existing mappings from all type pages in the wiki"""
        print("🔄 Loading existing mappings from wiki...")
        
        # Search for all pages under Products/ that match type page pattern
        try:
            # Get all pages in the Products namespace
            all_pages = list(self.site.allpages(prefix='Products/'))
            
            type_pages_found = 0
            for page in all_pages:
                page_title = page.name

                # Check if this is a type page
                parts = page_title.split('/')
                if len(parts) == 3 and parts[0] == 'Products':
                    product_part = parts[1].lower()
                    type_part = parts[2]

                    if type_part.startswith(f"{product_part}_"):
                        type_key = type_part

                        page_content = self.get_page_content(page_title)
                        if page_content:
                            existing_mappings = self.parse_existing_mappings_from_page(page_content, type_key)
                            if existing_mappings:
                                self.type_mappings[type_key] = existing_mappings

                                # Add to product-type index for listing later
                                self.product_type_index[product_part].add(type_key)

                                type_pages_found += 1
                                print(f"   ✓ Loaded {len(existing_mappings)} mappings for {type_key}")

            
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing mappings: {e}")
            print("   Continuing with empty mappings...")
    
    def create_main_products_page(self):
        """Create main products index page"""
        content = """= Products =

This is the main index for all product type mappings and schema documentation.

== Available Products ==

{| class="wikitable"
! Product !! Description
|-

| [[Products/Optimizer|Optimizer]] || Performance optimization platform
|-
| [[Products/APM|APM]] || Application and infrastructure monitoring 
|-
| [[Products/Resilience|Resilience]] || Security and vulnerability management 
|-
| [[Products/FinOpsSuite|FinOpsSuite]] || Cloud cost management platform
|}

[[Category:Type Mappings]]
"""
        
        self.create_page("Products", content, "Created main products index")
    
    def get_product_types_list(self, app_name: str) -> str:
        """Generate list of available types for a product"""
        types_list = ""
        for type_key in sorted(self.product_type_index.get(app_name, [])):
            type_name = '_'.join(type_key.split('_')[1:])
            page_title = self.get_type_page_title(app_name, type_name)
            types_list += f"* [[{page_title}|{app_name}_{type_name}]]\n"
        
        return types_list if types_list else "''No types available yet.''"

    def create_product_page(self, app_name: str):
        """Create main page for a product"""
        app_title = app_name.capitalize()
        
        content = f"""= {app_title} Product =

== Overview ==
Documentation for the '''[[Products/{app_title}|{app_title}]]''' product and its type mappings.

== Available Types ==

This product contains the following types:

{self.get_product_types_list(app_name)}

== Navigation ==
* [[Products]] - Back to products index

[[Category:Type Mappings]]
"""
        
        self.create_page(f"Products/{app_title}", content, f"Updated {app_title} product page")
    
    def parse_field_path(self, field_path: str) -> Tuple[str, str, str]:
        """
        Parse a field path to extract product, type name, and attribute
        
        Args:
            field_path: Field path like "optimizer_Action.actionType" or "resilience_PackageRecommendation.action"
            
        Returns:
            Tuple of (product, type_name,field)
        """
        # Split by first underscore to separate product from rest
        if '_' not in field_path:
            return 'unknown', field_path, ''
        
        parts = field_path.split('_', 1)
        product = parts[0].lower()
        
        # Split the rest by dot to separate type and field
        rest = parts[1]
        if '.' in rest:
            type_name, field = rest.split('.', 1)
        else:
            type_name = rest
            field = ''
        
        return product, type_name, field
    
    def get_type_page_title(self, product: str, type_name: str) -> str:
        """Get the page title for a type"""
        product_title = product.capitalize()
        return f"Products/{product_title}/{product}_{type_name}"
    
    def merge_mappings(self, existing_mappings: List[Dict[str, Any]], new_mappings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge existing and new mappings, avoiding duplicates and updating existing ones
        
        Args:
            existing_mappings: List of existing mappings
            new_mappings: List of new mappings to add
            
        Returns:
            Merged list of unique mappings
        """
        # Create a dictionary for quick lookup of existing mappings
        existing_map = {}
        for mapping in existing_mappings:
            key = (mapping['source_attribute'], mapping['target_type'], mapping['target_attribute'])
            existing_map[key] = mapping
        
        # Add or update with new mappings
        for mapping in new_mappings:
            key = (mapping['source_attribute'], mapping['target_type'], mapping['target_attribute'])
            existing_map[key] = mapping  # This will update if exists, or add if new
        
        return list(existing_map.values())
    
    def create_type_page(self, product: str, type_name: str, new_mappings: List[Dict[str, Any]]):
        """
        Create or update a type page with its mappings
        
        Args:
            product: Product name (e.g., 'optimizer')
            type_name: Type name (e.g., 'Action')
            new_mappings: List of new mapping information for this type
        """
        product_title = product.capitalize()
        page_title = self.get_type_page_title(product, type_name)
        type_full_name = f"{product}_{type_name}"
        
        # Get existing mappings for this type
        existing_mappings = self.type_mappings.get(type_full_name, [])
        
        # Merge existing and new mappings
        all_mappings = self.merge_mappings(existing_mappings, new_mappings)
        
        content = f"""= {type_full_name} Type =

== Overview ==
Documentation for the '''[[{page_title}|{type_full_name}]]''' type from the {product_title} product.

* '''Product:''' [[Products/{product_title}|{product_title}]]
* '''Type Name:''' {type_name}
* '''Full Name:''' {type_full_name}

== Type Mappings ==

This type has the following mappings:

"""
        
        if all_mappings:
            content += """{| class="wikitable sortable"
! Field !! Mapped To Type !! Target Field !! Justification
"""
            
            for mapping in all_mappings:
                source_attr = mapping['source_attribute']
                target_type = mapping['target_type']
                target_attr = mapping['target_attribute']
                justification = mapping['justification']
                target_product, target_type_name, _ = self.parse_field_path(target_type)
                target_page = self.get_type_page_title(target_product, target_type_name)
                
                content += f"""
|-
| <code>{source_attr}</code> || [[{target_page}|{target_type}]] || <code>{target_attr}</code> || {justification}"""
            
            content += "\n|}\n"
        else:
            content += "''No mappings available yet.''\n"
        
        content += f"""

== Navigation ==
* [[Products/{product_title}|{product_title}]] - Back to {product_title} product
* [[Products]] - Back to products index

[[Category:Type Mappings]]
"""
        
        # Update type mappings tracking with merged mappings
        if all_mappings:
            self.type_mappings[type_full_name] = all_mappings
            self.product_type_index[product].add(type_full_name) 

        
        # Create summary message
        added_count = len(all_mappings) - len(existing_mappings)
        if existing_mappings:
            summary = f"Updated {type_full_name} type page - added {added_count} new mappings, total: {len(all_mappings)}"
        else:
            summary = f"Created {type_full_name} type page with {len(all_mappings)} mappings"
        
        # Avoid unnecessary overwrite
        old_content = self.get_page_content(page_title)
        if old_content.strip() == content.strip():
            print(f"✓ No changes to {page_title}, skipping update.")
            return page_title

        self.create_page(page_title, content, summary)
        return page_title
    
    def process_mapping_json(self, mapping_json: Dict[str, Any]):
        """
        Process a mapping JSON and create type pages with bidirectional mappings
        
        Args:
            mapping_json: Mapping JSON with structure like the example provided
        """
        print("🔄 Processing mapping JSON and creating type pages...")
        
        # Ensure existing mappings are loaded (safe to call again)
        if not self.type_mappings:
            self.load_existing_mappings()

        # Extract source schema name (ignore for type creation)
        # source_schema = mapping_json.get('name', 'unknown')
        properties = mapping_json.get('properties', {})
        
        # Track all types that need to be created/updated
        types_to_update = defaultdict(list)
        
        # Process each property mapping
        for source_field_path, target_mappings in properties.items():
            # Parse source type
            source_product, source_type, source_attr = self.parse_field_path(source_field_path)
            source_type_key = f"{source_product}_{source_type}"
            
            # Process each target mapping
            for mapping in target_mappings:
                target_field_path = mapping.get('mapProperty', '')
                justification = mapping.get('justification', 'No justification provided')
                
                if not target_field_path:
                    continue
                
                # Parse target type
                target_product, target_type, target_attr = self.parse_field_path(target_field_path)
                target_type_key = f"{target_product}_{target_type}"
                
                # Add mapping for source type
                types_to_update[source_type_key].append({
                    'source_attribute': source_attr,
                    'target_type': target_field_path,
                    'target_attribute': target_attr,
                    'justification': justification
                })
                
                # Add reverse mapping for target type
                types_to_update[target_type_key].append({
                    'source_attribute': target_attr,
                    'target_type': source_field_path,
                    'target_attribute': source_attr,
                    'justification': justification
                })
        
        # Create/update type pages
        created_pages = []
        for type_key, new_mappings in types_to_update.items():
            product_name = type_key.split('_')[0]
            type_name = '_'.join(type_key.split('_')[1:])
            
            # Ensure product page exists
            if not self.page_exists(f"Products/{product_name.capitalize()}"):
                self.create_product_page(product_name)
            
            # Create/update type page
            page_title = self.create_type_page(product_name, type_name, new_mappings)
            created_pages.append(page_title)
        
        # Update all affected product pages to refresh their type lists
        affected_products = set()
        for type_key in types_to_update.keys():
            product_name = type_key.split('_')[0]
            affected_products.add(product_name)
        
        # Refresh state before regenerating product pages
        self.load_existing_mappings()

        for product in affected_products:
            if product in self.products:
                self.create_product_page(product)
        
        print(f"✅ Created/updated {len(created_pages)} type pages:")
        for page in created_pages:
            print(f"   - {page}")
            
        return created_pages
    
    def initialize_products(self):
        """Create the main products structure"""
        print("🏗️  Creating products structure...")
        
        # Create main products page
        self.create_main_products_page()
        
        # Create product pages for known products
        for product in self.products:
            if not self.page_exists(f"Products/{product.capitalize()}"):
                self.create_product_page(product)
        
        print("✅ Products structure created successfully!")

    def check_type_pages_status(self):
        """Check status of type pages"""
        print("🔍 Checking type pages status...")
        
        # Check main structure
        main_pages = [
            "Products",
            "Products/Optimizer",
            "Products/APM", 
            "Products/Resilience",
            "Products/FinOpsSuite"
        ]
        
        existing_pages = []
        missing_pages = []
        
        for page in main_pages:
            if self.page_exists(page):
                existing_pages.append(page)
            else:
                missing_pages.append(page)
        
        print(f"✅ Existing main pages ({len(existing_pages)}):")
        for page in existing_pages:
            print(f"   - {page}")
        
        if missing_pages:
            print(f"\n❌ Missing main pages ({len(missing_pages)}):")
            for page in missing_pages:
                print(f"   - {page}")
        
        # Check type pages
        type_pages = list(self.type_mappings.keys())
        print(f"\n📋 Tracked type pages ({len(type_pages)}):")
        for type_key in type_pages:
            product_name = type_key.split('_')[0]
            type_name = '_'.join(type_key.split('_')[1:])
            page_title = self.get_type_page_title(product_name, type_name)
            status = "✅" if self.page_exists(page_title) else "❌"
            mapping_count = len(self.type_mappings[type_key])
            print(f"   {status} {page_title} ({mapping_count} mappings)")
        
        return {
            'existing_main': existing_pages,
            'missing_main': missing_pages,
            'type_pages': type_pages
        }