#!/usr/bin/env python3
"""
Script to clean wake_tech.json file by:
1. Removing citation markers [1][2][3] from string values
2. Removing reference links outside JSON structure
3. Validating JSON structure
"""

import json
import re
import sys

def clean_citation_markers(text):
    """Remove citation markers like [1][2][3] from text."""
    if not isinstance(text, str):
        return text
    # Remove citation markers like [1], [2][3], etc.
    return re.sub(r'\[(\d+)\](\[\d+\])*', '', text).strip()

def clean_json_recursively(obj):
    """Recursively clean citation markers from all string values in JSON."""
    if isinstance(obj, dict):
        return {key: clean_json_recursively(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_recursively(item) for item in obj]
    elif isinstance(obj, str):
        return clean_citation_markers(obj)
    else:
        return obj

def clean_wake_tech_json(input_file, output_file):
    """Clean the wake_tech.json file."""
    try:
        # Read the file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove reference links at the end (lines after closing brace)
        # Find the last closing brace
        last_brace = content.rfind('}')
        if last_brace != -1:
            # Keep only up to and including the closing brace
            json_content = content[:last_brace + 1]
        else:
            json_content = content
        
        # Parse JSON
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON structure at line {e.lineno}, column {e.colno}")
            print(f"Error message: {e.msg}")
            return False
        
        # Clean citation markers
        cleaned_data = clean_json_recursively(data)
        
        # Write cleaned JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        # Validate the output
        with open(output_file, 'r', encoding='utf-8') as f:
            json.load(f)  # This will raise an error if invalid
        
        print(f"✓ Successfully cleaned {input_file}")
        print(f"✓ Output saved to {output_file}")
        print(f"✓ JSON is valid")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    input_file = "data/wake_tech.json"
    output_file = "data/wake_tech.json"  # Overwrite the original
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    success = clean_wake_tech_json(input_file, output_file)
    sys.exit(0 if success else 1)

