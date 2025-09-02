"""Tree building and manipulation utilities for API hierarchy."""


def build_tree(hierarchy_dict):
    """Build a tree structure from flat hierarchy data.
    
    Args:
        hierarchy_dict: Dictionary with 'qualified_name', 'name', 'type', 
                       'depth', and 'path_parts' keys
                       
    Returns:
        Nested dictionary representing the API tree
    """
    tree = {"name": "fenic", "type": "root", "children": {}}

    # Process each element
    for i, qual_name in enumerate(hierarchy_dict['qualified_name']):
        name = hierarchy_dict['name'][i]
        elem_type = hierarchy_dict['type'][i]
        depth = hierarchy_dict['depth'][i]
        path_parts = hierarchy_dict['path_parts'][i]

        # Navigate to the correct position in the tree
        current = tree
        for _j, part in enumerate(path_parts[:-1]):  # All but the last part
            if part not in current['children']:
                current['children'][part] = {
                    "name": part,
                    "type": "unknown",  # Will be updated when we process that element
                    "children": {}
                }
            current = current['children'][part]

        # Add the final element
        if len(path_parts) > 0:
            final_part = path_parts[-1]
            current['children'][final_part] = {
                "name": name,
                "type": elem_type,
                "qualified_name": qual_name,
                "depth": depth,
                "children": {}
            }

    return tree


def tree_to_string(node, indent=0, max_depth=3):
    """Convert tree structure to string with indentation.
    
    Args:
        node: Tree node to convert
        indent: Current indentation level
        max_depth: Maximum depth to display
        
    Returns:
        String representation of the tree
    """
    if indent > max_depth:
        return ""

    result = ""

    if indent > 0:  # Skip root
        result += "  " * (indent-1) + f"├─ [{node['type']}] {node['name']}\n"

    # Sort children by type then name for better readability
    children = sorted(
        node.get('children', {}).values(),
        key=lambda x: (
            0 if x['type'] == 'module' else
            1 if x['type'] == 'class' else
            2 if x['type'] == 'function' else
            3 if x['type'] == 'method' else
            4,
            x['name']
        )
    )

    for child in children[:10]:  # Limit to first 10 children
        result += tree_to_string(child, indent + 1, max_depth)

    if len(children) > 10:
        result += "  " * indent + f"... and {len(children) - 10} more\n"

    return result