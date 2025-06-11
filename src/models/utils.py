import re

def expand_stack(stack):
    # Expand shorthand
    # For example, LC4 -> LCLCLCLC
    expanded, block = '', ''
    parts = re.findall(r'\d+|[^\d]+', stack)
    for part in parts:
        if part.isdigit(): 
            expanded += block * int(part)
            block = ''
        else: block = part

    return expanded + block
