# convert_to_ipynb.py
import nbformat as nbf

# Wczytaj plik
with open('numpy_course.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Parse cells
nb = nbf.v4.new_notebook()
cells = []

current_cell = []
cell_type = 'code'

for line in content.split('\n'):
    if line.startswith('# %% [markdown]'):
        if current_cell:
            cells.append(nbf.v4.new_code_cell('\n'.join(current_cell)))
            current_cell = []
        cell_type = 'markdown'
    elif line.startswith('# %%'):
        if current_cell:
            if cell_type == 'markdown':
                cells.append(nbf.v4.new_markdown_cell('\n'.join(current_cell)))
            else:
                cells.append(nbf.v4.new_code_cell('\n'.join(current_cell)))
            current_cell = []
        cell_type = 'code'
    else:
        if cell_type == 'markdown' and line.startswith('"""'):
            continue
        current_cell.append(line)

nb['cells'] = cells

# Save
with open('numpy_course.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Converted! Open numpy_course.ipynb")