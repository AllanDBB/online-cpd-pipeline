"""Script para limpiar clearpage en results.tex"""

with open('results.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output = []
i = 0
removed_count = 0

while i < len(lines):
    line = lines[i]
    
    # Si encontramos \\clearpage
    if line.strip() == '\\clearpage':
        # Ver si está entre dos tablas (no antes de subsection)
        # Buscar hacia adelante
        next_significant = None
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].strip() and not lines[j].strip() == '':
                next_significant = lines[j].strip()
                break
        
        # Si lo siguiente es \begin{table}, eliminar el clearpage
        if next_significant and next_significant.startswith('\\begin{table}'):
            removed_count += 1
            print(f"Removiendo clearpage en línea {i+1}, siguiente: {next_significant[:30]}")
            i += 1
            continue
    
    output.append(line)
    i += 1

# Escribir
with open('results.tex', 'w', encoding='utf-8') as f:
    f.writelines(output)

print(f"\n✅ Eliminados {removed_count} clearpage")
