"""
Script para limpiar espacios en blanco excesivos en results.tex
- Elimina \clearpage entre tablas de escenarios
- Cambia [ht] a [H] para mejor posicionamiento
"""

# Leer archivo
with open('results.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Reemplazos
import re

# Eliminar \clearpage entre tablas (excepto antes de \subsection o \clearpage antes de Benchmark 2)
lines = content.split('\n')
cleaned_lines = []
skip_next_clearpage = False

for i, line in enumerate(lines):
    # Si encontramos \end{table} seguido de líneas vacías y luego \clearpage
    if line.strip() == '\\clearpage':
        # Verificar si viene después de una tabla y antes de otra tabla
        if i > 0 and i < len(lines) - 1:
            # Buscar hacia atrás si hay \end{table}
            has_table_before = False
            for j in range(max(0, i-3), i):
                if '\\end{table}' in lines[j]:
                    has_table_before = True
                    break
            
            # Buscar hacia adelante si hay \begin{table} o \subsection
            has_table_after = False
            has_subsection_after = False
            for j in range(i+1, min(len(lines), i+5)):
                if '\\begin{table}' in lines[j]:
                    has_table_after = True
                    break
                if '\\subsection' in lines[j]:
                    has_subsection_after = True
                    break
            
            # Si hay tabla antes y tabla después, eliminar el \clearpage
            # Pero mantenerlo si viene antes de \subsection (nueva sección)
            if has_table_before and has_table_after and not has_subsection_after:
                print(f"Eliminando \\clearpage en línea {i+1}")
                continue  # Skip this line
    
    cleaned_lines.append(line)

# Reunir contenido
cleaned_content = '\n'.join(cleaned_lines)

# Cambiar [ht] a [H] en tablas de escenarios (para mejor posicionamiento)
cleaned_content = re.sub(r'\\begin\{table\}\[ht\]', r'\\begin{table}[H]', cleaned_content)

# Guardar
with open('results.tex', 'w', encoding='utf-8') as f:
    f.write(cleaned_content)

print("\n✅ Limpieza completada!")
print(f"   - {len(lines) - len(cleaned_lines)} líneas \\clearpage eliminadas")
print("   - Todas las tablas cambiadas a posicionamiento [H]")
