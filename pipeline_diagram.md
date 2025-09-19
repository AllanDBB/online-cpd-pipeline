# Main 2 Synthetic Pipeline

## Overview
- Inicializar configuraciones globales (semilla, combinaciones ruido/fuerza/tipo, lista de algoritmos, rejillas de hiperparametros).
- Generar las ocho combinaciones de series sinteticas Allan_synthetic.
- Ejecutar cada algoritmo sobre cada dataset aplicando busqueda en rejilla y evaluacion.
- Registrar metricas clave y los mejores hiperparametros en el CSV final.

```mermaid
flowchart TD
    A[Config global\n- Semilla\n- Combinaciones ruido/fuerza/cambio\n- Lista algoritmos\n- Grids hiperparametros] --> B[Generacion sintetica\n8 combos Allan_synthetic]
    B --> C[Loop datasets]
    C --> D[Loop algoritmos]
    subgraph Algoritmos
        D1[changepoint_online\nFocus/Gaussian/NPFocus/MDFocus]
        D2[OCPDet\ncumsum/ewma/two sample tests/neural networks]
        D3[SSM-State Space Models\ncanary]
        D4[TAGI-LSTM/SSM\ncanary]
        D5[SKF Kalman Filter SSM\ncanary]
        D6[Bayesian Online CPD\ncpfinder]
        D7[ChangeFinder SDAR\nchangefinder]
        D8[RuLSIF\nRoerich]
    end
    D --> E[Grid Search\nValidacion y metricas]
    E --> F[Seleccionar mejor set de parametros]
    F --> G[Guardar resultados\nCSV con mejores hiperparametros y metricas clave]
```
