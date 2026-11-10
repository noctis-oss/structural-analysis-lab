# Structural Analysis Lab — supplementary materials

Solved Noctis Effect notebooks and check scripts for the book
**Structural Analysis Lab: Theory and Practice using Noctis Effect** (English edition 1.0.2) and its
Spanish edition **Lab de Análisis Estructural: Teoría y Práctica con Noctis Effect** (edición 1.0.1).

*Versión en español más abajo.*

## Solved notebooks

One notebook per lab with every worked example and every practice problem that has a model.

| Folder | Edition |
|---|---|
| `notebooks-en/` | English: `lab-1-solved.noefbook` … `lab-9-solved.noefbook`, `appendices-b-and-d-solved.noefbook` |
| `notebooks-es/` | Spanish: `lab-1-resuelto.noefbook` … `lab-9-resuelto.noefbook`, `apendices-b-y-d-resueltos.noefbook` |

**Import into Noctis Effect (iPad):** download the `.noefbook` file (open it on GitHub › *Download raw file*),
then tap it in the Files app, or in Noctis Effect tap **Import** under **Notebooks** on **Home**. It appears as a notebook with one
study per example. Built and read with Noctis Effect 1.0.0 (18).

Solve each example by hand first; open the notebook to compare, not to skip the work.

## Check scripts (`verify/`)

The author's independent check of the hand results: plain Python + NumPy, written from the equations of each
lab (equilibrium, virtual work, the force method and a small direct-stiffness solver). Each `*.py` prints the
values the book uses; the matching `*.out` is the output for this edition.

```bash
pip install -r requirements.txt
cd verify && python3 lab7.py
```

These scripts are the author's own check, not an independent peer review.

## Review

Before release the book went through the Noctis Scientific Review Protocol: blind independent solutions of
every example and practice problem, a pedagogical review and a technical/Spanish-fidelity review, followed by
corrections. It is internal quality control, not certification.

## License

MIT (scripts and notebooks). The book text is © 2026 Yonatan Ali Rodríguez Arias and is not part of this repository.

---

# Lab de Análisis Estructural — materiales complementarios

Cuadernos resueltos de Noctis Effect y scripts de comprobación del libro **Lab de Análisis Estructural:
Teoría y Práctica con Noctis Effect** (edición 1.0.1) y de su edición en inglés (1.0.2).

## Cuadernos resueltos

Un cuaderno por lab con cada ejemplo resuelto y cada problema de práctica que tiene modelo:
`notebooks-es/` (español) y `notebooks-en/` (inglés).

**Importar en Noctis Effect (iPad):** descarga el archivo `.noefbook` (ábrelo en GitHub › *Download raw file*),
tócalo en la app Archivos o en Noctis Effect toca **Importar** en **Cuadernos**, en **Inicio**. Aparece como un cuaderno con un
estudio por ejemplo. Hechos y leídos con Noctis Effect 1.0.0 (18).

Resuelve cada ejemplo a mano primero; abre el cuaderno para comparar, no para saltarte el trabajo.

## Scripts de comprobación (`verify/`)

La comprobación independiente del autor de los resultados a mano: Python + NumPy, escritos desde las
ecuaciones de cada lab. Cada `*.py` imprime los valores que usa el libro; el `*.out` es su salida.

## Licencia

MIT (scripts y cuadernos). El texto del libro es © 2026 Yonatan Ali Rodríguez Arias y no forma parte de este repositorio.
