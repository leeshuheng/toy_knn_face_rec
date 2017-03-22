# toy_knn_face_rec

## Run:

cmake .

make

./tst ~/data/att_faces/a.txt

## Create pdf(Chinese)

pdflatex -shell-escape ./blog43.tex

bibtex blog43.aux

pdflatex -shell-escape ./blog43.tex

pdflatex -shell-escape ./blog43.tex
