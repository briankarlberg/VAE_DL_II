
for filename in ./data/*.tsv;
do
   source venv/bin/activate
   python3 src/main.py --file "${filename}"
done