
for filename in ./data/*.csv;
do
   source venv/bin/activate
   python3 src/main.py --file "${filename}"
done