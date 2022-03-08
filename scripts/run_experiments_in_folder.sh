# Starts experiments for all *.jsons in the folder defined by arg1
jsons=$(find "$1" -name "*.json" -maxdepth 10)
for entry in $jsons
do
    echo "Start experiment for $entry"
    python main.py $entry &
    sleep 1
done
