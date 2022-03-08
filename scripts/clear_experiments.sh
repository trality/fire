DIR="results_of_experiments"
if [ -d "$DIR" ]; then
    while true; do
        read -p "Are you sure you want to delete ALL files in $DIR/ ? (Y/n) " yn
        case $yn in
            [Yy]* ) rm -r $DIR; break;;
            * ) exit;;
        esac
    done
else
    echo "There are no experiments to clear"
fi
