docker run --rm -p 8890:8890 -v $PWD:/home/ mmann1123/gw_pygis jupyter notebook --no-browser\
     --NotebookApp.token=SecretToken --port 8890 --ip 0.0.0.0 --allow-root
