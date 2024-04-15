docker run --rm -p 16555:16555 -v $PWD:/home/ mmann1123/gw_pygis jupyter notebook --no-browser\
     --NotebookApp.token=SecretToken --port 16555 --ip 0.0.0.0 --allow-root
