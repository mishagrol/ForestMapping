docker run --rm -p 8891:8891 -v $PWD:/app/ forest jupyter-notebook --no-browser\
     --NotebookApp.token=SecretToken --port 8891 --ip 0.0.0.0 --allow-root
