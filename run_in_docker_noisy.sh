docker run --rm -p 16433:8888 -v $PWD:/home/ -v /mnt/bulky2/mgasanov:/mnt/bulky2/mgasanov mmann1123/gw_pygis jupyter notebook --no-browser\
     --NotebookApp.token=SecretToken --port 8888 --ip 0.0.0.0 --allow-root
