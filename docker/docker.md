To install docker run in terminal:

````
sudo apt-get update
````
and one of the following which works for you
````
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo apt-get install docker-engine -y
sudo apt-get install docker.io
````
To change permission to use simple `make build` 
instead of `sudo make build` run:

````
newgrp docker
sudo usermod -aG docker $USER
````

