# Get CIFAR10

powershell -command "Invoke-webRequest 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' -O 'cifar-10-python.tar.gz'"
#wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
