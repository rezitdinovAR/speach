# Мануал по взаимодействию с сервером
## Первые шаги
### Подключение
```bash
ssh user_name@server_ip -p server_port
server_password
```

### Просмотр доступных видеокарт
```bash
sudo lshw -C display
```

### Просмотр доступной оперативной памяти
```bash
free
```

### Просмотр информации о CPU
```bash
lscpu
```

### Посмотреть доступное место на ssd
```bash
df -h
```

### Увеличение диска Ubuntu 18.04 в панели управления
#### Для начала необходимо выполнить сканирование новой конфигурации и передать данные ядру ОС:
```bash
sudo echo 1 > /sys/block/sda/device/rescan
sudo echo 1 | sudo tee /sys/block/sda/device/rescan
```
#### Далее запустите утилиту parted, которая предназначена для управления жесткими дисками:
```bash
sudo parted
```
#### С помощью опции p выведите таблицу разделов:
```bash
(parted) p
```
#### Результат выглядит следующим образом:
```bash
Model: QEMU QEMU HARDDISK (scsi)
Disk /dev/sda: 107GB
Sector size (logical/physical): 512B/512B
Partition Table: gpt
Disk Flags: 

Number  Start   End     Size    File system  Name  Flags
 1      1049kB  538MB   537MB   fat32              boot, esp
 2      538MB   1612MB  1074MB  ext4
 3      1612MB  68.7GB  67.1GB
 ```
#### С помощью следующей команды измените размер раздела, указав его номер:
```bash
(parted) resizepart <номер>
(parted) resizepart 3
```
#### Появится запрос о новом размере системы, введите объем, которой вы запомнили ранее:
```bash
End?  [XX.XGB]? YY.YGB
End?  [68.7GB]? 107GB
```
#### На этом работа с утилитой parted закончена, закройте ее:
```bash
(parted) quit
Information: You may need to update /etc/fstab.
```
#### Передайте ядру операционной системы Linux информацию об изменениях, указав имя устройства и номер раздела:
```bash
sudo pvresize <имя_устройства><номер_раздела>
sudo pvresize /dev/sda3
```
#### Измените логический том:
```bash
sudo lvextend -r -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv
```
#### Проверить, что винчестер расширен корректно, выполните следующую команду:
```bash
df -h
```

### Обновление 
```bash
sudo apt update;
sudo apt upgrade -y;
```

### Установка необходимых пакетов
```bash
sudo apt install git nginx gunicorn 
```

## Безопасность
### Добавление ssh ключа на сервер
```bash
ssh-copy-id username@remote_host
```

### Отключение аутентификации с помощью пароля на сервере
```bash
sudo nano /etc/ssh/sshd_config
```
Добавить
```
PasswordAuthentication no
```
Перезапустить службу 
```bash
sudo service ssh restart
```


## Установка python
### Method 1 – Install Python 3.11 on Ubuntu from deadsnakes PPA
#### Install the required packages
```bash
sudo apt install wget build-essential libncursesw5-dev libssl-dev \
libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
```
```bash
sudo apt install software-properties-common -y
```
#### Download python 3.11
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
```
#### Установка python3.11
```bash
sudo apt install python3.11
```


### Method 2 – Install Python 3.11 on Ubuntu from source.
#### Install the required packages
```bash
sudo apt update && sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev pkg-config -y
```
#### Download python 3.11
```bash 
wget https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tgz
```
#### Установка python3.11
```bash
tar -xf Python-3.11.*.tgz
```
```bash
cd Python-3.11.*/
```
```bash
./configure --enable-optimizations
```
```bash
make -j $(nproc)
```
```bash
sudo make altinstall
```
```bash
python3.11 --version
```

## Установка Anaconda
### Download the latest shell script to Ubuntu 18.04
```bash 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```
### Make the miniconda installation script executable
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
```
### Run miniconda installation script on Ubuntu 18.04
```bash
./Miniconda3-latest-Linux-x86_64.sh
```
### Create and activate an conda environment
```bash
conda create -n newenv
```
### delete conda environment
```bash
conda remove -n ENV_NAME --all
```

## Установка JupiterLab
### Установка пакетов через pip
```bash
sudo apt install python3-pip
```
```bash
pip3 install --upgrade setuptools
pip3 install ez_setup
```
```bash
conda install jupyter
```

## Установка, запуск и подключение к Jupyter Notebook на удаленном сервере
### Создание туннелей SSH в macOS или Linux
```bash
ssh -L 8888:localhost:8888 sammy@your_server_ip
```
### Change Interpreter in Jupyter notebook
```bash
python -m ipykernel install --user --name <kernel_name> --display-name "<Name_to_display>"
```
### Запуск jupyter notebook
```bash
jupyter notebook
```
### Подключение к jupyter notebook с локального компа
```bash
http://localhost:8000/
```

## This gist contains instructions about cuda v11.8 and cudnn 8.7 installation in Ubuntu 18.04

### If you have previous installation remove it first.
```bash
sudo apt-get purge *nvidia* -y
sudo apt remove --autoremove nvidia-* -y
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

### Verify your gpu cuda
```bash
lspci | grep -i nvidia
```
### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
```bash
gcc --version
```
### System update
```bash
sudo apt-get update
sudo apt-get upgrade
```

### Install other import packages
```bash
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```

### CUDA Toolkit 11.8 Downloads
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu1804-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Cudnn 8.7 Downloads
```bash
CUDNN_TAR_FILE="cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
sudo wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
sudo tar -xvf ${CUDNN_TAR_FILE}
sudo mv cudnn-linux-x86_64-8.7.0.84_cuda11-archive cuda
```

### Посмотреть куда установились пакеты 
У меня возникали с этим проблемы, это нужно для выполнения следующих команд
```bash
dpkg -L libcudnn8
```

### Copy the following files into the cuda toolkit directory.
```bash
sudo cp -P /usr/include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp -P /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.8/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.8/lib64/
sudo chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*
```
### Перезапустить сервер
```bash
sudo reboot
```
### Finally, to verify the installation, check (maybe the Cuda version in nvidia-smi and nvcc is different)
```bash
sudo apt install nvidia-cuda-toolkit
nvidia-smi
nvcc -V
```
### install conda package
```bash
conda install -c anaconda cudnn
conda install -c anaconda cudatoolkit
```
#### Если будут непонятные ошибки, как было у меня, то 
```bash
conda clean --all
conda update -n base conda
```

## Install pytorch 2+
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Download from yandex disk
```bash
curl -H "Authorization: OAuth YANDEX_DISK_TOKEN" -L -o current_data.rar "https://downloader.disk.yandex.ru/disk/..."
```
### unrar
```bash
unrar e file.rar
```

## nginx
### ssl
```bash
sudo apt-get install certbot
sudo apt install python-certbot-nginx
sudo certbot certonly --webroot -d tat-asr.api.tatarby.tugantel.tatar --webroot-path /home/asr/projects/speach/api

tat-asr.api.tatarby.tugantel.tatar

archive_dir = /etc/letsencrypt/archive/tat-asr.api.tatarby.tugantel.tatar
cert = /etc/letsencrypt/live/tat-asr.api.tatarby.tugantel.tatar/cert.pem
privkey = /etc/letsencrypt/live/tat-asr.api.tatarby.tugantel.tatar/privkey.pem
chain = /etc/letsencrypt/live/tat-asr.api.tatarby.tugantel.tatar/chain.pem
fullchain = /etc/letsencrypt/live/tat-asr.api.tatarby.tugantel.tatar/fullchain.pem


tat-tts.api.tatarby.tugantel.tatar
rus-asr.api.tatarby.tugantel.tatar
rus-tts.api.tatarby.tugantel.tatar
```

sudo nano /etc/nginx/sites-available/default
tts-asr

sudo service nginx restart
