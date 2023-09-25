## Empty place cebledrum detection

### Запуск контейнера
sudo docker run --name rtsp --runtime nvidia --gpus all -p 5000:5000 rtsp

### Запрос
http://192.168.0.4:5000/?address=<rtsp_address>
