sudo nano /lib/systemd/system/tatar-tts.service

sudo systemctl daemon-reload
sudo systemctl enable tatar-tts
sudo systemctl start tatar-tts
sudo systemctl restart tatar-tts
sudo systemctl status tatar-tts
sudo systemctl stop tatar-tts