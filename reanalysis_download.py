from ftplib import FTP
import os
from pathlib import Path


DOWNLOAD_FOLDER = Path("./american_reanalysis_data/")

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

ftp= FTP('ftp2.psl.noaa.gov')
ftp.login()

ftp.cwd('/Datasets/ncep.reanalysis2/pressure/')
files= ftp.nlst()

for file in files:
    if 'hgt' in file:
        file_path = DOWNLOAD_FOLDER / Path(file)
        print(file_path)
        if not file_path.is_file():
            with open(file_path, 'wb') as fp:
                ftp.retrbinary("RETR " + file, fp.write)

ftp.quit()
