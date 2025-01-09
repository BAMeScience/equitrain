from pathlib import Path

import requests
from bs4 import BeautifulSoup

# https://github.com/IntelLabs/matsciml/blob/main/matsciml/datasets/alexandria/api.py

# ? necessary?
__author__ = 'Yuan Chiang, Janosh Riebesell'
__date__ = '2024-09-04'

base_url = 'https://alexandria.icams.rub.de/data'


# Alexandria provides materials of three different dimensions
config_urls = {
    '3D': base_url + '/pbe/geo_opt_paths/',
    '2D': base_url + '/pbe_2d/geo_opt_paths/',
    '1D': base_url + '/pbe_1d/geo_opt_paths/',
}

# %%
for config, url in config_urls.items():
    response = requests.get(url, timeout=5)

    # Parse all links on the page
    soup = BeautifulSoup(response.text, 'html.parser')
    all_links = soup.find_all('a', href=True)
    file_links = [
        link['href'] for link in all_links if link['href'].endswith('.json.bz2')
    ]

    print(f'{config}: {len(file_links)} files found')

    save_dir = Path(config)
    save_dir.mkdir(exist_ok=True)

    for idx, file_name in enumerate(file_links, start=1):
        fpath = save_dir / file_name
        if not fpath.exists():
            file_url = url + file_name
            print(f'Downloading {idx}/{len(file_links)}: {file_name}')

            file_response = requests.get(file_url, timeout=5)
            with fpath.open('wb') as file:
                file.write(file_response.content)

    print(f'Downloaded {len(file_links)} files to {save_dir}')
