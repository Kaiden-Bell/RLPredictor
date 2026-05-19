import requests
from bs4 import BeautifulSoup

r = requests.get('https://liquipedia.net/rocketleague/Aztro', headers={'User-Agent': 'Mozilla/5.0'})
soup = BeautifulSoup(r.text, 'html.parser')
for div in soup.find_all('div', class_='infobox-cell-2'):
    print(div.get_text(strip=True))
