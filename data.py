from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler
import os

#Folder name
os.makedirs('place_folder', exist_ok=True)

keyword = ('nice Bananas')

bing_crawler = BingImageCrawler(storage={'root_dir': 'folder_name'})
bing_crawler.crawl(keyword=keyword, max_num=300, file_idx_offset=1700)


crawler2 = GoogleImageCrawler(storage={'root_dir': 'folder_name'})
crawler2.crawl(keyword=keyword, max_num=300, file_idx_offset=2700)