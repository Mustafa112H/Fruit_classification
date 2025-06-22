from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler
import os

#Folder name
os.makedirs('blueberries', exist_ok=True)

keyword = 'natural blueberries'

bing_crawler = BingImageCrawler(storage={'root_dir': 'blueberries'})
bing_crawler.crawl(keyword=keyword, max_num=300, file_idx_offset=1100)



crawler2 = GoogleImageCrawler(storage={'root_dir': 'blueberries'})
crawler2.crawl(keyword=keyword, max_num=300, file_idx_offset=60)
