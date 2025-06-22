from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler
import os

#Folder name
os.makedirs('pomegranates', exist_ok=True)

keyword = 'fresh promagrante'

bing_crawler = BingImageCrawler(storage={'root_dir': 'pomegranates'})
bing_crawler.crawl(keyword=keyword, max_num=300, file_idx_offset=1900)



crawler2 = GoogleImageCrawler(storage={'root_dir': 'pomegranates'})
crawler2.crawl(keyword=keyword, max_num=300, file_idx_offset=900)
