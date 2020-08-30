from scrapy import cmdline
cmdline.execute('scrapy crawl baidu -o crawled_data.json'.split())