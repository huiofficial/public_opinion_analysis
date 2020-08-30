from scrapy import cmdline
cmdline.execute('scrapy crawl wechat -o crawled_data.json'.split())