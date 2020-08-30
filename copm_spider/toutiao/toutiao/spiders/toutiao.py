import scrapy
from selenium import webdriver
import time
import numpy as np
from scrapy.selector import Selector
from urllib.parse import quote
from toutiao.items import ToutiaoItem
import re

class ToutiaoSpider(scrapy.Spider):
    name = "toutiao"
    mediaName = "头条"
    reg_filter = re.compile(r'<[^>]+>')
    def __init__(self, page='5', *args, **kwargs):
        super(ToutiaoSpider, self).__init__(*args, **kwargs)
        self.page = int(page)

    def start_requests(self):
        for keyword in self.keyword:
            search_result_url = 'https://www.toutiao.com/search/?keyword={}'.format(quote(keyword))
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument(
                'user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36"')
            driver = webdriver.Chrome(chrome_options=chrome_options)
            driver.get(search_result_url)
            time.sleep(np.random.randint(3, 6))
            for i in range(self.page):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(np.random.randint(3, 6))

            url_tails = Selector(text=driver.page_source).css('.title-box').re(r'a class="link title" target="_blank" href="(\S+)"')
            url_head = 'https://www.toutiao.com'
            driver.quit()
            
            for url_tail in url_tails:
                yield scrapy.Request(url=url_head + url_tail, meta={"keyword": keyword}, callback=self.parse)

    def parse(self, response):
        # self.log(response.text)
        title = response.css('.article-title::text').extract_first()
        date = response.css('.article-sub span+ span').extract_first()
        content = ''.join(response.css('.article-content p::text').extract())
        abstract = content[:140]
        if title is not None and date is not None and abstract is not None:
            title = title.strip()
            date = date.strip()
            date = self.reg_filter.sub('', date)
            abstract = abstract.strip()
            if title != '' and date != '' and abstract != '':
                item = ToutiaoItem()
                item["title"] = title
                item["abstract"] = abstract
                item["url"] = response.url
                item["date"] = date
                item["content"] = content
                item["mediaName"] = self.mediaName
                item["keyword"] = response.meta["keyword"]
                yield item

