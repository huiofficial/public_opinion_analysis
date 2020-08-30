# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import Request
from wechat.items import WechatItem
import re
import time
import random
from datetime import datetime


class WechatSpider(scrapy.Spider):
    name = "wechat"
    mediaName = '微信'
    reg_filter = re.compile(r'<[^>]+>')

    def __init__(self, page='10', *args, **kwargs):
        super(WechatSpider, self).__init__(*args, **kwargs)
        self.page = int(page)

    def start_requests(self):
        for key in self.keyword:
            for i in range(self.page):
                url = 'http://weixin.sogou.com/weixin?query=' + key + '&type=2&page=' + str(i + 1)
                time.sleep(random.randint(1, 4))
                yield Request(url=url, meta={"keyword": key})

    def parse(self, response):
        titles = response.xpath('//div[@class="txt-box"]/h3/a').extract()
        urls = response.xpath('//div[@class="txt-box"]/h3/a/@data-share').extract()
        abstracts = response.xpath('//p[@class="txt-info"]').extract()
        dates = response.css('.s2').extract()
        len_items = min(map(len, [titles, urls, abstracts, dates]))
        for i in range(len_items):
            item = WechatItem()
            title = titles[i]
            item['title'] = self.reg_filter.sub('', title)
            abstract = abstracts[i]
            item['abstract'] = self.reg_filter.sub('', abstract)
            item['url'] = urls[i]
            date = dates[i]
            date = re.search(r"timeConvert\D+(\d+)", date)
            if date is None:
                item['date'] = ''
            else:
                date = date.group(1)
                item['date'] = datetime.fromtimestamp(int(date)).strftime('%Y-%m-%d')
            item['mediaName'] = self.mediaName
            item['keyword'] = response.meta["keyword"]
            yield item
