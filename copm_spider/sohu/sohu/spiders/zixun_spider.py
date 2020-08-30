import os
import re
import scrapy
from sohu.items import ZixunItem


class ZixunSpider(scrapy.Spider):
    name = "zixun"
    mediaName = '搜狐资讯'
    
    def __init__(self, *args, **kwargs):
        super(ZixunSpider, self).__init__(*args, **kwargs)
        self.keyword = []
        with open('/var/run/copm_spider/keyword.txt', 'r') as keyword_file:
            for line in keyword_file.readlines():
                self.keyword.append(line.strip()) 

    def start_requests(self):
        for i in range(1, 1000):
            url = "https://zixun.focus.cn/" + str(i) + "/"
            yield scrapy.Request(url=url, callback=self.parse_zixun_list)

    def parse_zixun_list(self, response):
        if response.status != 200:
            print("zixun list " + response.url + " is EMPTY!")
            return
        zixun_list = response.css(".news-list-detail-tlt")
        for index, zixun in enumerate(zixun_list):
            curr_zixun_url = zixun.xpath("./@href").extract_first()
            curr_zixun_title = zixun.xpath("./@title").extract_first()
            curr_zixun_detail_page = response.urljoin(curr_zixun_url)
            yield scrapy.Request(curr_zixun_detail_page, callback=self.parse_zixun_detail)

    def parse_zixun_detail(self, response):
        if response.status != 200:
            print("zixun detail " + response.url + " is EMPTY!")
            return

        def extract_with_css(query):
            return response.css(query).xpath('string(.)').extract_first().strip()

        title = extract_with_css("h1")
        publisher = extract_with_css(".info-source a")
        pub_time = extract_with_css(".info-source span+ span")
        abst = extract_with_css(".abstract")
        # content = re.sub(r'</?\w+[^>]*>', '', extract_with_css(".abstract+ .info-content p"))
        content = response.css(".abstract+ .info-content p").xpath("string(.)").extract()
        content = ''.join(content)
        content = re.sub(r'</?\w+[^>]*>', '', content)
        #  content = ''.join(content)
        detail = {
            "url": response.url,
            "origin": "sohu",
            "publisher": publisher,
            "time": pub_time,
            "title": title,
            "abstraction": abst,
            "content": content,
            "mediaName": self.mediaName,
        }
        item = ZixunItem(detail)
        for key in self.keyword:
            if item["content"].find(key) != -1:
                item['keyword'] = key
                yield item
                break
        self.log("Page %s saved" % response.url)
